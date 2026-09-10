"""Pixel-only Double-DQN training for the Dodge NG ablation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import random
import signal
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import torch
from torch import Tensor, nn

from dodge.control import PROJECT_ROOT, ControlRuntimeError
from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import (
    NativeBatchEnvironment,
    NativeBatchResult,
    NativePixelBatchResult,
)
from dodge.native.differential import FRAME_HEIGHT, FRAME_WIDTH
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest
from dodge.ng.report import summarize_evaluation
from dodge.ng.telemetry import DashboardTelemetry
from dodge.ng.waypoint import WaypointController, WaypointGrid
from dodge.rl.ppo import PixelFeatureEncoder

PIXEL_DQN_VERSION: Final[int] = 1
PIXEL_DQN_MODEL_TYPE: Final[str] = "DodgePixelDuelingDQN"
PIXEL_PALETTE_MAX: Final[int] = 15
PIXEL_ARCHITECTURES: tuple[str, ...] = (
    "fast",
    "small",
    "current",
    "palette-spatial",
)
PIXEL_SHAPE: tuple[int, int] = (FRAME_HEIGHT, FRAME_WIDTH)
RELEVANCE_GATE_FRAMES: Final[int] = 800
PixelArchitecture = Literal["fast", "small", "current", "palette-spatial"]
PixelExecution = Literal["serial", "parallel"]
PixelBoundary = Literal["legacy", "fast"]
ResetMode = Literal["native-startup", "legacy"]
MacroResetMode = Literal["freeze-lanes", "stop-macro", "legacy-continue"]


@dataclass(frozen=True, slots=True)
class PixelDQNConfig:
    """Pixel DQN hyperparameters and the native waypoint action contract."""

    total_steps: int = 200_000
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_rate: float = 0.00024074661619742944
    weight_decay: float = 0.0001
    gamma: float = 0.99
    n_step: int = 5
    warmup_steps: int = 2_000
    train_frequency: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 500
    target_update_unit: Literal["optimizer", "collection"] = "optimizer"
    reward_scale: float = 1.0
    huber_beta: float = 1.0
    hidden_size: int = 128
    pixel_stack: int = 4
    pixel_architecture: PixelArchitecture = "fast"
    grid_spacing: int = 32
    hold_decisions: int = 8
    steering_tolerance: float = 2.0
    arrival_latching: bool = False
    ban_corner_nodes: bool = False
    corner_node_penalty: float = 0.0
    step_frames: int = 4
    max_episode_steps: int = 2_000
    native_lanes: int = 12
    native_execution: PixelExecution = "parallel"
    native_pixel_boundary: PixelBoundary = "legacy"
    reset_mode: ResetMode = "native-startup"
    macro_reset_mode: MacroResetMode = "freeze-lanes"
    training_lives: int = 1
    life_loss_penalty: float = -64.0
    epsilon_decay_steps: int = 50_000
    epsilon_final: float = 0.05
    checkpoint_every: int = 10_000
    eval_every: int = 10_000
    seed: int = 2_026_0903
    device: str = "cpu"
    torch_threads: int = 8

    def validate(self) -> None:
        positive = (
            self.total_steps,
            self.batch_size,
            self.replay_capacity,
            self.n_step,
            self.warmup_steps,
            self.train_frequency,
            self.gradient_steps,
            self.target_update_interval,
            self.hidden_size,
            self.pixel_stack,
            self.hold_decisions,
            self.max_episode_steps,
            self.native_lanes,
            self.checkpoint_every,
            self.eval_every,
        )
        if any(value < 1 for value in positive):
            raise ValueError("pixel DQN counts and intervals must be positive")
        if self.batch_size > self.replay_capacity:
            raise ValueError("pixel DQN batch size must not exceed replay capacity")
        if self.learning_rate <= 0:
            raise ValueError("pixel DQN learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("pixel DQN weight decay must not be negative")
        if self.target_update_unit not in {"optimizer", "collection"}:
            raise ValueError("pixel target update unit is invalid")
        if not np.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("pixel DQN reward scale must be finite and positive")
        if not np.isfinite(self.huber_beta) or self.huber_beta <= 0:
            raise ValueError("pixel DQN Huber beta must be finite and positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("pixel DQN gamma must be between 0 and 1")
        if not 3 <= self.step_frames <= 5:
            raise ValueError("pixel DQN step frames must be between 3 and 5")
        if self.grid_spacing < 1:
            raise ValueError("pixel DQN grid spacing must be positive")
        if not np.isfinite(self.steering_tolerance) or self.steering_tolerance < 0:
            raise ValueError(
                "pixel DQN steering tolerance must be finite and non-negative"
            )
        if self.steering_tolerance >= self.grid_spacing / 2:
            raise ValueError("pixel DQN steering tolerance must be below half spacing")
        if not isinstance(self.arrival_latching, bool):
            raise ValueError("pixel DQN arrival latching must be a boolean")
        if not isinstance(self.ban_corner_nodes, bool):
            raise ValueError("pixel DQN corner-node policy must be a boolean")
        if not np.isfinite(self.corner_node_penalty) or self.corner_node_penalty > 0:
            raise ValueError(
                "pixel DQN corner-node penalty must be finite and non-positive"
            )
        if not 1 <= self.pixel_stack <= 8:
            raise ValueError("pixel DQN stack must be between 1 and 8")
        if self.pixel_architecture not in PIXEL_ARCHITECTURES:
            raise ValueError("pixel DQN architecture is invalid")
        if self.n_step > 2**8 - 1:
            raise ValueError("pixel DQN n-step horizon must fit replay storage")
        if self.native_execution not in {"serial", "parallel"}:
            raise ValueError("pixel DQN execution must be serial or parallel")
        if self.native_pixel_boundary not in {"legacy", "fast"}:
            raise ValueError("pixel DQN native pixel boundary is invalid")
        if self.reset_mode not in {"native-startup", "legacy"}:
            raise ValueError("pixel DQN reset mode is invalid")
        if self.macro_reset_mode not in {
            "freeze-lanes",
            "stop-macro",
            "legacy-continue",
        }:
            raise ValueError("pixel DQN macro reset mode is invalid")
        if (
            isinstance(self.training_lives, bool)
            or not isinstance(self.training_lives, int)
            or self.training_lives < 1
        ):
            raise ValueError("pixel DQN training lives must be a positive integer")
        if not np.isfinite(self.life_loss_penalty) or self.life_loss_penalty > 0:
            raise ValueError(
                "pixel DQN life-loss penalty must be finite and non-positive"
            )
        if self.epsilon_decay_steps < 0:
            raise ValueError("pixel DQN epsilon decay steps must not be negative")
        if not 0 <= self.epsilon_final < 1:
            raise ValueError("pixel DQN final epsilon must be in [0, 1)")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("pixel DQN device must be cpu, cuda, or auto")
        if self.torch_threads < 0:
            raise ValueError("pixel DQN torch threads must not be negative")

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class DuelingPixelDQN(nn.Module):
    """Dueling Q network whose sole observation is an indexed pixel stack."""

    def __init__(
        self,
        *,
        stack_size: int = 4,
        hidden_size: int = 128,
        architecture: PixelArchitecture = "fast",
        action_count: int = len(ACTION_CHOICES),
    ) -> None:
        super().__init__()
        if action_count < 1:
            raise ValueError("pixel DQN action count must be positive")
        self.stack_size = stack_size
        self.action_count = action_count
        self.pixel_architecture = architecture
        if architecture == "palette-spatial":
            self.features = PaletteSpatialEncoder(stack_size, hidden_size)
        else:
            self.features = PixelFeatureEncoder(
                stack_size=stack_size,
                hidden_size=hidden_size,
                architecture=architecture,
            )
        self.value = nn.Linear(hidden_size, 1)
        self.advantage = nn.Linear(hidden_size, action_count)
        self._initialize_weights()

    def forward(self, observations: Tensor) -> Tensor:
        if observations.dtype != torch.uint8:
            raise ValueError("pixel DQN observations must use uint8 palette indexes")
        features = self.features(observations)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.advantage.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)


class PaletteSpatialEncoder(nn.Module):
    """Encode palette labels without pretending that their numeric ids are ordered."""

    def __init__(self, stack_size: int, hidden_size: int) -> None:
        super().__init__()
        if not 1 <= stack_size <= 8:
            raise ValueError("pixel stack must be between 1 and 8")
        if hidden_size < 1:
            raise ValueError("hidden size must be positive")
        self.stack_size = stack_size
        self.embedding = nn.Embedding(PIXEL_PALETTE_MAX + 1, 4)
        self.convolution = nn.Sequential(
            nn.Conv2d(stack_size * 4, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 4)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden_size),
            nn.ReLU(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        if observations.dtype != torch.uint8:
            raise ValueError("pixel DQN observations must use uint8 palette indexes")
        expected_shape = (self.stack_size, FRAME_HEIGHT, FRAME_WIDTH)
        if observations.ndim != 4 or tuple(observations.shape[1:]) != expected_shape:
            raise ValueError(
                "pixel observations must have shape "
                f"(N, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]})"
            )
        embedded = self.embedding(observations.long())
        batch, stack, height, width, channels = embedded.shape
        features = embedded.permute(0, 1, 4, 2, 3).reshape(
            batch, stack * channels, height, width
        )
        return self.projection(self.convolution(features))


@dataclass(frozen=True, slots=True)
class PixelReplaySample:
    observations: np.ndarray
    actions: np.ndarray
    target_columns: np.ndarray
    target_rows: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    discounts: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    n_steps: np.ndarray


class PixelReplayBuffer:
    """Uniform replay with on-disk frame storage and in-memory references.

    A transition stores references to four palette frames instead of copying
    two four-frame tensors into every checkpoint. The frame file is a bounded
    ring, so checkpoint size remains independent of raster payload size.
    """

    def __init__(
        self,
        capacity: int,
        stack_size: int,
        lane_count: int,
        run_directory: Path,
        *,
        resume: bool = False,
    ) -> None:
        if capacity < 1 or not 1 <= stack_size <= 8 or lane_count < 1:
            raise ValueError("pixel replay dimensions must be positive")
        self.capacity = capacity
        self.stack_size = stack_size
        self.lane_count = lane_count
        self.frame_capacity = capacity * stack_size * 2 + lane_count * stack_size + 1
        self.frame_path = Path(run_directory) / ".pixel-frames.u8"
        expected_bytes = self.frame_capacity * FRAME_HEIGHT * FRAME_WIDTH
        if resume:
            if not self.frame_path.is_file():
                raise ControlRuntimeError(
                    f"pixel replay frame store does not exist: {self.frame_path}"
                )
            if self.frame_path.stat().st_size != expected_bytes:
                raise ValueError("pixel replay frame store size does not match config")
            mode = "r+"
        else:
            if self.frame_path.exists():
                raise FileExistsError(
                    f"pixel replay frame store already exists: {self.frame_path}"
                )
            mode = "w+"
        self.frames = np.memmap(
            self.frame_path,
            dtype=np.uint8,
            mode=mode,
            shape=(self.frame_capacity, FRAME_HEIGHT, FRAME_WIDTH),
        )
        self.frame_generations = np.full(self.frame_capacity, -1, dtype=np.int64)
        self.frame_serial = 0
        self.frame_count = 0
        self.position = 0
        self.size = 0
        self.observation_frames = np.zeros((capacity, stack_size), dtype=np.int64)
        self.next_observation_frames = np.zeros((capacity, stack_size), dtype=np.int64)
        self.actions = np.zeros(capacity, dtype=np.uint8)
        self.target_columns = np.zeros(capacity, dtype=np.uint16)
        self.target_rows = np.zeros(capacity, dtype=np.uint16)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)
        self.n_steps = np.zeros(capacity, dtype=np.uint8)

    def store_stacks(self, stacks: np.ndarray) -> np.ndarray:
        """Store `(N, stack, 128, 128)` frames and return global frame IDs."""
        values = np.asarray(stacks)
        expected = (values.shape[0], self.stack_size, *PIXEL_SHAPE)
        if values.shape != expected or values.dtype != np.uint8:
            raise ValueError(
                "pixel replay stacks must have shape "
                f"(N, {self.stack_size}, {FRAME_HEIGHT}, {FRAME_WIDTH}) and uint8"
            )
        count = values.shape[0] * self.stack_size
        if count == 0:
            return np.empty((0, self.stack_size), dtype=np.int64)
        serials = np.arange(
            self.frame_serial,
            self.frame_serial + count,
            dtype=np.int64,
        )
        slots = serials % self.frame_capacity
        self.frames[slots] = values.reshape(count, FRAME_HEIGHT, FRAME_WIDTH)
        self.frame_generations[slots] = serials
        self.frame_serial += count
        self.frame_count = min(self.frame_capacity, self.frame_count + count)
        return serials.reshape(values.shape[0], self.stack_size)

    def add(
        self,
        observation_frames: np.ndarray,
        action: int,
        target_cell: tuple[int, int],
        reward: float,
        next_observation_frames: np.ndarray,
        discount: float,
        terminated: bool,
        truncated: bool,
        n_steps: int,
    ) -> None:
        observation = np.asarray(observation_frames)
        next_observation = np.asarray(next_observation_frames)
        expected = (self.stack_size,)
        if observation.shape != expected or next_observation.shape != expected:
            raise ValueError("pixel replay frame references have an invalid shape")
        if observation.dtype != np.int64 or next_observation.dtype != np.int64:
            raise ValueError("pixel replay frame references must use int64 IDs")
        if not 0 <= action < len(ACTION_CHOICES):
            raise ValueError("pixel replay action is outside the nine-action space")
        if n_steps < 1 or n_steps > 2**8 - 1:
            raise ValueError("pixel replay n-step value is invalid")
        index = self.position
        self.observation_frames[index] = observation
        self.next_observation_frames[index] = next_observation
        self.actions[index] = action
        self.target_columns[index] = target_cell[0]
        self.target_rows[index] = target_cell[1]
        self.rewards[index] = reward
        self.discounts[index] = discount
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.n_steps[index] = n_steps
        self.position = (index + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int, rng: np.random.Generator) -> PixelReplaySample:
        if not 1 <= batch_size <= self.size:
            raise ValueError("pixel replay sample size is invalid")
        indices = rng.integers(0, self.size, size=batch_size)
        observation_refs = self.observation_frames[indices]
        next_observation_refs = self.next_observation_frames[indices]
        observations = self._load_stacks(observation_refs)
        next_observations = self._load_stacks(next_observation_refs)
        return PixelReplaySample(
            observations=observations,
            actions=self.actions[indices].copy(),
            target_columns=self.target_columns[indices].copy(),
            target_rows=self.target_rows[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_observations=next_observations,
            discounts=self.discounts[indices].copy(),
            terminated=self.terminated[indices].copy(),
            truncated=self.truncated[indices].copy(),
            n_steps=self.n_steps[indices].copy(),
        )

    def state_dict(self) -> dict[str, object]:
        self.frames.flush()
        return {
            "capacity": self.capacity,
            "stack_size": self.stack_size,
            "lane_count": self.lane_count,
            "frame_capacity": self.frame_capacity,
            "frame_file": self.frame_path.name,
            "frame_serial": self.frame_serial,
            "frame_count": self.frame_count,
            "frame_generations": self.frame_generations.copy(),
            "position": self.position,
            "size": self.size,
            "observation_frames": self.observation_frames.copy(),
            "next_observation_frames": self.next_observation_frames.copy(),
            "actions": self.actions.copy(),
            "target_columns": self.target_columns.copy(),
            "target_rows": self.target_rows.copy(),
            "rewards": self.rewards.copy(),
            "discounts": self.discounts.copy(),
            "terminated": self.terminated.copy(),
            "truncated": self.truncated.copy(),
            "n_steps": self.n_steps.copy(),
        }

    def load_state_dict(self, value: object) -> None:
        if not isinstance(value, dict):
            raise ValueError("pixel replay checkpoint state must be an object")
        for name, expected in (
            ("capacity", self.capacity),
            ("stack_size", self.stack_size),
            ("lane_count", self.lane_count),
            ("frame_capacity", self.frame_capacity),
        ):
            if value.get(name) != expected:
                raise ValueError(f"pixel replay checkpoint {name} does not match")
        if value.get("frame_file") != self.frame_path.name:
            raise ValueError("pixel replay checkpoint frame file does not match")
        frame_serial = value.get("frame_serial")
        frame_count = value.get("frame_count")
        position = value.get("position")
        size = value.get("size")
        if not all(
            isinstance(item, int)
            for item in (frame_serial, frame_count, position, size)
        ):
            raise ValueError("pixel replay checkpoint progress is invalid")
        if not 0 <= frame_count <= self.frame_capacity:
            raise ValueError("pixel replay checkpoint frame count is invalid")
        if not 0 <= position < self.capacity or not 0 <= size <= self.capacity:
            raise ValueError("pixel replay checkpoint ring progress is invalid")
        arrays: tuple[tuple[str, np.ndarray, tuple[int, ...]], ...] = (
            (
                "frame_generations",
                self.frame_generations,
                (self.frame_capacity,),
            ),
            (
                "observation_frames",
                self.observation_frames,
                self.observation_frames.shape,
            ),
            (
                "next_observation_frames",
                self.next_observation_frames,
                self.next_observation_frames.shape,
            ),
            ("actions", self.actions, self.actions.shape),
            ("target_columns", self.target_columns, self.target_columns.shape),
            ("target_rows", self.target_rows, self.target_rows.shape),
            ("rewards", self.rewards, self.rewards.shape),
            ("discounts", self.discounts, self.discounts.shape),
            ("terminated", self.terminated, self.terminated.shape),
            ("truncated", self.truncated, self.truncated.shape),
            ("n_steps", self.n_steps, self.n_steps.shape),
        )
        for name, destination, shape in arrays:
            source = value.get(name)
            if not isinstance(source, np.ndarray) or source.shape != shape:
                raise ValueError(f"pixel replay checkpoint field {name} is invalid")
            if source.dtype != destination.dtype:
                raise ValueError(
                    f"pixel replay checkpoint field {name} dtype is invalid"
                )
            destination[...] = source
        self.frame_serial = frame_serial
        self.frame_count = frame_count
        self.position = position
        self.size = size

    def close(self) -> None:
        self.frames.flush()

    def _load_stacks(self, references: np.ndarray) -> np.ndarray:
        if references.ndim != 2 or references.shape[1] != self.stack_size:
            raise ValueError("pixel replay frame reference shape is invalid")
        slots = references % self.frame_capacity
        generations = self.frame_generations[slots]
        if not np.array_equal(generations, references):
            raise ControlRuntimeError("pixel replay frame history was overwritten")
        return np.asarray(self.frames[slots], dtype=np.uint8).copy()


@dataclass(frozen=True, slots=True)
class _PixelPendingTransition:
    observation_frames: np.ndarray
    action: int
    target_cell: tuple[int, int]
    reward: float
    next_observation_frames: np.ndarray
    terminated: bool
    truncated: bool


class PixelNStepAccumulator:
    """Aggregate pixel transitions without crossing terminal boundaries."""

    def __init__(
        self,
        lane_count: int,
        n_step: int,
        gamma: float,
        replay: PixelReplayBuffer,
    ) -> None:
        if lane_count < 1 or n_step < 1:
            raise ValueError("pixel n-step dimensions must be positive")
        self.n_step = n_step
        self.gamma = gamma
        self.replay = replay
        self.queues: list[deque[_PixelPendingTransition]] = [
            deque() for _ in range(lane_count)
        ]

    def append(
        self,
        lane: int,
        observation_frames: np.ndarray,
        action: int,
        target_cell: tuple[int, int],
        reward: float,
        next_observation_frames: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if not 0 <= lane < len(self.queues):
            raise ValueError("pixel n-step lane is invalid")
        self.queues[lane].append(
            _PixelPendingTransition(
                observation_frames.copy(),
                action,
                target_cell,
                reward,
                next_observation_frames.copy(),
                terminated,
                truncated,
            )
        )
        if len(self.queues[lane]) >= self.n_step:
            self._emit(lane)
        if terminated or truncated:
            while self.queues[lane]:
                self._emit(lane)

    def state_dict(self) -> dict[str, object]:
        return {
            "n_step": self.n_step,
            "gamma": self.gamma,
            "queues": [
                [
                    {
                        "observation_frames": item.observation_frames.copy(),
                        "action": item.action,
                        "target_cell": list(item.target_cell),
                        "reward": item.reward,
                        "next_observation_frames": item.next_observation_frames.copy(),
                        "terminated": item.terminated,
                        "truncated": item.truncated,
                    }
                    for item in queue
                ]
                for queue in self.queues
            ],
        }

    def _emit(self, lane: int) -> None:
        queue = self.queues[lane]
        if not queue:
            return
        reward = 0.0
        discount = 1.0
        horizon = 0
        last: _PixelPendingTransition | None = None
        for item in queue:
            reward += discount * item.reward
            discount *= self.gamma
            horizon += 1
            last = item
            if item.terminated or item.truncated or horizon >= self.n_step:
                break
        if last is None:
            raise ControlRuntimeError("pixel n-step queue emitted without a transition")
        boundary = last.terminated or last.truncated
        first = queue[0]
        self.replay.add(
            first.observation_frames,
            first.action,
            first.target_cell,
            reward,
            last.next_observation_frames,
            0.0 if boundary else self.gamma**horizon,
            last.terminated,
            last.truncated,
            horizon,
        )
        queue.popleft()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("pixel DQN requested cuda but cuda is unavailable")
    return torch.device(device)


class _PixelProfiler:
    """Bounded opt-in profiler for one training-loop window."""

    def __init__(
        self,
        directory: Path | None,
        device: torch.device,
        max_steps: int,
    ) -> None:
        self.directory = Path(directory) if directory is not None else None
        self.device = device
        self.max_steps = max_steps
        self.captured_steps = 0
        self._profiler: object | None = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self.directory is not None and self.max_steps > 0

    def start(self) -> None:
        if not self.enabled or self._profiler is not None:
            return
        from torch.profiler import ProfilerActivity

        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        assert self.directory is not None
        self.directory.mkdir(parents=True, exist_ok=True)
        self._profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self._profiler.start()  # type: ignore[union-attr]

    def step(self) -> None:
        if self._profiler is None or self._closed:
            return
        self._profiler.step()  # type: ignore[union-attr]
        self.captured_steps += 1
        if self.captured_steps >= self.max_steps:
            self.close()

    def close(self) -> None:
        if self._profiler is None or self._closed:
            return
        assert self.directory is not None
        profiler = self._profiler
        profiler.stop()  # type: ignore[union-attr]
        trace_path = self.directory / "trace.json"
        summary_path = self.directory / "summary.txt"
        profiler.export_chrome_trace(str(trace_path))  # type: ignore[union-attr]
        sort_by = (
            "self_cuda_time_total"
            if self.device.type == "cuda"
            else "self_cpu_time_total"
        )
        table = profiler.key_averages().table(  # type: ignore[union-attr]
            sort_by=sort_by,
            row_limit=40,
        )
        summary_path.write_text(table + "\n", encoding="utf-8")
        _write_json(
            self.directory / "profile.json",
            {
                "schema_version": 1,
                "kind": "dodge_ng_pixel_dqn_profile",
                "device": str(self.device),
                "activities": [
                    "cpu",
                    "cuda",
                ]
                if self.device.type == "cuda"
                else ["cpu"],
                "requested_steps": self.max_steps,
                "captured_steps": self.captured_steps,
                "trace_file": trace_path.name,
                "summary_file": summary_path.name,
            },
        )
        self._closed = True


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _epsilon(config: PixelDQNConfig, step: int) -> float:
    decay_steps = config.epsilon_decay_steps or max(config.total_steps // 2, 1)
    progress = min(1.0, step / decay_steps)
    return 1.0 + progress * (config.epsilon_final - 1.0)


def _consume_training_life(
    lives_remaining: np.ndarray,
    episode_steps: np.ndarray,
    lane: int,
    training_lives: int,
) -> bool:
    if lives_remaining.ndim != 1 or episode_steps.ndim != 1:
        raise ValueError("pixel DQN life state must be one-dimensional")
    if not 0 <= lane < len(lives_remaining) or len(episode_steps) != len(
        lives_remaining
    ):
        raise ValueError("pixel DQN life state lane is invalid")
    if int(lives_remaining[lane]) < 1:
        raise ValueError("pixel DQN life state is already exhausted")
    lives_remaining[lane] -= 1
    final_loss = int(lives_remaining[lane]) == 0
    if final_loss:
        lives_remaining[lane] = training_lives
    episode_steps[lane] = 0
    return final_loss


def _validate_pixel_result(
    result: NativeBatchResult | NativePixelBatchResult,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = result.pixels
    positions = result.player_positions
    if pixels is None or positions is None:
        raise ControlRuntimeError(
            "pixel DQN requires native pixels and player-coordinate plumbing"
        )
    expected_pixels = (result.lane_count, FRAME_HEIGHT, FRAME_WIDTH)
    if pixels.shape != expected_pixels or pixels.dtype != np.uint8:
        raise ControlRuntimeError(
            "native pixel observations have unexpected shape or dtype: "
            f"expected {expected_pixels} and uint8, got {pixels.shape} and "
            f"{pixels.dtype}"
        )
    if pixels.size and (int(pixels.min()) < 0 or int(pixels.max()) > PIXEL_PALETTE_MAX):
        raise ControlRuntimeError(
            "native pixel observations contain an invalid palette index"
        )
    expected_positions = (result.lane_count, 2)
    if positions.shape != expected_positions or positions.dtype != np.float32:
        raise ControlRuntimeError(
            "native pixel player positions have unexpected shape or dtype: "
            f"expected {expected_positions} and float32, got {positions.shape} and "
            f"{positions.dtype}"
        )
    if not np.isfinite(positions).all():
        raise ControlRuntimeError("native pixel player positions must be finite")
    return pixels, positions


def _step_pixel_batch(
    environment: NativeBatchEnvironment,
    actions: np.ndarray,
    config: PixelDQNConfig,
) -> NativeBatchResult | NativePixelBatchResult:
    if config.native_pixel_boundary == "fast":
        return environment.step_pixels(actions)
    return environment.step_batch(actions)


def _initial_pixel_stacks(
    pixels: np.ndarray,
    stack_size: int,
) -> np.ndarray:
    return np.repeat(pixels[:, None, :, :], stack_size, axis=1).copy()


def _advance_pixel_stacks(
    current: np.ndarray,
    next_pixels: np.ndarray,
) -> np.ndarray:
    next_stacks = np.empty_like(current)
    if current.shape[1] > 1:
        next_stacks[:, :-1] = current[:, 1:]
    next_stacks[:, -1] = next_pixels
    return next_stacks


def _reset_pixels(
    environment: NativeBatchEnvironment,
    seeds: np.ndarray,
    config: PixelDQNConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.reset_mode == "native-startup":
        result = environment.reset_batch_with_startup(seeds)
    else:
        result = environment.reset_batch(seeds)
    pixels, positions = _validate_pixel_result(result)
    return _initial_pixel_stacks(pixels, config.pixel_stack), positions.copy()


def _reset_pixel_lanes(
    environment: NativeBatchEnvironment,
    lanes: np.ndarray,
    seeds: np.ndarray,
    config: PixelDQNConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.reset_mode == "native-startup":
        result = environment.reset_lanes_with_startup(lanes, seeds)
    else:
        result = environment.reset_lanes(lanes, seeds)
    pixels, positions = _validate_pixel_result(result)
    return _initial_pixel_stacks(pixels, config.pixel_stack), positions.copy()


def _choose_actions(
    model: DuelingPixelDQN,
    observations: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    with torch.inference_mode():
        values = model(torch.from_numpy(observations).to(device))
    greedy_tensor = values.argmax(dim=1)
    top_values = values.topk(k=min(2, values.shape[1]), dim=1).values
    greedy = greedy_tensor.detach().cpu().numpy().astype(np.uint8)
    greedy_before_exploration = greedy.copy()
    random_mask = rng.random(len(greedy)) < epsilon
    random_actions = rng.integers(0, len(ACTION_CHOICES), size=len(greedy))
    greedy[random_mask] = random_actions[random_mask]
    diagnostics = {
        "random_action_fraction": float(random_mask.mean()),
        "greedy_action_entropy": _categorical_entropy(
            greedy_before_exploration,
            len(ACTION_CHOICES),
        ),
        "q_action_margin_mean": float(
            (top_values[:, 0] - top_values[:, 1]).mean().item()
        ),
        **{
            f"action_count_{name}": float(np.count_nonzero(greedy == index))
            for index, name in enumerate(ACTION_CHOICES)
        },
    }
    return greedy, diagnostics


def _categorical_entropy(values: np.ndarray, category_count: int) -> float:
    counts = np.bincount(values.astype(np.int64), minlength=category_count)
    probabilities = counts[counts > 0] / max(1, len(values))
    return float(-(probabilities * np.log(probabilities)).sum())


def _learn_step(
    model: DuelingPixelDQN,
    target_model: DuelingPixelDQN,
    optimizer: torch.optim.Optimizer,
    replay: PixelReplayBuffer,
    config: PixelDQNConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    sample_started = time.perf_counter()
    sample = replay.sample(config.batch_size, rng)
    replay_sample_seconds = time.perf_counter() - sample_started
    transfer_started = time.perf_counter()
    observations = torch.from_numpy(sample.observations).to(device)
    actions = torch.from_numpy(sample.actions.astype(np.int64)).to(device)
    rewards = torch.from_numpy(sample.rewards).to(device)
    next_observations = torch.from_numpy(sample.next_observations).to(device)
    discounts = torch.from_numpy(sample.discounts).to(device)
    host_to_device_seconds = time.perf_counter() - transfer_started
    learner_started = time.perf_counter()
    q_values = model(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = model(next_observations).argmax(dim=1)
        next_values = (
            target_model(next_observations)
            .gather(1, next_actions.unsqueeze(1))
            .squeeze(1)
        )
        targets = rewards + discounts * next_values
    loss = nn.functional.smooth_l1_loss(
        q_values,
        targets,
        beta=config.huber_beta,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    )
    optimizer.step()
    learner_compute_seconds = time.perf_counter() - learner_started
    return {
        "loss": float(loss.item()),
        "q_mean": float(q_values.detach().mean().item()),
        "target_mean": float(targets.mean().item()),
        "td_error": float((q_values.detach() - targets).abs().mean().item()),
        "gradient_norm": gradient_norm,
        "gradient_clipped": float(gradient_norm > 10.0),
        "sample_reward_mean": float(sample.rewards.mean()),
        "sample_reward_min": float(sample.rewards.min()),
        "sample_reward_max": float(sample.rewards.max()),
        "sample_terminal_fraction": float(sample.terminated.mean()),
        "sample_n_step_mean": float(sample.n_steps.mean()),
        "replay_sample_seconds": replay_sample_seconds,
        "host_to_device_seconds": host_to_device_seconds,
        "learner_compute_seconds": learner_compute_seconds,
    }


def _next_training_seed(
    training_seeds: tuple[int, ...],
    cursor: int,
) -> tuple[int, int]:
    if not training_seeds:
        raise ValueError("pixel DQN requires training seeds")
    return training_seeds[cursor % len(training_seeds)], cursor + 1


def _controller_for_config(config: PixelDQNConfig) -> WaypointController:
    grid = WaypointGrid(
        config.grid_spacing,
        ban_corner_nodes=config.ban_corner_nodes,
    )
    return WaypointController(
        grid,
        tolerance=config.steering_tolerance,
        arrival_latching=config.arrival_latching,
    )


def _collect_macro_transition(
    environment: NativeBatchEnvironment,
    current_stacks: np.ndarray,
    current_frame_refs: np.ndarray,
    current_positions: np.ndarray,
    controller: WaypointController,
    model: DuelingPixelDQN,
    config: PixelDQNConfig,
    episode_steps: np.ndarray,
    episode_seeds: np.ndarray,
    lives_remaining: np.ndarray,
    training_seeds: tuple[int, ...],
    seed_cursor: int,
    accumulator: PixelNStepAccumulator,
    replay: PixelReplayBuffer,
    rng: np.random.Generator,
    device: torch.device,
    global_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, dict[str, float]]:
    lane_count = len(current_stacks)
    if (
        current_stacks.shape
        != (lane_count, config.pixel_stack, FRAME_HEIGHT, FRAME_WIDTH)
        or current_frame_refs.shape != (lane_count, config.pixel_stack)
        or current_positions.shape != (lane_count, 2)
        or episode_steps.shape != (lane_count,)
        or episode_seeds.shape != (lane_count,)
        or lives_remaining.shape != (lane_count,)
    ):
        raise ValueError("pixel DQN lane state shapes do not match observations")
    observations = current_stacks.copy()
    observation_refs = current_frame_refs.copy()
    epsilon = _epsilon(config, global_step)
    waypoint_actions, action_diagnostics = _choose_actions(
        model,
        observations,
        epsilon,
        rng,
        device,
    )
    raw_target_cells = [
        controller.grid.neighbor_cell(
            controller.grid.nearest_cell(
                float(current_positions[lane, 0]),
                float(current_positions[lane, 1]),
            ),
            int(action),
        )
        for lane, action in enumerate(waypoint_actions)
    ]
    target_cells = [
        controller.grid.target_cell_for_action(
            float(current_positions[lane, 0]),
            float(current_positions[lane, 1]),
            int(action),
        )
        for lane, action in enumerate(waypoint_actions)
    ]
    corner_target_count = sum(
        controller.grid.is_corner(target_cell) for target_cell in raw_target_cells
    )
    macro_rewards = np.zeros(lane_count, dtype=np.float32)
    if config.corner_node_penalty:
        for lane, target_cell in enumerate(raw_target_cells):
            if controller.grid.is_corner(target_cell):
                macro_rewards[lane] += config.corner_node_penalty
    macro_terminated = np.zeros(lane_count, dtype=bool)
    macro_truncated = np.zeros(lane_count, dtype=bool)
    macro_next_stacks = np.zeros_like(current_stacks)
    boundary = np.zeros(lane_count, dtype=bool)
    life_reset_lanes: set[int] = set()
    reset_seed_overrides: dict[int, int] = {}
    life_loss_count = 0
    final_death_count = 0
    native_steps = 0
    game_frames = 0
    pending_resets: set[int] = set()
    arrived = np.zeros(lane_count, dtype=bool)
    if controller.arrival_latching:
        for lane, target_cell in enumerate(target_cells):
            arrived[lane] = controller.target_reached(
                float(current_positions[lane, 0]),
                float(current_positions[lane, 1]),
                target_cell,
            )

    for micro_step in range(config.hold_decisions):
        native_actions = np.zeros(lane_count, dtype=np.uint8)
        for lane in range(lane_count):
            if boundary[lane]:
                continue
            x, y = current_positions[lane]
            native_actions[lane] = controller.native_action_index_for_position(
                float(x),
                float(y),
                target_cells[lane],
                arrived=bool(arrived[lane]),
            )
        if config.macro_reset_mode == "freeze-lanes":
            result = environment.step_pixels_active(native_actions, ~boundary)
        else:
            result = _step_pixel_batch(environment, native_actions, config)
        result_pixels, result_positions = _validate_pixel_result(result)
        result_lanes = result.lane_ids.astype(np.int64)
        current_stacks[result_lanes] = _advance_pixel_stacks(
            current_stacks[result_lanes], result_pixels
        )
        current_positions[result_lanes] = result_positions
        native_steps += len(result_lanes)
        game_frames += int(result.frames_advanced.sum())
        reset_lanes: set[int] = set()
        for result_index, lane in enumerate(result_lanes.tolist()):
            actual_terminal = bool(result.done[result_index])
            if boundary[lane]:
                if actual_terminal:
                    reset_lanes.add(lane)
                else:
                    episode_steps[lane] += 1
                continue
            macro_rewards[lane] += float(result.rewards[result_index])
            episode_steps[lane] += 1
            truncated = (
                not actual_terminal and episode_steps[lane] >= config.max_episode_steps
            )
            if actual_terminal:
                life_loss_count += 1
                macro_rewards[lane] += config.life_loss_penalty
                final_loss = _consume_training_life(
                    lives_remaining,
                    episode_steps,
                    lane,
                    config.training_lives,
                )
                final_death_count += int(final_loss)
                boundary[lane] = True
                # Every collision is an MDP boundary because the following
                # pixels come from a reset. Outer "lives" only control seed
                # reuse; they must never bootstrap through an invisible reset.
                macro_terminated[lane] = (
                    final_loss if config.macro_reset_mode == "legacy-continue" else True
                )
                macro_next_stacks[lane] = current_stacks[lane]
                reset_lanes.add(lane)
                if not final_loss:
                    life_reset_lanes.add(lane)
            elif truncated:
                boundary[lane] = True
                macro_truncated[lane] = True
                macro_next_stacks[lane] = current_stacks[lane]
                reset_lanes.add(lane)
            elif controller.arrival_latching and not arrived[lane]:
                arrived[lane] = controller.target_reached(
                    float(current_positions[lane, 0]),
                    float(current_positions[lane, 1]),
                    target_cells[lane],
                )
        if config.macro_reset_mode == "freeze-lanes":
            pending_resets.update(reset_lanes)
            reset_lanes = (
                pending_resets
                if micro_step == config.hold_decisions - 1 or bool(boundary.all())
                else set()
            )
        if reset_lanes:
            ordered_reset_lanes = sorted(reset_lanes)
            replacement_seeds: list[int] = []
            for lane in ordered_reset_lanes:
                if lane in reset_seed_overrides:
                    seed = reset_seed_overrides[lane]
                elif lane in life_reset_lanes:
                    seed = int(episode_seeds[lane])
                    reset_seed_overrides[lane] = seed
                else:
                    seed, seed_cursor = _next_training_seed(
                        training_seeds,
                        seed_cursor,
                    )
                    episode_seeds[lane] = seed
                    lives_remaining[lane] = config.training_lives
                    reset_seed_overrides[lane] = seed
                replacement_seeds.append(seed)
            reset_stacks, reset_positions = _reset_pixel_lanes(
                environment,
                np.asarray(ordered_reset_lanes, dtype=np.uint32),
                np.asarray(replacement_seeds, dtype=np.uint32),
                config,
            )
            for index, lane in enumerate(ordered_reset_lanes):
                current_stacks[lane] = reset_stacks[index]
                current_positions[lane] = reset_positions[index]
                episode_steps[lane] = 0
                if (
                    lane in life_reset_lanes
                    and config.macro_reset_mode == "legacy-continue"
                ):
                    macro_next_stacks[lane] = reset_stacks[index]
            if config.macro_reset_mode == "stop-macro":
                break
        if bool(boundary.all()):
            break

    next_stacks = current_stacks.copy()
    next_stacks[boundary] = macro_next_stacks[boundary]
    transition_refs = replay.store_stacks(next_stacks)
    state_refs = transition_refs.copy()
    boundary_lanes = np.flatnonzero(boundary)
    if len(boundary_lanes):
        state_refs[boundary_lanes] = replay.store_stacks(current_stacks[boundary_lanes])
    for lane in range(lane_count):
        accumulator.append(
            lane,
            observation_refs[lane],
            int(waypoint_actions[lane]),
            target_cells[lane],
            float(macro_rewards[lane] * config.reward_scale),
            transition_refs[lane],
            bool(macro_terminated[lane]),
            bool(macro_truncated[lane]),
        )
    return (
        current_stacks,
        current_positions,
        state_refs,
        seed_cursor,
        native_steps,
        {
            "epsilon": epsilon,
            "macro_reward_mean": float(macro_rewards.mean()),
            "macro_reward_max": float(macro_rewards.max()),
            "scaled_macro_reward_mean": float(
                macro_rewards.mean() * config.reward_scale
            ),
            "life_loss_count": float(life_loss_count),
            "final_death_count": float(final_death_count),
            "lives_remaining_mean": float(lives_remaining.mean()),
            "corner_target_count": float(corner_target_count),
            "game_frames_collected": float(game_frames),
            **action_diagnostics,
        },
    )


def _new_pixel_environment(config: PixelDQNConfig) -> NativeBatchEnvironment:
    """Create the render boundary required for native indexed pixels."""
    return NativeBatchEnvironment(
        step_frames=config.step_frames,
        execution=config.native_execution,
        full_state=False,
        pixels=True,
        board=False,
        ml=True,
        ml_grid_spacing=config.grid_spacing,
    )


def evaluate_pixel_dqn(
    model: DuelingPixelDQN,
    seeds: Sequence[int],
    config: PixelDQNConfig,
) -> dict[str, object]:
    """Evaluate one greedy, one-life episode per seed."""
    if not seeds:
        raise ValueError("pixel DQN evaluation requires at least one seed")
    device = next(model.parameters()).device
    model.eval()
    survival: list[int] = []
    terminated: list[bool] = []
    for start in range(0, len(seeds), config.native_lanes):
        local_seeds = tuple(
            int(seed) for seed in seeds[start : start + config.native_lanes]
        )
        local_survival, local_terminated = _evaluate_pixel_batch(
            model,
            local_seeds,
            config,
            device,
        )
        survival.extend(local_survival)
        terminated.extend(local_terminated)
    value = {
        "seeds": [int(seed) for seed in seeds],
        "survival_frames": survival,
        "terminated": terminated,
    }
    return {**value, "summary": summarize_evaluation(value)}


def _evaluate_pixel_batch(
    model: DuelingPixelDQN,
    seeds: tuple[int, ...],
    config: PixelDQNConfig,
    device: torch.device,
) -> tuple[list[int], list[bool]]:
    environment = _new_pixel_environment(config)
    lane_count = len(seeds)
    active = np.ones(lane_count, dtype=bool)
    episode_steps = np.zeros(lane_count, dtype=np.int64)
    survival = np.zeros(lane_count, dtype=np.float32)
    terminated = np.zeros(lane_count, dtype=bool)
    try:
        current_stacks, current_positions = _reset_pixels(
            environment,
            np.asarray(seeds, dtype=np.uint32),
            config,
        )
        controller = _controller_for_config(config)
        while bool(active.any()):
            active_indices = np.flatnonzero(active)
            observations = current_stacks[active_indices]
            with torch.inference_mode():
                waypoint_actions = (
                    model(torch.from_numpy(observations).to(device))
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
            target_cells = [
                controller.grid.target_cell_for_action(
                    float(current_positions[int(lane), 0]),
                    float(current_positions[int(lane), 1]),
                    int(action),
                )
                for lane, action in zip(active_indices, waypoint_actions, strict=True)
            ]
            arrived = np.asarray(
                [
                    controller.target_reached(
                        float(current_positions[int(lane), 0]),
                        float(current_positions[int(lane), 1]),
                        target_cell,
                    )
                    for lane, target_cell in zip(
                        active_indices,
                        target_cells,
                        strict=True,
                    )
                ],
                dtype=bool,
            )
            if not controller.arrival_latching:
                arrived.fill(False)
            block_done = np.zeros(len(active_indices), dtype=bool)
            for _ in range(config.hold_decisions):
                native_actions = np.zeros(lane_count, dtype=np.uint8)
                for local, lane_value in enumerate(active_indices):
                    if block_done[local]:
                        continue
                    lane = int(lane_value)
                    native_actions[lane] = controller.native_action_index_for_position(
                        float(current_positions[lane, 0]),
                        float(current_positions[lane, 1]),
                        target_cells[local],
                        arrived=bool(arrived[local]),
                    )
                result = _step_pixel_batch(environment, native_actions, config)
                result_pixels, result_positions = _validate_pixel_result(result)
                current_stacks[:] = _advance_pixel_stacks(
                    current_stacks,
                    result_pixels,
                )
                current_positions[:] = result_positions
                completed: list[int] = []
                for local, lane_value in enumerate(active_indices):
                    if block_done[local]:
                        continue
                    lane = int(lane_value)
                    survival[lane] += float(result.rewards[lane])
                    episode_steps[lane] += 1
                    actual_terminal = bool(result.done[lane])
                    truncated = (
                        not actual_terminal
                        and episode_steps[lane] >= config.max_episode_steps
                    )
                    if actual_terminal or truncated:
                        block_done[local] = True
                        active[lane] = False
                        terminated[lane] = actual_terminal
                        if truncated:
                            survival[lane] = (
                                config.max_episode_steps * config.step_frames
                            )
                        completed.append(lane)
                    elif controller.arrival_latching and not arrived[local]:
                        arrived[local] = controller.target_reached(
                            float(current_positions[lane, 0]),
                            float(current_positions[lane, 1]),
                            target_cells[local],
                        )
                all_done = [lane for lane, done in enumerate(result.done) if bool(done)]
                reset_lanes = sorted(set(completed) | set(all_done))
                if reset_lanes:
                    reset_stacks, reset_positions = _reset_pixel_lanes(
                        environment,
                        np.asarray(reset_lanes, dtype=np.uint32),
                        np.zeros(len(reset_lanes), dtype=np.uint32),
                        config,
                    )
                    for index, lane in enumerate(reset_lanes):
                        current_stacks[lane] = reset_stacks[index]
                        current_positions[lane] = reset_positions[index]
                        episode_steps[lane] = 0
                if not bool(active.any()):
                    break
    finally:
        environment.close()
    return survival.astype(np.int64).tolist(), terminated.tolist()


def _checkpoint_contract(config: PixelDQNConfig) -> dict[str, object]:
    grid = WaypointGrid(
        config.grid_spacing,
        ban_corner_nodes=config.ban_corner_nodes,
    )
    return {
        "observation_mode": "pixels",
        "observation_source": "native_indexed_pixels_only",
        "pixel_shape": [config.pixel_stack, FRAME_HEIGHT, FRAME_WIDTH],
        "pixel_architecture": config.pixel_architecture,
        "model": PIXEL_DQN_MODEL_TYPE,
        "grid_spacing": config.grid_spacing,
        "grid_shape": list(grid.shape),
        "corner_nodes": "banned" if config.ban_corner_nodes else "allowed",
        "controller": {
            "tolerance": config.steering_tolerance,
            "arrival_latching": config.arrival_latching,
            "corner_node_penalty": config.corner_node_penalty,
            "steering": "sign(target_position-current_position)",
        },
        "cadence": {
            "step_frames": config.step_frames,
            "hold_decisions": config.hold_decisions,
            "decision_interval": config.hold_decisions,
        },
        "reset_mode": config.reset_mode,
        "macro_reset_mode": config.macro_reset_mode,
        "training_lives": config.training_lives,
        "life_loss_penalty": config.life_loss_penalty,
        "replay": {
            "capacity": config.replay_capacity,
            "storage": "uint8_memmapped_frame_ring",
            "n_step": config.n_step,
            "gamma": config.gamma,
            "reward_scale": config.reward_scale,
            "boundary": "terminated_or_truncated_zero_bootstrap",
        },
        "target": {
            "algorithm": "double_dqn",
            "network": "dueling",
            "update_interval": config.target_update_interval,
            "update_unit": config.target_update_unit,
            "train_frequency": config.train_frequency,
            "gradient_steps": config.gradient_steps,
            "huber_beta": config.huber_beta,
        },
        "actions": list(ACTION_CHOICES),
    }


def _checkpoint_payload(
    model: DuelingPixelDQN,
    target_model: DuelingPixelDQN,
    optimizer: torch.optim.Optimizer,
    config: PixelDQNConfig,
    manifest: SeedManifest,
    *,
    step: int,
    seed_cursor: int,
    best_inner: dict[str, object] | None,
    best_model_state: dict[str, Tensor] | None,
    replay: PixelReplayBuffer,
    accumulator: PixelNStepAccumulator,
    rng: np.random.Generator,
    total_native_steps: int,
) -> dict[str, object]:
    return {
        "version": PIXEL_DQN_VERSION,
        "kind": "dodge_ng_pixel_dqn_checkpoint",
        "manifest_sha256": manifest.sha256,
        "config": config.to_json(),
        "contract": _checkpoint_contract(config),
        "model_state_dict": model.state_dict(),
        "target_model_state_dict": target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "seed_cursor": seed_cursor,
        "best_inner": best_inner,
        "best_model_state": best_model_state,
        "training_state": {
            "replay": replay.state_dict(),
            "n_step": accumulator.state_dict(),
            "rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "total_native_steps": total_native_steps,
            "resume_boundary": (
                "native_lane_state_restarts; pending n-step queue is not reused"
            ),
        },
    }


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    """Commit weights and an immutable replay image at the same boundary.

    The live memory map is overwritten by subsequent collection. Its existence
    alone cannot make an older checkpoint resumable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    snapshot_path = path.parent / f"replay-{uuid.uuid4().hex}.u8.gz"
    snapshot_temporary = snapshot_path.with_suffix(".tmp")
    previous_snapshot = _checkpoint_snapshot_name(path)
    committed = False
    try:
        replay_state = payload["training_state"]["replay"]
        frame_path = path.parent / str(replay_state["frame_file"])
        digest = hashlib.sha256()
        raw_bytes = 0
        with (
            frame_path.open("rb") as source,
            gzip.open(snapshot_temporary, "wb", compresslevel=1) as destination,
        ):
            for block in iter(lambda: source.read(1024 * 1024), b""):
                destination.write(block)
                digest.update(block)
                raw_bytes += len(block)
        snapshot_temporary.replace(snapshot_path)
        with snapshot_path.open("rb") as stream:
            os.fsync(stream.fileno())
        payload = {
            **payload,
            "replay_snapshot": {
                "path": snapshot_path.name,
                "raw_bytes": raw_bytes,
                "raw_sha256": digest.hexdigest(),
                "format": "gzip-u8-v1",
            },
        }
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        committed = True
        if previous_snapshot is not None:
            with suppress(OSError):
                (path.parent / previous_snapshot).unlink(missing_ok=True)
    except BaseException:
        # The old checkpoint still owns its old immutable replay image.
        if not committed:
            snapshot_path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        snapshot_temporary.unlink(missing_ok=True)


def _checkpoint_snapshot_name(path: Path) -> str | None:
    if not path.is_file():
        return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    snapshot = saved.get("replay_snapshot")
    if not isinstance(snapshot, dict):
        return None
    name = snapshot.get("path")
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.startswith("replay-")
    ):
        raise ValueError("pixel checkpoint replay snapshot name is invalid")
    return name


def _restore_replay_snapshot(checkpoint: Path, frame_path: Path) -> bool:
    """Restore the exact saved ring before opening the mutable replay map."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    snapshot = saved.get("replay_snapshot")
    if snapshot is None:
        return False  # Legacy checkpoint uses its historical external frame map.
    name = _checkpoint_snapshot_name(checkpoint)
    if name is None or snapshot.get("format") != "gzip-u8-v1":
        raise ValueError("pixel checkpoint replay snapshot format is invalid")
    expected_bytes = snapshot.get("raw_bytes")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ValueError("pixel checkpoint replay snapshot size is invalid")
    temporary = frame_path.with_name(f".{frame_path.name}.restore")
    try:
        digest = hashlib.sha256()
        restored_bytes = 0
        with (
            gzip.open(checkpoint.parent / name, "rb") as source,
            temporary.open("wb") as output,
        ):
            for block in iter(lambda: source.read(1024 * 1024), b""):
                restored_bytes += len(block)
                if restored_bytes > expected_bytes:
                    raise ValueError("replay snapshot exceeds declared size")
                digest.update(block)
                output.write(block)
        if restored_bytes != expected_bytes or digest.hexdigest() != snapshot.get(
            "raw_sha256"
        ):
            raise ValueError("replay snapshot hash or size does not match checkpoint")
        temporary.replace(frame_path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _checkpoint_config_matches(
    saved: object,
    config: PixelDQNConfig,
) -> bool:
    if not isinstance(saved, dict):
        return False
    normalized = dict(saved)
    normalized.setdefault("native_pixel_boundary", "legacy")
    normalized.setdefault("macro_reset_mode", "legacy-continue")
    normalized.setdefault("gradient_steps", 1)
    normalized.setdefault("reward_scale", 1.0)
    normalized.setdefault("huber_beta", 1.0)
    normalized.setdefault("target_update_unit", "collection")
    saved_total = normalized.get("total_steps")
    if (
        isinstance(saved_total, bool)
        or not isinstance(saved_total, int)
        or config.total_steps < saved_total
    ):
        return False
    normalized["total_steps"] = config.total_steps
    return normalized == config.to_json()


def _checkpoint_contract_matches(
    saved: object,
    config: PixelDQNConfig,
) -> bool:
    if not isinstance(saved, dict):
        return False
    normalized = dict(saved)
    normalized.setdefault("macro_reset_mode", "legacy-continue")
    replay = normalized.get("replay")
    if isinstance(replay, dict):
        replay = dict(replay)
        replay.setdefault("reward_scale", 1.0)
        normalized["replay"] = replay
    target = normalized.get("target")
    if isinstance(target, dict):
        target = dict(target)
        target.setdefault("train_frequency", 1)
        target.setdefault("gradient_steps", 1)
        target.setdefault("huber_beta", 1.0)
        target.setdefault("update_unit", "collection")
        normalized["target"] = target
    return normalized == _checkpoint_contract(config)


def _load_checkpoint(
    path: Path,
    model: DuelingPixelDQN,
    target_model: DuelingPixelDQN,
    optimizer: torch.optim.Optimizer,
    config: PixelDQNConfig,
    manifest: SeedManifest,
    replay: PixelReplayBuffer,
    rng: np.random.Generator,
) -> tuple[int, int, dict[str, object] | None, dict[str, Tensor] | None, int]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ControlRuntimeError(
            f"could not load pixel DQN checkpoint: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("pixel DQN checkpoint must contain an object")
    if payload.get("kind") != "dodge_ng_pixel_dqn_checkpoint":
        raise ValueError("pixel DQN checkpoint kind is invalid")
    if payload.get("version") != PIXEL_DQN_VERSION:
        raise ValueError("pixel DQN checkpoint version is invalid")
    if payload.get("manifest_sha256") != manifest.sha256:
        raise ValueError("pixel DQN checkpoint manifest does not match NG manifest")
    if not _checkpoint_config_matches(payload.get("config"), config):
        raise ValueError("pixel DQN checkpoint configuration does not match")
    if not _checkpoint_contract_matches(payload.get("contract"), config):
        raise ValueError("pixel DQN checkpoint contract does not match")
    try:
        model.load_state_dict(payload["model_state_dict"])
        target_model.load_state_dict(payload["target_model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        step = int(payload["step"])
        seed_cursor = int(payload["seed_cursor"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"pixel DQN checkpoint state is invalid: {error}") from error
    if not 0 <= step <= config.total_steps or seed_cursor < 0:
        raise ValueError("pixel DQN checkpoint progress is invalid")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("pixel DQN checkpoint has no resumable training state")
    replay.load_state_dict(training_state.get("replay"))
    try:
        rng.bit_generator.state = training_state["rng_state"]
        torch_rng_state = training_state["torch_rng_state"]
        if not isinstance(torch_rng_state, Tensor):
            raise ValueError("pixel DQN torch RNG state is invalid")
        torch.set_rng_state(torch_rng_state)
        total_native_steps = int(training_state["total_native_steps"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            f"pixel DQN checkpoint training state is invalid: {error}"
        ) from error
    if total_native_steps < 0:
        raise ValueError("pixel DQN native-step count is invalid")
    best_inner = payload.get("best_inner")
    best_model_state = payload.get("best_model_state")
    if best_model_state is not None and not isinstance(best_model_state, dict):
        raise ValueError("pixel DQN best model state is invalid")
    return (
        step,
        seed_cursor,
        best_inner if isinstance(best_inner, dict) else None,
        best_model_state,
        total_native_steps,
    )


def _install_stop_signal_handlers() -> tuple[list[bool], dict[signal.Signals, object]]:
    requested = [False]
    previous: dict[signal.Signals, object] = {}

    def request_stop(_signum: int, _frame: object) -> None:
        requested[0] = True

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
    except ValueError:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        previous = {}
    return requested, previous


def _restore_stop_signal_handlers(
    previous: dict[signal.Signals, object],
) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _status(
    config: PixelDQNConfig,
    manifest: SeedManifest,
    *,
    state: str,
    step: int,
    total_native_steps: int,
    replay_size: int,
    best_inner: dict[str, object] | None,
    record: dict[str, object] | None = None,
    final_training: dict[str, object] | None = None,
    final_holdout: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "state": state,
        "step": step,
        "total_steps": config.total_steps,
        "native_steps": total_native_steps,
        "replay_size": replay_size,
        "best_inner": dict(best_inner) if best_inner is not None else None,
        "config": config.to_json(),
        "manifest_sha256": manifest.sha256,
        "training_seeds": list(manifest.training_seeds),
        "record": dict(record) if record is not None else None,
    }
    if final_training is not None:
        payload["final_training"] = final_training["summary"]
    if final_holdout is not None:
        payload["final_holdout"] = final_holdout["summary"]
    return payload


def _configure_torch_threads(thread_count: int) -> None:
    if thread_count:
        torch.set_num_threads(thread_count)
    with suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _read_metrics(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ControlRuntimeError(f"pixel metrics are unreadable: {path}") from error
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ControlRuntimeError(
                f"pixel metrics contain invalid JSON: {path}"
            ) from error
        if not isinstance(value, dict):
            raise ControlRuntimeError("pixel metrics records must be objects")
        records.append(value)
    return records


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_metrics(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _mark_previous_finalization_stale(run_directory: Path, resumed_step: int) -> None:
    marker = {
        "state": "stale_during_resumed_training",
        "resumed_from_step": resumed_step,
        "marked_at": time.time(),
    }
    for name in ("run.json", "report.json"):
        path = run_directory / name
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            value["artifact_state"] = marker
            _write_json(path, value)
    report_path = run_directory / "REPORT.md"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        warning = (
            "> **Stale:** training resumed from this finalization; use live dashboard "
            "status until the next stop or completion.\n\n"
        )
        if not report.startswith("> **Stale:**"):
            report_path.write_text(warning + report, encoding="utf-8")


def _copy_model_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _train_pixel_dqn_impl(
    config: PixelDQNConfig,
    run_directory: Path,
    manifest: SeedManifest,
    *,
    resume: bool,
    evaluate_holdout: bool,
    evaluate_training: bool,
    stop_requested: list[bool],
    profile_directory: Path | None = None,
    profile_steps: int = 0,
) -> dict[str, object]:
    config.validate()
    manifest.validate()
    if profile_steps < 0:
        raise ValueError("pixel profiler steps must not be negative")
    if profile_steps and profile_directory is None:
        profile_directory = run_directory / "profile"
    if len(manifest.training_seeds) < config.native_lanes:
        raise ValueError("pixel DQN lane count exceeds NG training seed count")
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "checkpoint-latest.pt"
    metrics_path = run_directory / "metrics.jsonl"
    if not resume and any(
        path.exists()
        for path in (
            checkpoint_path,
            run_directory / "run.json",
            run_directory / ".pixel-frames.u8",
        )
    ):
        raise ControlRuntimeError(
            f"pixel DQN run directory is not empty; use --resume: {run_directory}"
        )
    _configure_torch_threads(config.torch_threads)
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    profiler = _PixelProfiler(profile_directory, device, profile_steps)
    model = DuelingPixelDQN(
        stack_size=config.pixel_stack,
        hidden_size=config.hidden_size,
        architecture=config.pixel_architecture,
    ).to(device)
    target_model = DuelingPixelDQN(
        stack_size=config.pixel_stack,
        hidden_size=config.hidden_size,
        architecture=config.pixel_architecture,
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if resume and checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            saved.get("manifest_sha256") != manifest.sha256
            or not _checkpoint_config_matches(saved.get("config"), config)
            or not _checkpoint_contract_matches(saved.get("contract"), config)
        ):
            raise ValueError(
                "pixel checkpoint configuration or manifest does not match"
            )
        _restore_replay_snapshot(checkpoint_path, run_directory / ".pixel-frames.u8")
    replay = PixelReplayBuffer(
        config.replay_capacity,
        config.pixel_stack,
        config.native_lanes,
        run_directory,
        resume=resume,
    )
    accumulator = PixelNStepAccumulator(
        config.native_lanes,
        config.n_step,
        config.gamma,
        replay,
    )
    rng = np.random.default_rng(config.seed)
    step = 0
    seed_cursor = 0
    best_inner: dict[str, object] | None = None
    best_model_state: dict[str, Tensor] | None = None
    total_native_steps = 0
    if resume:
        if not checkpoint_path.is_file():
            replay.close()
            raise ControlRuntimeError(
                f"pixel DQN checkpoint does not exist: {checkpoint_path}"
            )
        (
            step,
            seed_cursor,
            best_inner,
            best_model_state,
            total_native_steps,
        ) = _load_checkpoint(
            checkpoint_path,
            model,
            target_model,
            optimizer,
            config,
            manifest,
            replay,
            rng,
        )
        _mark_previous_finalization_stale(run_directory, step)
    metrics = _read_metrics(metrics_path)
    if resume:
        metrics = [
            item
            for item in metrics
            if isinstance(item.get("step"), int) and item["step"] <= step
        ]
        _write_metrics(metrics_path, metrics)
    elif metrics_path.exists():
        replay.close()
        raise ControlRuntimeError(f"pixel metrics already exist: {metrics_path}")
    optimizer_steps = sum(
        int(item.get("gradient_steps", 1)) for item in metrics if "loss" in item
    )
    samples_drawn = optimizer_steps * config.batch_size
    total_game_frames = (
        int(metrics[-1].get("game_frames", total_native_steps * config.step_frames))
        if metrics
        else 0
    )
    session_started = time.monotonic()
    session_start_game_frames = total_game_frames
    telemetry = DashboardTelemetry(run_directory)
    environment = _new_pixel_environment(config)
    controller = _controller_for_config(config)
    checkpoint_best_path = run_directory / "checkpoint-best.pt"
    stopped = False
    paused = False
    metrics_stream = metrics_path.open("a", encoding="utf-8", buffering=1)
    try:
        initial_seeds: list[int] = []
        for _ in range(config.native_lanes):
            seed, seed_cursor = _next_training_seed(
                manifest.training_seeds,
                seed_cursor,
            )
            initial_seeds.append(seed)
        episode_seeds = np.asarray(initial_seeds, dtype=np.uint32)
        current_stacks, current_positions = _reset_pixels(
            environment,
            episode_seeds,
            config,
        )
        current_frame_refs = replay.store_stacks(current_stacks)
        episode_steps = np.zeros(config.native_lanes, dtype=np.int64)
        lives_remaining = np.full(
            config.native_lanes,
            config.training_lives,
            dtype=np.int64,
        )
        telemetry.publish(
            _status(
                config,
                manifest,
                state="starting",
                step=step,
                total_native_steps=total_native_steps,
                replay_size=replay.size,
                best_inner=best_inner,
            )
        )
        model.train()
        while step < config.total_steps:
            command = telemetry.consume_control()
            if command == "pause":
                paused = True
            elif command == "resume":
                paused = False
            elif command == "stop" or stop_requested[0]:
                stopped = True
                telemetry.publish(
                    _status(
                        config,
                        manifest,
                        state="stopping",
                        step=step,
                        total_native_steps=total_native_steps,
                        replay_size=replay.size,
                        best_inner=best_inner,
                    )
                )
                break
            if paused:
                telemetry.publish(
                    _status(
                        config,
                        manifest,
                        state="paused",
                        step=step,
                        total_native_steps=total_native_steps,
                        replay_size=replay.size,
                        best_inner=best_inner,
                    )
                )
                time.sleep(0.1)
                continue
            step_started = time.perf_counter()
            profiler.start()
            collection_started = time.monotonic()
            (
                current_stacks,
                current_positions,
                current_frame_refs,
                seed_cursor,
                native_steps,
                collection,
            ) = _collect_macro_transition(
                environment,
                current_stacks,
                current_frame_refs,
                current_positions,
                controller,
                model,
                config,
                episode_steps,
                episode_seeds,
                lives_remaining,
                manifest.training_seeds,
                seed_cursor,
                accumulator,
                replay,
                rng,
                device,
                step,
            )
            collection_seconds = time.monotonic() - collection_started
            total_native_steps += native_steps
            total_game_frames += int(collection["game_frames_collected"])
            step += 1
            learning: dict[str, float] = {}
            learning_seconds = 0.0
            if (
                step >= config.warmup_steps
                and step % config.train_frequency == 0
                and replay.size >= config.batch_size
            ):
                model.train()
                learning_started = time.monotonic()
                learning_records: list[dict[str, float]] = []
                for _ in range(config.gradient_steps):
                    learning_records.append(
                        _learn_step(
                            model,
                            target_model,
                            optimizer,
                            replay,
                            config,
                            rng,
                            device,
                        )
                    )
                    optimizer_steps += 1
                    samples_drawn += config.batch_size
                    if (
                        config.target_update_unit == "optimizer"
                        and optimizer_steps % config.target_update_interval == 0
                    ):
                        target_model.load_state_dict(model.state_dict())
                learning_seconds = time.monotonic() - learning_started
                learning = {
                    name: float(np.mean([record[name] for record in learning_records]))
                    for name in learning_records[0]
                }
                learning["gradient_steps"] = float(config.gradient_steps)
            evaluation_seconds = 0.0
            checkpoint_seconds = 0.0
            if (
                config.target_update_unit == "collection"
                and step % config.target_update_interval == 0
            ):
                target_model.load_state_dict(model.state_dict())
            elapsed_seconds = time.monotonic() - session_started
            transitions_collected = step * config.native_lanes
            record: dict[str, object] = {
                "step": step,
                "replay_size": replay.size,
                "native_steps": total_native_steps,
                "game_frames": total_game_frames,
                "optimizer_steps": optimizer_steps,
                "samples_drawn": samples_drawn,
                "update_to_data_ratio": optimizer_steps / max(1, transitions_collected),
                "sample_replay_ratio": samples_drawn / max(1, transitions_collected),
                "collection_seconds": collection_seconds,
                "learning_seconds": learning_seconds,
                "replay_sample_seconds": float(
                    learning.get("replay_sample_seconds", 0.0)
                ),
                "host_to_device_seconds": float(
                    learning.get("host_to_device_seconds", 0.0)
                ),
                "learner_compute_seconds": float(
                    learning.get("learner_compute_seconds", 0.0)
                ),
                "evaluation_seconds": evaluation_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "session_elapsed_seconds": elapsed_seconds,
                "session_game_frames_per_second": (
                    (total_game_frames - session_start_game_frames)
                    / max(elapsed_seconds, 1e-9)
                ),
                **collection,
                **learning,
            }
            evaluation_due = step % config.eval_every == 0 or step == config.total_steps
            if evaluation_due:
                # Evaluation can take substantially longer than collection on
                # large seed sets. Publish a resumable boundary before entering
                # it so a runtime loss cannot discard the just-completed budget.
                checkpoint_started = time.perf_counter()
                _save_checkpoint(
                    checkpoint_path,
                    _checkpoint_payload(
                        model,
                        target_model,
                        optimizer,
                        config,
                        manifest,
                        step=step,
                        seed_cursor=seed_cursor,
                        best_inner=best_inner,
                        best_model_state=best_model_state,
                        replay=replay,
                        accumulator=accumulator,
                        rng=rng,
                        total_native_steps=total_native_steps,
                    ),
                )
                checkpoint_seconds += time.perf_counter() - checkpoint_started
                telemetry.publish(
                    _status(
                        config,
                        manifest,
                        state="evaluating",
                        step=step,
                        total_native_steps=total_native_steps,
                        replay_size=replay.size,
                        best_inner=best_inner,
                        record=record,
                    )
                )
                evaluation_started = time.perf_counter()
                inner = evaluate_pixel_dqn(
                    model,
                    manifest.training_seeds[:10],
                    config,
                )
                evaluation_seconds += time.perf_counter() - evaluation_started
                record["inner_validation"] = inner["summary"]
                inner_mean = float(inner["summary"]["mean_survival_frames"])
                if best_inner is None or inner_mean > float(
                    best_inner["mean_survival_frames"]
                ):
                    best_inner = {
                        "mean_survival_frames": inner_mean,
                        "step": step,
                    }
                    best_model_state = _copy_model_state(model)
                    checkpoint_started = time.perf_counter()
                    _save_checkpoint(
                        checkpoint_best_path,
                        _checkpoint_payload(
                            model,
                            target_model,
                            optimizer,
                            config,
                            manifest,
                            step=step,
                            seed_cursor=seed_cursor,
                            best_inner=best_inner,
                            best_model_state=best_model_state,
                            replay=replay,
                            accumulator=accumulator,
                            rng=rng,
                            total_native_steps=total_native_steps,
                        ),
                    )
                    checkpoint_seconds += time.perf_counter() - checkpoint_started
                model.train()
            if step % config.checkpoint_every == 0 or step == config.total_steps:
                checkpoint_started = time.perf_counter()
                _save_checkpoint(
                    checkpoint_path,
                    _checkpoint_payload(
                        model,
                        target_model,
                        optimizer,
                        config,
                        manifest,
                        step=step,
                        seed_cursor=seed_cursor,
                        best_inner=best_inner,
                        best_model_state=best_model_state,
                        replay=replay,
                        accumulator=accumulator,
                        rng=rng,
                        total_native_steps=total_native_steps,
                    ),
                )
                checkpoint_seconds += time.perf_counter() - checkpoint_started
                record["checkpoint_seconds"] = checkpoint_seconds
            step_wall_seconds = time.perf_counter() - step_started
            phase_seconds = (
                collection_seconds
                + learning_seconds
                + evaluation_seconds
                + checkpoint_seconds
            )
            record["step_wall_seconds"] = step_wall_seconds
            record["phase_overhead_seconds"] = max(
                0.0,
                step_wall_seconds - phase_seconds,
            )
            profiler.step()
            metrics_stream.write(json.dumps(record, sort_keys=True) + "\n")
            telemetry.publish(
                _status(
                    config,
                    manifest,
                    state="running",
                    step=step,
                    total_native_steps=total_native_steps,
                    replay_size=replay.size,
                    best_inner=best_inner,
                    record=record,
                )
            )
    except Exception as error:
        telemetry.publish(
            {
                **_status(
                    config,
                    manifest,
                    state="failed",
                    step=step,
                    total_native_steps=total_native_steps,
                    replay_size=replay.size,
                    best_inner=best_inner,
                ),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        telemetry.close()
        replay.close()
        raise
    finally:
        metrics_stream.close()
        profiler.close()
        environment.close()

    _save_checkpoint(
        checkpoint_path,
        _checkpoint_payload(
            model,
            target_model,
            optimizer,
            config,
            manifest,
            step=step,
            seed_cursor=seed_cursor,
            best_inner=best_inner,
            best_model_state=best_model_state,
            replay=replay,
            accumulator=accumulator,
            rng=rng,
            total_native_steps=total_native_steps,
        ),
    )
    final_model = model
    selected_model = "final_at_configured_budget"
    selected_model_step = step
    if best_model_state is not None and best_inner is not None:
        final_model = DuelingPixelDQN(
            stack_size=config.pixel_stack,
            hidden_size=config.hidden_size,
            architecture=config.pixel_architecture,
        ).to(device)
        final_model.load_state_dict(best_model_state)
        selected_model = "best_inner_checkpoint"
        selected_model_step = int(best_inner["step"])
    final_inner = evaluate_pixel_dqn(
        final_model,
        manifest.training_seeds[:10],
        config,
    )
    final_training = (
        evaluate_pixel_dqn(final_model, manifest.training_seeds, config)
        if evaluate_training
        else None
    )
    final_holdout = (
        evaluate_pixel_dqn(final_model, manifest.holdout_seeds, config)
        if evaluate_holdout
        else None
    )
    run_record = {
        "version": PIXEL_DQN_VERSION,
        "kind": "dodge_ng_pixel_dqn_run",
        "host": platform.node(),
        "manifest_sha256": manifest.sha256,
        "config": config.to_json(),
        "contract": _checkpoint_contract(config),
        "observation_mode": "pixels",
        "observation_source": "native_indexed_pixels_only",
        "pixel_shape": [config.pixel_stack, FRAME_HEIGHT, FRAME_WIDTH],
        "pixel_palette": "indexed_u8_0_to_15",
        "model": PIXEL_DQN_MODEL_TYPE,
        "actions": list(ACTION_CHOICES),
        "training_seeds": list(manifest.training_seeds),
        "holdout_seeds": list(manifest.holdout_seeds),
        "updates_completed": step,
        "native_steps": total_native_steps,
        "game_frames": total_game_frames,
        "collection_steps": step,
        "optimizer_steps": optimizer_steps,
        "samples_drawn": samples_drawn,
        "wall_seconds_this_session": time.monotonic() - session_started,
        "profiling": {
            "enabled": profiler.enabled,
            "directory": (
                str(profile_directory) if profile_directory is not None else None
            ),
            "requested_steps": profile_steps,
            "captured_steps": profiler.captured_steps,
        },
        "stopped_early": stopped,
        "best_inner": best_inner,
        "selected_model": selected_model,
        "selected_model_step": selected_model_step,
        "final_validation": final_inner,
        "final_training_evaluation": final_training,
        "final_evaluation": final_holdout,
        "target": {
            "comparison_budget_updates": 200_000,
            "relevance_gate_frames": RELEVANCE_GATE_FRAMES,
            "safety_limit_frames": config.max_episode_steps * config.step_frames,
            "training_mean_reached": (
                final_training is not None
                and float(final_training["summary"]["mean_survival_frames"])
                >= RELEVANCE_GATE_FRAMES
            ),
        },
        "resume_boundary": (
            "replay and learner resume; native lanes restart at fresh training seeds"
        ),
    }
    _write_json(run_directory / "run.json", run_record)
    build_pixel_report(run_directory)
    telemetry.publish(
        _status(
            config,
            manifest,
            state="stopped" if stopped else "completed",
            step=step,
            total_native_steps=total_native_steps,
            replay_size=replay.size,
            best_inner=best_inner,
            final_training=final_training,
            final_holdout=final_holdout,
        )
    )
    telemetry.close()
    replay.close()
    return run_record


def train_pixel_dqn(
    config: PixelDQNConfig,
    run_directory: Path,
    manifest: SeedManifest,
    *,
    resume: bool = False,
    evaluate_holdout: bool = True,
    evaluate_training: bool = True,
    profile_directory: Path | None = None,
    profile_steps: int = 0,
) -> dict[str, object]:
    """Train one pixel DQN run with graceful stop handling."""
    stop_requested, previous_handlers = _install_stop_signal_handlers()
    try:
        return _train_pixel_dqn_impl(
            config,
            run_directory,
            manifest,
            resume=resume,
            evaluate_holdout=evaluate_holdout,
            evaluate_training=evaluate_training,
            stop_requested=stop_requested,
            profile_directory=profile_directory,
            profile_steps=profile_steps,
        )
    finally:
        _restore_stop_signal_handlers(previous_handlers)


def _evaluation_summary(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    summary = value.get("summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    try:
        return summarize_evaluation(value)
    except (TypeError, ValueError):
        return None


def _metric_points(
    metrics: Sequence[Mapping[str, object]],
    field: str,
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for metric in metrics:
        step = metric.get("step")
        value = metric.get(field)
        if isinstance(step, (int, float)) and isinstance(value, (int, float)):
            points.append({"step": float(step), "value": float(value)})
    return points


def _trend(points: Sequence[Mapping[str, float]]) -> dict[str, float] | None:
    if not points:
        return None
    first = points[0]
    last = points[-1]
    best = max(points, key=lambda item: item["value"])
    return {
        "first_step": first["step"],
        "first_value": first["value"],
        "last_step": last["step"],
        "last_value": last["value"],
        "best_step": best["step"],
        "best_value": best["value"],
        "change": last["value"] - first["value"],
    }


def build_pixel_report(run_directory: Path) -> dict[str, object]:
    """Write compact metrics, plots, and Markdown for one pixel run."""
    run_path = Path(run_directory)
    run_record_path = run_path / "run.json"
    try:
        run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlRuntimeError(
            f"pixel run metadata is unreadable: {run_record_path}"
        ) from error
    if not isinstance(run_record, dict):
        raise ValueError("pixel run metadata must be an object")
    metrics = _read_metrics(run_path / "metrics.jsonl")
    final_training = _evaluation_summary(run_record.get("final_training_evaluation"))
    final_holdout = _evaluation_summary(run_record.get("final_evaluation"))
    final_inner = _evaluation_summary(run_record.get("final_validation"))
    comparison = None
    if final_training is not None and final_holdout is not None:
        comparison = {
            "mean_train_minus_holdout": float(final_training["mean_survival_frames"])
            - float(final_holdout["mean_survival_frames"]),
            "median_train_minus_holdout": float(
                final_training["median_survival_frames"]
            )
            - float(final_holdout["median_survival_frames"]),
            "p10_train_minus_holdout": float(final_training["p10_survival_frames"])
            - float(final_holdout["p10_survival_frames"]),
        }
    curves = {
        field: _metric_points(metrics, field)
        for field in (
            "epsilon",
            "loss",
            "q_mean",
            "td_error",
            "life_loss_count",
            "final_death_count",
            "macro_reward_mean",
        )
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "dodge_ng_pixel_dqn_report",
        "run_directory": str(run_path),
        "provenance": run_record,
        "splits": {
            "training": final_training,
            "inner_validation": final_inner,
            "holdout": final_holdout,
        },
        "comparison": comparison,
        "curves": curves,
        "trend": {field: _trend(points) for field, points in curves.items()},
        "metrics_count": len(metrics),
        "plots": ["pixel_training_curves.png", "pixel_split_survival.png"],
    }
    _write_json(run_path / "report.json", report)
    _write_pixel_plots(run_path, curves, final_training, final_holdout)
    (run_path / "REPORT.md").write_text(
        _pixel_report_markdown(run_record, final_training, final_holdout, comparison),
        encoding="utf-8",
    )
    return report


def _write_pixel_plots(
    run_directory: Path,
    curves: Mapping[str, Sequence[Mapping[str, float]]],
    training: Mapping[str, object] | None,
    holdout: Mapping[str, object] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for field, color in (
        ("loss", "tab:blue"),
        ("q_mean", "tab:orange"),
        ("td_error", "tab:red"),
    ):
        points = curves[field]
        if points:
            axes[0].plot(
                [point["step"] for point in points],
                [point["value"] for point in points],
                label=field,
                color=color,
            )
    axes[0].set_title("Pixel DQN learner diagnostics")
    axes[0].set_xlabel("update")
    if any(curves[field] for field in ("loss", "q_mean", "td_error")):
        axes[0].legend()
    epsilon = curves["epsilon"]
    if epsilon:
        axes[1].plot(
            [point["step"] for point in epsilon],
            [point["value"] for point in epsilon],
            label="epsilon",
            color="tab:green",
        )
    for field, color in (
        ("life_loss_count", "tab:red"),
        ("macro_reward_mean", "tab:purple"),
    ):
        points = curves[field]
        if points:
            axes[1].plot(
                [point["step"] for point in points],
                [point["value"] for point in points],
                label=field,
                color=color,
            )
    axes[1].set_title("Exploration, life losses, and collection reward")
    axes[1].set_xlabel("update")
    axes[1].legend()
    figure.savefig(run_directory / "pixel_training_curves.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    plotted = False
    for label, value, color in (
        ("training", training, "tab:blue"),
        ("holdout", holdout, "tab:orange"),
    ):
        if value is None:
            continue
        survival = value.get("survival_frames")
        if isinstance(survival, list):
            axis.plot(
                sorted(int(item) for item in survival),
                marker=".",
                label=label,
                color=color,
            )
            plotted = True
    axis.set_title("Final survival by seed")
    axis.set_xlabel("sorted seed rank")
    axis.set_ylabel("survival frames")
    if plotted:
        axis.legend()
    figure.savefig(run_directory / "pixel_split_survival.png", dpi=120)
    plt.close(figure)


def _pixel_report_markdown(
    run_record: Mapping[str, object],
    training: Mapping[str, object] | None,
    holdout: Mapping[str, object] | None,
    comparison: Mapping[str, object] | None,
) -> str:
    config = run_record.get("config")
    config_value = config if isinstance(config, Mapping) else {}
    lines = [
        "# Pixel-only DQN run",
        "",
        f"- Updates completed: `{run_record.get('updates_completed')}`",
        f"- Observation: `{run_record.get('observation_source')}`",
        "- Native pixel boundary: "
        f"`{config_value.get('native_pixel_boundary', 'legacy')}`",
        f"- Pixel shape: `{run_record.get('pixel_shape')}`",
        f"- Training lives: `{config_value.get('training_lives')}`",
        f"- Life-loss penalty: `{config_value.get('life_loss_penalty')}`",
        f"- Manifest: `{run_record.get('manifest_sha256')}`",
        "",
        "| split | mean | median | p10 | worst | best | completion |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, value in (("training", training), ("holdout", holdout)):
        if value is None:
            lines.append(f"| {label} | unavailable | - | - | - | - | - |")
            continue
        lines.append(
            "| {label} | {mean:.1f} | {median:.1f} | {p10} | {worst} | "
            "{best} | {completion:.1%} |".format(
                label=label,
                mean=float(value["mean_survival_frames"]),
                median=float(value["median_survival_frames"]),
                p10=int(value["p10_survival_frames"]),
                worst=int(value["worst_survival_frames"]),
                best=int(value["best_survival_frames"]),
                completion=float(value["horizon_completion_fraction"]),
            )
        )
    if comparison is not None:
        lines.extend(
            [
                "",
                "Train-minus-holdout gap:",
                "",
                f"- Mean: `{float(comparison['mean_train_minus_holdout']):.1f}` frames",
                "- Median: "
                f"`{float(comparison['median_train_minus_holdout']):.1f}` frames",
                f"- P10: `{float(comparison['p10_train_minus_holdout']):.1f}` frames",
            ]
        )
    lines.extend(
        [
            "",
            "The final split evaluation uses the model at the configured budget; "
            "inner validation is reported separately and does not replace it.",
            "",
        ]
    )
    return "\n".join(lines)


def compare_pixel_runs(
    no_lives_directory: Path,
    three_lives_directory: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Compare matched one-life and three-life runs after exactly 200k updates."""
    no_lives = _load_run_record(no_lives_directory)
    three_lives = _load_run_record(three_lives_directory)
    for label, record in (("no-lives", no_lives), ("three-lives", three_lives)):
        if record.get("kind") != "dodge_ng_pixel_dqn_run":
            raise ValueError(f"{label} run kind is not pixel DQN")
        if record.get("updates_completed") != 200_000:
            raise ValueError(f"{label} run did not complete exactly 200,000 updates")
    if no_lives.get("manifest_sha256") != three_lives.get("manifest_sha256"):
        raise ValueError("pixel comparison manifests do not match")
    first_config = no_lives.get("config")
    second_config = three_lives.get("config")
    if not isinstance(first_config, Mapping) or not isinstance(second_config, Mapping):
        raise ValueError("pixel comparison configs are invalid")
    left = dict(first_config)
    right = dict(second_config)
    left.setdefault("native_pixel_boundary", "legacy")
    right.setdefault("native_pixel_boundary", "legacy")
    if left.get("training_lives") == right.get("training_lives"):
        raise ValueError("pixel comparison requires different training_lives")
    left.pop("training_lives", None)
    right.pop("training_lives", None)
    if left != right:
        raise ValueError("pixel comparison configs differ beyond training_lives")
    rows: list[dict[str, object]] = []
    for label, record in (("no-lives", no_lives), ("three-lives", three_lives)):
        for split, key in (
            ("training", "final_training_evaluation"),
            ("holdout", "final_evaluation"),
        ):
            summary = _evaluation_summary(record.get(key))
            if summary is None:
                raise ValueError(f"{label} {split} evaluation is unavailable")
            rows.append(
                {
                    "run": label,
                    "split": split,
                    "training_lives": record["config"]["training_lives"],
                    "mean_survival_frames": summary["mean_survival_frames"],
                    "median_survival_frames": summary["median_survival_frames"],
                    "p10_survival_frames": summary["p10_survival_frames"],
                    "worst_survival_frames": summary["worst_survival_frames"],
                    "best_survival_frames": summary["best_survival_frames"],
                    "horizon_completion_fraction": summary[
                        "horizon_completion_fraction"
                    ],
                }
            )
    comparison = {
        "schema_version": 1,
        "kind": "dodge_ng_pixel_dqn_life_comparison",
        "budget_updates": 200_000,
        "manifest_sha256": no_lives["manifest_sha256"],
        "runs": {
            "no_lives": str(no_lives_directory),
            "three_lives": str(three_lives_directory),
        },
        "rows": rows,
        "configuration_difference": {
            "field": "training_lives",
            "no_lives": no_lives["config"]["training_lives"],
            "three_lives": three_lives["config"]["training_lives"],
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_json(output_directory / "comparison.json", comparison)
    (output_directory / "COMPARISON.md").write_text(
        _comparison_markdown(comparison),
        encoding="utf-8",
    )
    return comparison


def _load_run_record(run_directory: Path) -> dict[str, object]:
    try:
        value = json.loads(
            (Path(run_directory) / "run.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ControlRuntimeError(
            f"pixel run metadata is unreadable: {run_directory}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("pixel run metadata must be an object")
    return value


def _comparison_markdown(value: Mapping[str, object]) -> str:
    rows = value["rows"]
    assert isinstance(rows, list)
    lines = [
        "# Pixel-only DQN life ablation",
        "",
        "Both runs completed exactly 200,000 updates under the same manifest and "
        "configuration except `training_lives`.",
        "",
        "| run | lives | split | mean | median | p10 | worst | best | completion |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        assert isinstance(row, Mapping)
        lines.append(
            "| {run} | {lives} | {split} | {mean:.1f} | {median:.1f} | {p10} | "
            "{worst} | {best} | {completion:.1%} |".format(
                run=row["run"],
                lives=row["training_lives"],
                split=row["split"],
                mean=float(row["mean_survival_frames"]),
                median=float(row["median_survival_frames"]),
                p10=int(row["p10_survival_frames"]),
                worst=int(row["worst_survival_frames"]),
                best=int(row["best_survival_frames"]),
                completion=float(row["horizon_completion_fraction"]),
            )
        )
    lines.extend(["", f"Manifest: `{value['manifest_sha256']}`", ""])
    return "\n".join(lines)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-pixel-dqn")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "history" / "dodge" / "ng" / "pixel-dqn-200k",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-holdout", action="store_true")
    parser.add_argument("--no-training-evaluation", action="store_true")
    parser.add_argument("--total-steps", type=_positive_int, default=200_000)
    parser.add_argument("--batch-size", type=_positive_int, default=128)
    parser.add_argument("--replay-capacity", type=_positive_int, default=100_000)
    parser.add_argument("--training-lives", type=_positive_int, default=1)
    parser.add_argument("--life-loss-penalty", type=float, default=-64.0)
    parser.add_argument("--native-lanes", type=_positive_int, default=12)
    parser.add_argument(
        "--native-pixel-boundary",
        choices=("legacy", "fast"),
        default="legacy",
    )
    parser.add_argument("--checkpoint-every", type=_positive_int, default=10_000)
    parser.add_argument("--eval-every", type=_positive_int, default=10_000)
    parser.add_argument("--pixel-stack", type=_positive_int, default=4)
    parser.add_argument(
        "--pixel-architecture",
        choices=PIXEL_ARCHITECTURES,
        default="fast",
    )
    parser.add_argument("--hidden-size", type=_positive_int, default=128)
    parser.add_argument("--grid-spacing", type=_positive_int, default=32)
    parser.add_argument("--hold-decisions", type=_positive_int, default=8)
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=_positive_int, default=2_000)
    parser.add_argument("--steering-tolerance", type=float, default=2.0)
    parser.add_argument("--arrival-latching", action="store_true")
    parser.add_argument("--ban-corner-nodes", action="store_true")
    parser.add_argument("--corner-node-penalty", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.00024074661619742944)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=_positive_int, default=5)
    parser.add_argument("--warmup-steps", type=_positive_int, default=2_000)
    parser.add_argument("--train-frequency", type=_positive_int, default=1)
    parser.add_argument("--gradient-steps", type=_positive_int, default=1)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--target-update-interval", type=_positive_int, default=500)
    parser.add_argument(
        "--target-update-unit", choices=("optimizer", "collection"), default="optimizer"
    )
    parser.add_argument(
        "--macro-reset-mode",
        choices=("freeze-lanes", "stop-macro", "legacy-continue"),
        default="freeze-lanes",
    )
    parser.add_argument("--epsilon-decay-steps", type=_nonnegative_int, default=50_000)
    parser.add_argument("--epsilon-final", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2_026_0903)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cpu")
    parser.add_argument("--torch-threads", type=_nonnegative_int, default=8)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-steps", type=_nonnegative_int, default=0)
    arguments = parser.parse_args(argv)
    config = PixelDQNConfig(
        total_steps=arguments.total_steps,
        batch_size=arguments.batch_size,
        replay_capacity=arguments.replay_capacity,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        gamma=arguments.gamma,
        n_step=arguments.n_step,
        warmup_steps=arguments.warmup_steps,
        train_frequency=arguments.train_frequency,
        gradient_steps=arguments.gradient_steps,
        reward_scale=arguments.reward_scale,
        huber_beta=arguments.huber_beta,
        target_update_interval=arguments.target_update_interval,
        target_update_unit=arguments.target_update_unit,
        hidden_size=arguments.hidden_size,
        pixel_stack=arguments.pixel_stack,
        pixel_architecture=arguments.pixel_architecture,
        grid_spacing=arguments.grid_spacing,
        hold_decisions=arguments.hold_decisions,
        steering_tolerance=arguments.steering_tolerance,
        arrival_latching=arguments.arrival_latching,
        ban_corner_nodes=arguments.ban_corner_nodes,
        corner_node_penalty=arguments.corner_node_penalty,
        step_frames=arguments.step_frames,
        max_episode_steps=arguments.max_episode_steps,
        native_lanes=arguments.native_lanes,
        native_pixel_boundary=arguments.native_pixel_boundary,
        macro_reset_mode=arguments.macro_reset_mode,
        training_lives=arguments.training_lives,
        life_loss_penalty=arguments.life_loss_penalty,
        epsilon_decay_steps=arguments.epsilon_decay_steps,
        epsilon_final=arguments.epsilon_final,
        checkpoint_every=arguments.checkpoint_every,
        eval_every=arguments.eval_every,
        seed=arguments.seed,
        device=arguments.device,
        torch_threads=arguments.torch_threads,
    )
    try:
        manifest = load_manifest(arguments.manifest)
        started = time.monotonic()
        run = train_pixel_dqn(
            config,
            arguments.run_dir,
            manifest,
            resume=arguments.resume,
            evaluate_holdout=not arguments.no_holdout,
            evaluate_training=not arguments.no_training_evaluation,
            profile_directory=arguments.profile_dir,
            profile_steps=arguments.profile_steps,
        )
    except (ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-ng-pixel-dqn: {error}")
        return 1
    print(
        json.dumps(
            {
                "run_directory": str(arguments.run_dir),
                "updates_completed": run["updates_completed"],
                "stopped_early": run["stopped_early"],
                "wall_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        )
    )
    return 0


def compare_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-pixel-compare")
    parser.add_argument("--no-lives", type=Path, required=True)
    parser.add_argument("--three-lives", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    arguments = parser.parse_args(argv)
    try:
        comparison = compare_pixel_runs(
            arguments.no_lives,
            arguments.three_lives,
            arguments.output_dir,
        )
    except (ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-ng-pixel-compare: {error}")
        return 1
    print(json.dumps(comparison, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
