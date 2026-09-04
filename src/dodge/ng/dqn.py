"""Waypoint DQN learner for the Dodge NG training boundary."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import torch
from torch import Tensor, nn

from dodge.control import ControlRuntimeError
from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import (
    NativeBatchEnvironment,
    NativeBatchResult,
    _decode_snapshot,
    _raw_state_from_snapshot,
)
from dodge.native.differential import decode_native_player_position
from dodge.neat.state import (
    OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION,
    RawState,
    project_state,
)
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest
from dodge.ng.report import summarize_evaluation
from dodge.ng.waypoint import WaypointController, WaypointGrid

WAYPOINT_OBSERVATION_SIZE: Final[int] = OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION + 4
WAYPOINT_DQN_VERSION: Final[int] = 2
RELEVANCE_GATE_FRAMES: Final[int] = 800
WaypointExecution = Literal["serial", "parallel"]


def encode_waypoint_observation(state: RawState, grid: WaypointGrid) -> np.ndarray:
    """Encode declared state features plus the player's discrete grid cell."""
    projected = project_state(
        state,
        include_time_to_intersection=True,
    )
    column, row = grid.nearest_cell(state.player.x, state.player.y)
    denominator = max(1, len(grid.axis_points) - 1)
    values = (
        *projected.values,
        column / denominator,
        row / denominator,
        float(projected.enemy_overflow),
        float(projected.aoe_overflow),
    )
    observation = np.asarray(values, dtype=np.float32)
    if observation.shape != (WAYPOINT_OBSERVATION_SIZE,):
        raise ControlRuntimeError(
            "waypoint observation has unexpected shape: "
            f"expected {(WAYPOINT_OBSERVATION_SIZE,)}, got {observation.shape}"
        )
    if not np.isfinite(observation).all():
        raise ControlRuntimeError("waypoint observation contains non-finite values")
    return observation


class DuelingWaypointDQN(nn.Module):
    """Dueling MLP for relative waypoint actions."""

    def __init__(
        self,
        input_size: int = WAYPOINT_OBSERVATION_SIZE,
        hidden_size: int = 256,
        action_count: int = len(ACTION_CHOICES),
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or action_count < 1:
            raise ValueError("DQN dimensions must be positive")
        self.input_size = input_size
        self.action_count = action_count
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_size, 1)
        self.advantage = nn.Linear(hidden_size, action_count)

    def forward(self, observations: Tensor) -> Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.input_size:
            raise ValueError(
                f"waypoint DQN observations must have shape (N, {self.input_size})"
            )
        features = self.features(observations)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Waypoint DQN hyperparameters and native interaction contract."""

    total_steps: int = 20_000
    batch_size: int = 256
    replay_capacity: int = 100_000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    n_step: int = 3
    warmup_steps: int = 2_000
    train_frequency: int = 1
    target_update_interval: int = 1_000
    hidden_size: int = 256
    grid_spacing: int = 32
    hold_decisions: int = 8
    step_frames: int = 4
    max_episode_steps: int = 2_000
    native_lanes: int = 32
    native_execution: WaypointExecution = "parallel"
    checkpoint_every: int = 2_000
    eval_every: int = 2_000
    seed: int = 2_026_0903
    device: str = "cpu"

    def validate(self) -> None:
        positive = (
            self.total_steps,
            self.batch_size,
            self.replay_capacity,
            self.n_step,
            self.warmup_steps,
            self.train_frequency,
            self.target_update_interval,
            self.hidden_size,
            self.hold_decisions,
            self.max_episode_steps,
            self.native_lanes,
            self.checkpoint_every,
            self.eval_every,
        )
        if any(value < 1 for value in positive):
            raise ValueError("DQN counts and intervals must be positive")
        if self.batch_size > self.replay_capacity:
            raise ValueError("DQN batch size must not exceed replay capacity")
        if self.learning_rate <= 0:
            raise ValueError("DQN learning rate must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("DQN gamma must be between 0 and 1")
        if not 3 <= self.step_frames <= 5:
            raise ValueError("step frames must be between 3 and 5")
        if self.grid_spacing < 1:
            raise ValueError("DQN grid spacing must be positive")
        if self.n_step > 2**8 - 1:
            raise ValueError("DQN n-step horizon must fit replay storage")
        if self.native_execution not in {"serial", "parallel"}:
            raise ValueError("native execution must be serial or parallel")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("DQN device must be cpu, cuda, or auto")

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReplaySample:
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


class ReplayBuffer:
    """Fixed-size uniform replay storage with explicit episode boundaries."""

    def __init__(self, capacity: int, observation_size: int) -> None:
        if capacity < 1 or observation_size < 1:
            raise ValueError("replay dimensions must be positive")
        self.capacity = capacity
        self.observation_size = observation_size
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.next_observations = np.empty(
            (capacity, observation_size), dtype=np.float32
        )
        self.actions = np.empty(capacity, dtype=np.uint8)
        self.target_columns = np.empty(capacity, dtype=np.uint16)
        self.target_rows = np.empty(capacity, dtype=np.uint16)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self.n_steps = np.empty(capacity, dtype=np.uint8)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        target_cell: tuple[int, int],
        reward: float,
        next_observation: np.ndarray,
        discount: float,
        terminated: bool,
        truncated: bool,
        n_steps: int,
    ) -> None:
        if observation.shape != (self.observation_size,):
            raise ValueError("replay observation shape is invalid")
        if next_observation.shape != (self.observation_size,):
            raise ValueError("replay next-observation shape is invalid")
        if not 0 <= action < len(ACTION_CHOICES):
            raise ValueError("replay action is outside the native action space")
        if not all(isinstance(value, int) and value >= 0 for value in target_cell):
            raise ValueError("replay target cell is invalid")
        if target_cell[0] >= 2**16 or target_cell[1] >= 2**16:
            raise ValueError("replay target cell is too large")
        if (
            not np.isfinite(observation).all()
            or not np.isfinite(next_observation).all()
        ):
            raise ValueError("replay observations must be finite")
        if not np.isfinite(reward) or not np.isfinite(discount):
            raise ValueError("replay reward and discount must be finite")
        if not 1 <= n_steps <= 2**8 - 1:
            raise ValueError("replay n-step horizon is invalid")
        index = self.position
        self.observations[index] = observation
        self.next_observations[index] = next_observation
        self.actions[index] = action
        self.target_columns[index] = target_cell[0]
        self.target_rows[index] = target_cell[1]
        self.rewards[index] = reward
        self.discounts[index] = discount
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.n_steps[index] = n_steps
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplaySample:
        if not 1 <= batch_size <= self.size:
            raise ValueError("replay sample size is unavailable")
        indices = rng.integers(0, self.size, size=batch_size)
        return ReplaySample(
            observations=self.observations[indices],
            actions=self.actions[indices],
            target_columns=self.target_columns[indices],
            target_rows=self.target_rows[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            discounts=self.discounts[indices],
            terminated=self.terminated[indices],
            truncated=self.truncated[indices],
            n_steps=self.n_steps[indices],
        )

    def state_dict(self) -> dict[str, object]:
        """Serialize occupied replay storage for resumable checkpoints."""
        size = self.size
        return {
            "capacity": self.capacity,
            "observation_size": self.observation_size,
            "position": self.position,
            "size": size,
            "observations": self.observations[:size].copy(),
            "next_observations": self.next_observations[:size].copy(),
            "actions": self.actions[:size].copy(),
            "target_columns": self.target_columns[:size].copy(),
            "target_rows": self.target_rows[:size].copy(),
            "rewards": self.rewards[:size].copy(),
            "discounts": self.discounts[:size].copy(),
            "terminated": self.terminated[:size].copy(),
            "truncated": self.truncated[:size].copy(),
            "n_steps": self.n_steps[:size].copy(),
        }

    def load_state_dict(self, value: object) -> None:
        if not isinstance(value, dict):
            raise ValueError("replay checkpoint state must be an object")
        if value.get("capacity") != self.capacity:
            raise ValueError("replay checkpoint capacity does not match")
        if value.get("observation_size") != self.observation_size:
            raise ValueError("replay checkpoint observation size does not match")
        size = value.get("size")
        position = value.get("position")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 <= size <= self.capacity
            or isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < self.capacity
        ):
            raise ValueError("replay checkpoint cursor is invalid")
        fields = (
            ("observations", self.observations, (size, self.observation_size)),
            (
                "next_observations",
                self.next_observations,
                (size, self.observation_size),
            ),
            ("actions", self.actions, (size,)),
            ("target_columns", self.target_columns, (size,)),
            ("target_rows", self.target_rows, (size,)),
            ("rewards", self.rewards, (size,)),
            ("discounts", self.discounts, (size,)),
            ("terminated", self.terminated, (size,)),
            ("truncated", self.truncated, (size,)),
            ("n_steps", self.n_steps, (size,)),
        )
        for name, destination, shape in fields:
            source = value.get(name)
            if not isinstance(source, np.ndarray) or source.shape != shape:
                raise ValueError(f"replay checkpoint field {name} is invalid")
            if source.dtype != destination.dtype:
                raise ValueError(f"replay checkpoint field {name} dtype is invalid")
            destination[:size] = source
        self.position = position
        self.size = size


@dataclass(frozen=True, slots=True)
class _PendingTransition:
    observation: np.ndarray
    action: int
    target_cell: tuple[int, int]
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool


class NStepAccumulator:
    """Aggregate per-lane transitions without crossing episode boundaries."""

    def __init__(
        self,
        lane_count: int,
        n_step: int,
        gamma: float,
        replay: ReplayBuffer,
    ):
        if lane_count < 1 or n_step < 1:
            raise ValueError("n-step dimensions must be positive")
        self.n_step = n_step
        self.gamma = gamma
        self.replay = replay
        self.queues: list[deque[_PendingTransition]] = [
            deque() for _ in range(lane_count)
        ]

    def append(
        self,
        lane: int,
        observation: np.ndarray,
        action: int,
        target_cell: tuple[int, int],
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        queue = self.queues[lane]
        queue.append(
            _PendingTransition(
                observation.copy(),
                action,
                target_cell,
                reward,
                next_observation.copy(),
                terminated,
                truncated,
            )
        )
        if len(queue) >= self.n_step:
            self._emit(lane)
        if terminated or truncated:
            while queue:
                self._emit(lane)

    def _emit(self, lane: int) -> None:
        queue = self.queues[lane]
        if not queue:
            return
        reward = 0.0
        discount = 1.0
        horizon = 0
        last: _PendingTransition | None = None
        for pending in queue:
            reward += discount * pending.reward
            horizon += 1
            discount *= self.gamma
            last = pending
            if pending.terminated or pending.truncated or horizon >= self.n_step:
                break
        if last is None:
            raise ControlRuntimeError("n-step queue emitted without a transition")
        boundary = last.terminated or last.truncated
        self.replay.add(
            queue[0].observation,
            queue[0].action,
            queue[0].target_cell,
            reward,
            last.next_observation,
            0.0 if boundary else self.gamma**horizon,
            last.terminated,
            last.truncated,
            horizon,
        )
        queue.popleft()

    def state_dict(self) -> dict[str, object]:
        return {
            "n_step": self.n_step,
            "gamma": self.gamma,
            "queues": [
                [
                    {
                        "observation": pending.observation.copy(),
                        "action": pending.action,
                        "target_cell": list(pending.target_cell),
                        "reward": pending.reward,
                        "next_observation": pending.next_observation.copy(),
                        "terminated": pending.terminated,
                        "truncated": pending.truncated,
                    }
                    for pending in queue
                ]
                for queue in self.queues
            ],
        }

    def load_state_dict(self, value: object) -> None:
        if not isinstance(value, dict):
            raise ValueError("n-step checkpoint state must be an object")
        if value.get("n_step") != self.n_step or value.get("gamma") != self.gamma:
            raise ValueError("n-step checkpoint configuration does not match")
        queues = value.get("queues")
        if not isinstance(queues, list) or len(queues) != len(self.queues):
            raise ValueError("n-step checkpoint queues are invalid")
        restored: list[deque[_PendingTransition]] = []
        for entries in queues:
            if not isinstance(entries, list) or len(entries) >= self.n_step:
                raise ValueError("n-step checkpoint queue length is invalid")
            queue: deque[_PendingTransition] = deque()
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("n-step checkpoint transition is invalid")
                observation = entry.get("observation")
                next_observation = entry.get("next_observation")
                action = entry.get("action")
                target_cell = entry.get("target_cell")
                reward = entry.get("reward")
                terminated = entry.get("terminated")
                truncated = entry.get("truncated")
                if (
                    not isinstance(observation, np.ndarray)
                    or observation.shape != (self.replay.observation_size,)
                    or observation.dtype != np.float32
                    or not isinstance(next_observation, np.ndarray)
                    or next_observation.shape != (self.replay.observation_size,)
                    or next_observation.dtype != np.float32
                    or not isinstance(action, int)
                    or not isinstance(target_cell, list)
                    or len(target_cell) != 2
                    or not all(isinstance(item, int) for item in target_cell)
                    or not isinstance(reward, (int, float))
                    or not isinstance(terminated, bool)
                    or not isinstance(truncated, bool)
                ):
                    raise ValueError("n-step checkpoint transition fields are invalid")
                queue.append(
                    _PendingTransition(
                        observation.copy(),
                        action,
                        (target_cell[0], target_cell[1]),
                        float(reward),
                        next_observation.copy(),
                        terminated,
                        truncated,
                    )
                )
            restored.append(queue)
        self.queues = restored


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("DQN requested cuda but cuda is unavailable")
    return torch.device(device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _snapshot_at(snapshots: list[bytes | None], lane: int) -> bytes:
    snapshot = snapshots[lane]
    if snapshot is None:
        raise ControlRuntimeError("DQN lost a live native snapshot")
    return snapshot


def _player_position(snapshot: bytes) -> tuple[float, float]:
    return decode_native_player_position(snapshot)


def _observations_from_snapshots(
    snapshots: Sequence[bytes], grid: WaypointGrid
) -> np.ndarray:
    return np.stack(
        [
            encode_waypoint_observation(
                _raw_state_from_snapshot(_decode_snapshot(snapshot)),
                grid,
            )
            for snapshot in snapshots
        ]
    )


def _native_ml_state(result: NativeBatchResult) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return the native ML features and player positions."""
    observations = result.ml_observation
    positions = result.player_positions
    if observations is None or positions is None:
        raise ControlRuntimeError(
            "waypoint DQN requires native ML observations and player positions"
        )
    if observations.shape != (result.lane_count, WAYPOINT_OBSERVATION_SIZE):
        raise ControlRuntimeError(
            "native ML observations have unexpected shape: "
            f"expected {(result.lane_count, WAYPOINT_OBSERVATION_SIZE)}, "
            f"got {observations.shape}"
        )
    if positions.shape != (result.lane_count, 2):
        raise ControlRuntimeError(
            "native ML player positions have unexpected shape: "
            f"expected {(result.lane_count, 2)}, got {positions.shape}"
        )
    if observations.dtype != np.float32 or positions.dtype != np.float32:
        raise ControlRuntimeError("native ML arrays must use float32 values")
    if not np.isfinite(observations).all() or not np.isfinite(positions).all():
        raise ControlRuntimeError("native ML arrays must be finite")
    return observations, positions


def _epsilon(config: DQNConfig, step: int) -> float:
    decay_steps = max(config.total_steps // 2, 1)
    progress = min(1.0, step / decay_steps)
    return 1.0 + progress * (0.05 - 1.0)


def _choose_actions(
    model: DuelingWaypointDQN,
    observations: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> np.ndarray:
    with torch.inference_mode():
        values = model(torch.from_numpy(observations).to(device))
    greedy = values.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
    random_mask = rng.random(len(greedy)) < epsilon
    random_actions = rng.integers(0, len(ACTION_CHOICES), size=len(greedy))
    greedy[random_mask] = random_actions[random_mask]
    return greedy


def _learn_step(
    model: DuelingWaypointDQN,
    target_model: DuelingWaypointDQN,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: DQNConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, float]:
    sample = replay.sample(config.batch_size, rng)
    observations = torch.from_numpy(sample.observations).to(device)
    actions = torch.from_numpy(sample.actions.astype(np.int64)).to(device)
    rewards = torch.from_numpy(sample.rewards).to(device)
    next_observations = torch.from_numpy(sample.next_observations).to(device)
    discounts = torch.from_numpy(sample.discounts).to(device)
    q_values = model(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = model(next_observations).argmax(dim=1)
        next_values = (
            target_model(next_observations)
            .gather(1, next_actions.unsqueeze(1))
            .squeeze(1)
        )
        targets = rewards + discounts * next_values
    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    )
    optimizer.step()
    td_error = (q_values.detach() - targets).abs().mean().item()
    return {
        "loss": float(loss.item()),
        "q_mean": float(q_values.detach().mean().item()),
        "target_mean": float(targets.mean().item()),
        "td_error": float(td_error),
        "gradient_norm": gradient_norm,
    }


def _next_training_seed(
    training_seeds: tuple[int, ...], cursor: int
) -> tuple[int, int]:
    if not training_seeds:
        raise ValueError("DQN requires training seeds")
    return training_seeds[cursor % len(training_seeds)], cursor + 1


def _collect_macro_transition(
    environment: NativeBatchEnvironment,
    current_observations: np.ndarray,
    current_positions: np.ndarray,
    controller: WaypointController,
    model: DuelingWaypointDQN,
    config: DQNConfig,
    episode_steps: np.ndarray,
    training_seeds: tuple[int, ...],
    seed_cursor: int,
    replay_accumulator: NStepAccumulator,
    rng: np.random.Generator,
    device: torch.device,
    global_step: int,
) -> tuple[np.ndarray, np.ndarray, int, int, dict[str, float]]:
    lane_count = len(current_observations)
    observations = current_observations.copy()
    epsilon = _epsilon(config, global_step)
    waypoint_actions = _choose_actions(model, observations, epsilon, rng, device)
    target_cells = [
        controller.grid.target_cell_for_action(
            float(current_positions[lane, 0]),
            float(current_positions[lane, 1]),
            int(action),
        )
        for lane, action in enumerate(waypoint_actions)
    ]
    macro_rewards = np.zeros(lane_count, dtype=np.float32)
    macro_terminated = np.zeros(lane_count, dtype=bool)
    macro_truncated = np.zeros(lane_count, dtype=bool)
    macro_next_observations = np.zeros_like(observations)
    boundary = np.zeros(lane_count, dtype=bool)
    native_steps = 0
    for _ in range(config.hold_decisions):
        native_actions = np.zeros(lane_count, dtype=np.uint8)
        for lane in range(lane_count):
            if boundary[lane]:
                continue
            x, y = current_positions[lane]
            native_actions[lane] = controller.native_action_index_for_position(
                float(x),
                float(y),
                target_cells[lane],
            )
        result = environment.step_batch(native_actions)
        result_observations, result_positions = _native_ml_state(result)
        native_steps += lane_count
        reset_lanes: set[int] = set()
        for lane in range(lane_count):
            actual_terminal = bool(result.done[lane])
            if boundary[lane]:
                if actual_terminal:
                    reset_lanes.add(lane)
                else:
                    current_observations[lane] = result_observations[lane]
                    current_positions[lane] = result_positions[lane]
                    episode_steps[lane] += 1
                continue
            macro_rewards[lane] += float(result.rewards[lane])
            episode_steps[lane] += 1
            truncated = (
                not actual_terminal and episode_steps[lane] >= config.max_episode_steps
            )
            if actual_terminal or truncated:
                boundary[lane] = True
                macro_terminated[lane] = actual_terminal
                macro_truncated[lane] = truncated
                macro_next_observations[lane] = result_observations[lane]
                reset_lanes.add(lane)
            else:
                current_observations[lane] = result_observations[lane]
                current_positions[lane] = result_positions[lane]
        if reset_lanes:
            replacement_seeds: list[int] = []
            ordered_reset_lanes = sorted(reset_lanes)
            for _lane in ordered_reset_lanes:
                seed, seed_cursor = _next_training_seed(training_seeds, seed_cursor)
                replacement_seeds.append(seed)
            reset = environment.reset_lanes(
                np.asarray(ordered_reset_lanes, dtype=np.uint32),
                np.asarray(replacement_seeds, dtype=np.uint32),
            )
            reset_observations, reset_positions = _native_ml_state(reset)
            for index, lane in enumerate(ordered_reset_lanes):
                current_observations[lane] = reset_observations[index]
                current_positions[lane] = reset_positions[index]
                episode_steps[lane] = 0
    next_observations = np.where(
        boundary[:, None], macro_next_observations, current_observations
    )
    for lane in range(lane_count):
        replay_accumulator.append(
            lane,
            observations[lane],
            int(waypoint_actions[lane]),
            target_cells[lane],
            float(macro_rewards[lane]),
            next_observations[lane],
            bool(macro_terminated[lane]),
            bool(macro_truncated[lane]),
        )
    return (
        current_observations,
        current_positions,
        seed_cursor,
        native_steps,
        {
            "epsilon": epsilon,
            "macro_reward_mean": float(macro_rewards.mean()),
            "macro_reward_max": float(macro_rewards.max()),
        },
    )


def evaluate_waypoint_dqn(
    model: DuelingWaypointDQN,
    seeds: Sequence[int],
    config: DQNConfig,
    *,
    grid: WaypointGrid | None = None,
) -> dict[str, object]:
    """Greedy macro-waypoint evaluation with one episode per seed."""
    if not seeds:
        raise ValueError("DQN evaluation requires at least one seed")
    grid = grid or WaypointGrid(config.grid_spacing)
    controller = WaypointController(grid)
    device = next(model.parameters()).device
    model.eval()
    survival: list[int] = []
    terminated: list[bool] = []
    for start in range(0, len(seeds), config.native_lanes):
        local_seeds = tuple(
            int(seed) for seed in seeds[start : start + config.native_lanes]
        )
        local_survival, local_terminated = _evaluate_batch(
            model,
            local_seeds,
            controller,
            config,
            device,
        )
        survival.extend(local_survival)
        terminated.extend(local_terminated)
    return {
        "seeds": [int(seed) for seed in seeds],
        "survival_frames": survival,
        "terminated": terminated,
        "summary": summarize_evaluation(
            {
                "seeds": [int(seed) for seed in seeds],
                "survival_frames": survival,
                "terminated": terminated,
            }
        ),
    }


def _evaluate_batch(
    model: DuelingWaypointDQN,
    seeds: tuple[int, ...],
    controller: WaypointController,
    config: DQNConfig,
    device: torch.device,
) -> tuple[list[int], list[bool]]:
    environment = NativeBatchEnvironment(
        step_frames=config.step_frames,
        execution=config.native_execution,
        full_state=False,
        pixels=False,
        board=False,
        ml=True,
        ml_grid_spacing=config.grid_spacing,
    )
    lane_count = len(seeds)
    active = np.ones(lane_count, dtype=bool)
    episode_steps = np.zeros(lane_count, dtype=np.int64)
    survival = np.zeros(lane_count, dtype=np.float32)
    terminated = np.zeros(lane_count, dtype=bool)
    current_observations: np.ndarray
    current_positions: np.ndarray
    try:
        result = environment.reset_batch(np.asarray(seeds, dtype=np.uint32))
        current_observations, current_positions = _native_ml_state(result)
        current_observations = current_observations.copy()
        current_positions = current_positions.copy()
        while bool(active.any()):
            active_indices = np.flatnonzero(active)
            observations = current_observations[active_indices]
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
            block_done = np.zeros(len(active_indices), dtype=bool)
            for _ in range(config.hold_decisions):
                native_actions = np.zeros(lane_count, dtype=np.uint8)
                for local, lane_value in enumerate(active_indices):
                    if block_done[local]:
                        continue
                    lane = int(lane_value)
                    x, y = current_positions[lane]
                    native_actions[lane] = controller.native_action_index_for_position(
                        float(x),
                        float(y),
                        target_cells[local],
                    )
                result = environment.step_batch(native_actions)
                result_observations, result_positions = _native_ml_state(result)
                current_observations[:] = result_observations
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
                all_done = [lane for lane, done in enumerate(result.done) if bool(done)]
                reset_lanes = sorted(set(completed) | set(all_done))
                if reset_lanes:
                    reset = environment.reset_lanes(
                        np.asarray(reset_lanes, dtype=np.uint32),
                        np.zeros(len(reset_lanes), dtype=np.uint32),
                    )
                    reset_observations, reset_positions = _native_ml_state(reset)
                    for index, lane in enumerate(reset_lanes):
                        current_observations[lane] = reset_observations[index]
                        current_positions[lane] = reset_positions[index]
                        episode_steps[lane] = 0
                if not bool(active.any()):
                    break
    finally:
        environment.close()
    return survival.astype(np.int64).tolist(), terminated.tolist()


def _checkpoint_contract(config: DQNConfig) -> dict[str, object]:
    grid = WaypointGrid(config.grid_spacing)
    return {
        "grid_spacing": config.grid_spacing,
        "grid_shape": list(grid.shape),
        "controller": {
            "tolerance": 2.0,
            "steering": "sign(target_position-current_position)",
        },
        "cadence": {
            "step_frames": config.step_frames,
            "hold_decisions": config.hold_decisions,
        },
        "observation_size": WAYPOINT_OBSERVATION_SIZE,
        "observation_contract": (
            "native_projected_state_with_time_to_intersection+grid_cell+overflow"
        ),
        "observation_source": "native_ml_with_python_reference_parity",
        "relevance_gate_frames": RELEVANCE_GATE_FRAMES,
        "max_episode_steps": config.max_episode_steps,
        "replay": {
            "capacity": config.replay_capacity,
            "n_step": config.n_step,
            "gamma": config.gamma,
            "boundary": "terminated_or_truncated_zero_bootstrap",
        },
        "target": {
            "algorithm": "double_dqn",
            "network": "dueling",
            "update_interval": config.target_update_interval,
        },
        "actions": list(ACTION_CHOICES),
    }


def _checkpoint_payload(
    model: DuelingWaypointDQN,
    target_model: DuelingWaypointDQN,
    optimizer: torch.optim.Optimizer,
    config: DQNConfig,
    manifest: SeedManifest,
    *,
    step: int,
    seed_cursor: int,
    best_inner: dict[str, object] | None,
    best_model_state: dict[str, Tensor] | None = None,
    replay: ReplayBuffer | None = None,
    replay_accumulator: NStepAccumulator | None = None,
    rng: np.random.Generator | None = None,
    total_native_steps: int = 0,
    metrics: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": WAYPOINT_DQN_VERSION,
        "kind": "dodge_ng_waypoint_dqn_checkpoint",
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
    }
    if replay is not None or replay_accumulator is not None or rng is not None:
        if replay is None or replay_accumulator is None or rng is None:
            raise ValueError("checkpoint training state must be complete")
        payload["training_state"] = {
            "replay": replay.state_dict(),
            "n_step": replay_accumulator.state_dict(),
            "rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "total_native_steps": total_native_steps,
            "metrics": list(metrics),
        }
    return payload


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(
    path: Path,
    model: DuelingWaypointDQN,
    target_model: DuelingWaypointDQN,
    optimizer: torch.optim.Optimizer,
    config: DQNConfig,
    manifest: SeedManifest,
    replay: ReplayBuffer,
    replay_accumulator: NStepAccumulator,
    rng: np.random.Generator,
) -> tuple[
    int,
    int,
    dict[str, object] | None,
    dict[str, Tensor] | None,
    int,
    list[dict[str, object]],
]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ControlRuntimeError(f"could not load DQN checkpoint: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("DQN checkpoint must contain an object")
    if payload.get("kind") != "dodge_ng_waypoint_dqn_checkpoint":
        raise ValueError("DQN checkpoint kind is invalid")
    if payload.get("version") != WAYPOINT_DQN_VERSION:
        raise ValueError("DQN checkpoint version is invalid")
    if payload.get("manifest_sha256") != manifest.sha256:
        raise ValueError("DQN checkpoint manifest does not match NG manifest")
    if payload.get("config") != config.to_json():
        raise ValueError("DQN checkpoint configuration does not match")
    if payload.get("contract") != _checkpoint_contract(config):
        raise ValueError("DQN checkpoint controller contract is invalid")
    try:
        model.load_state_dict(payload["model_state_dict"])
        target_model.load_state_dict(payload["target_model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        step = int(payload["step"])
        seed_cursor = int(payload["seed_cursor"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"DQN checkpoint state is invalid: {error}") from error
    if not 0 <= step <= config.total_steps or seed_cursor < 0:
        raise ValueError("DQN checkpoint progress is invalid")
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("DQN checkpoint has no resumable training state")
    replay.load_state_dict(training_state.get("replay"))
    replay_accumulator.load_state_dict(training_state.get("n_step"))
    try:
        rng.bit_generator.state = training_state["rng_state"]
        torch_rng_state = training_state["torch_rng_state"]
        if not isinstance(torch_rng_state, Tensor):
            raise ValueError("DQN torch RNG state is invalid")
        torch.set_rng_state(torch_rng_state)
        total_native_steps = int(training_state["total_native_steps"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            f"DQN checkpoint training state is invalid: {error}"
        ) from error
    if total_native_steps < 0:
        raise ValueError("DQN checkpoint native-step count is invalid")
    metrics = training_state.get("metrics", [])
    if not isinstance(metrics, list) or any(
        not isinstance(item, dict) for item in metrics
    ):
        raise ValueError("DQN checkpoint metrics are invalid")
    best_inner = payload.get("best_inner")
    best_model_state = payload.get("best_model_state")
    if best_model_state is not None and not isinstance(best_model_state, dict):
        raise ValueError("DQN checkpoint best model state is invalid")
    return (
        step,
        seed_cursor,
        best_inner if isinstance(best_inner, dict) else None,
        best_model_state,
        total_native_steps,
        metrics,
    )


def train_waypoint_dqn(
    config: DQNConfig,
    run_directory: Path,
    manifest: SeedManifest,
    *,
    resume: bool = False,
) -> dict[str, object]:
    """Train and evaluate one waypoint DQN run."""
    config.validate()
    manifest.validate()
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    grid = WaypointGrid(config.grid_spacing)
    controller = WaypointController(grid)
    model = DuelingWaypointDQN(hidden_size=config.hidden_size).to(device)
    target_model = DuelingWaypointDQN(hidden_size=config.hidden_size).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity, WAYPOINT_OBSERVATION_SIZE)
    accumulator = NStepAccumulator(
        config.native_lanes,
        config.n_step,
        config.gamma,
        replay,
    )
    rng = np.random.default_rng(config.seed)
    checkpoint = run_directory / "checkpoint-latest.pt"
    step = 0
    seed_cursor = 0
    best_inner: dict[str, object] | None = None
    best_model_state: dict[str, Tensor] | None = None
    metrics: list[dict[str, object]] = []
    total_native_steps = 0
    if resume:
        if not checkpoint.is_file():
            raise ControlRuntimeError(f"DQN checkpoint does not exist: {checkpoint}")
        (
            step,
            seed_cursor,
            best_inner,
            best_model_state,
            total_native_steps,
            metrics,
        ) = _load_checkpoint(
            checkpoint,
            model,
            target_model,
            optimizer,
            config,
            manifest,
            replay,
            accumulator,
            rng,
        )
    model.train()
    environment = NativeBatchEnvironment(
        step_frames=config.step_frames,
        execution=config.native_execution,
        full_state=False,
        pixels=False,
        board=False,
        ml=True,
        ml_grid_spacing=config.grid_spacing,
    )
    initial_seeds = manifest.training_seeds[: config.native_lanes]
    if len(initial_seeds) < config.native_lanes:
        raise ValueError("native lane count exceeds NG training seed count")
    episode_steps = np.zeros(config.native_lanes, dtype=np.int64)
    current_observations: np.ndarray
    current_positions: np.ndarray
    try:
        result = environment.reset_batch(np.asarray(initial_seeds, dtype=np.uint32))
        current_observations, current_positions = _native_ml_state(result)
        current_observations = current_observations.copy()
        current_positions = current_positions.copy()
        while step < config.total_steps:
            (
                current_observations,
                current_positions,
                seed_cursor,
                native_steps,
                collection,
            ) = _collect_macro_transition(
                environment,
                current_observations,
                current_positions,
                controller,
                model,
                config,
                episode_steps,
                manifest.training_seeds,
                seed_cursor,
                accumulator,
                rng,
                device,
                step,
            )
            total_native_steps += native_steps
            step += 1
            learning: dict[str, float] = {}
            if (
                step >= config.warmup_steps
                and step % config.train_frequency == 0
                and replay.size >= config.batch_size
            ):
                model.train()
                learning = _learn_step(
                    model,
                    target_model,
                    optimizer,
                    replay,
                    config,
                    rng,
                    device,
                )
            if step % config.target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())
            record: dict[str, object] = {
                "step": step,
                "replay_size": replay.size,
                "native_steps": total_native_steps,
                **collection,
                **learning,
            }
            if step % config.eval_every == 0 or step == config.total_steps:
                inner = evaluate_waypoint_dqn(
                    model,
                    manifest.training_seeds[:10],
                    config,
                    grid=grid,
                )
                record["inner_validation"] = inner["summary"]
                inner_mean = float(inner["summary"]["mean_survival_frames"])
                if best_inner is None or inner_mean > float(
                    best_inner["mean_survival_frames"]
                ):
                    best_inner = {
                        "mean_survival_frames": inner_mean,
                        "step": step,
                    }
                    best_model_state = {
                        name: parameter.detach().clone()
                        for name, parameter in model.state_dict().items()
                    }
                    _save_checkpoint(
                        run_directory / "checkpoint-best.pt",
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
                        ),
                    )
                model.train()
            metrics.append(record)
            if step % config.checkpoint_every == 0 or step == config.total_steps:
                _save_checkpoint(
                    checkpoint,
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
                        replay_accumulator=accumulator,
                        rng=rng,
                        total_native_steps=total_native_steps,
                        metrics=metrics,
                    ),
                )
    finally:
        environment.close()
    final_model = model
    selected_model = "final"
    if best_model_state is not None:
        final_model = DuelingWaypointDQN(hidden_size=config.hidden_size).to(device)
        final_model.load_state_dict(best_model_state)
        selected_model = "best_inner"
    final_model.eval()
    final_training = evaluate_waypoint_dqn(
        final_model,
        manifest.training_seeds,
        config,
        grid=grid,
    )
    final_holdout = evaluate_waypoint_dqn(
        final_model,
        manifest.holdout_seeds,
        config,
        grid=grid,
    )
    final_inner = evaluate_waypoint_dqn(
        final_model,
        manifest.training_seeds[:10],
        config,
        grid=grid,
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    for path, value in (
        (run_directory / "metrics.jsonl", metrics),
        (
            run_directory / "run.json",
            {
                "version": WAYPOINT_DQN_VERSION,
                "kind": "dodge_ng_waypoint_dqn_run",
                "manifest_sha256": manifest.sha256,
                "config": config.to_json(),
                "contract": _checkpoint_contract(config),
                "observation_size": WAYPOINT_OBSERVATION_SIZE,
                "observation_contract": (
                    "native_projected_state_with_time_to_intersection+grid_cell+overflow"
                ),
                "observation_source": "native_ml_with_python_reference_parity",
                "grid_shape": list(grid.shape),
                "point_count": grid.point_count,
                "actions": list(ACTION_CHOICES),
                "updates_completed": step,
                "native_steps": total_native_steps,
                "best_inner": best_inner,
                "selected_model": selected_model,
                "final_validation": final_inner,
                "final_training_evaluation": final_training,
                "final_evaluation": final_holdout,
                "target": {
                    "relevance_gate_frames": RELEVANCE_GATE_FRAMES,
                    "safety_limit_frames": (
                        config.max_episode_steps * config.step_frames
                    ),
                    "training_mean_reached": float(
                        final_training["summary"]["mean_survival_frames"]
                    )
                    >= RELEVANCE_GATE_FRAMES,
                },
            },
        ),
    ):
        if path.suffix == ".jsonl":
            path.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in value),
                encoding="utf-8",
            )
        else:
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    _save_checkpoint(
        checkpoint,
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
            replay_accumulator=accumulator,
            rng=rng,
            total_native_steps=total_native_steps,
            metrics=metrics,
        ),
    )
    report = _build_report(run_directory / "run.json", manifest)
    (run_directory / "REPORT.md").write_text(report, encoding="utf-8")
    return json.loads((run_directory / "run.json").read_text())


def _build_report(path: Path, manifest: SeedManifest) -> str:
    run = json.loads(path.read_text(encoding="utf-8"))
    training = run["final_training_evaluation"]["summary"]
    holdout = run["final_evaluation"]["summary"]
    inner = run["final_validation"]["summary"]
    return "\n".join(
        [
            "# Dodge NG waypoint DQN",
            "",
            f"Manifest SHA-256: `{manifest.sha256}`  ",
            f"Grid: `{run['config']['grid_spacing']}` px; "
            f"{run['grid_shape'][0]}x{run['grid_shape'][1]} points  ",
            f"Macro hold: `{run['config']['hold_decisions']}` native decisions  ",
            f"Observation size: `{run['observation_size']}`  ",
            "",
            "| Split | Mean | Median | P10 | Worst | Complete |",
            "|---|---:|---:|---:|---:|---:|",
            _summary_row("Inner", inner),
            _summary_row("Training", training),
            _summary_row("Holdout", holdout),
            "",
            "800-frame relevance gate: **"
            f"{'PASS' if run['target']['training_mean_reached'] else 'FAIL'}**.",
            "",
            "Selection used inner training seeds; holdout was evaluated "
            "after training.",
            "",
        ]
    )


def _summary_row(label: str, summary: dict[str, object]) -> str:
    return (
        f"| {label} | {float(summary['mean_survival_frames']):.1f} | "
        f"{float(summary['median_survival_frames']):.1f} | "
        f"{float(summary['p10_survival_frames']):.1f} | "
        f"{float(summary['worst_survival_frames']):.1f} | "
        f"{float(summary['horizon_completion_fraction']):.1%} |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-dqn")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("history/dodge/ng/waypoint-dqn-20260904")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--grid-spacing", type=int, default=32)
    parser.add_argument("--hold-decisions", type=int, default=8)
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=2_000)
    parser.add_argument("--native-lanes", type=int, default=32)
    parser.add_argument(
        "--native-execution",
        choices=("serial", "parallel"),
        default="parallel",
    )
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2_026_0903)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    arguments = parser.parse_args(argv)
    config = DQNConfig(
        total_steps=arguments.total_steps,
        batch_size=arguments.batch_size,
        replay_capacity=arguments.replay_capacity,
        learning_rate=arguments.learning_rate,
        gamma=arguments.gamma,
        n_step=arguments.n_step,
        warmup_steps=arguments.warmup_steps,
        train_frequency=arguments.train_frequency,
        target_update_interval=arguments.target_update_interval,
        hidden_size=arguments.hidden_size,
        grid_spacing=arguments.grid_spacing,
        hold_decisions=arguments.hold_decisions,
        step_frames=arguments.step_frames,
        max_episode_steps=arguments.max_episode_steps,
        native_lanes=arguments.native_lanes,
        native_execution=arguments.native_execution,
        checkpoint_every=arguments.checkpoint_every,
        eval_every=arguments.eval_every,
        seed=arguments.seed,
        device=arguments.device,
    )
    try:
        run = train_waypoint_dqn(
            config,
            arguments.run_dir,
            load_manifest(arguments.manifest),
            resume=arguments.resume,
        )
    except (ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-ng-dqn: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_directory": str(arguments.run_dir),
                "manifest_sha256": run["manifest_sha256"],
                "updates_completed": run["updates_completed"],
                "training_gate": run["target"]["training_mean_reached"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
