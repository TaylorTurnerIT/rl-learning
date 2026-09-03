from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from dodge.control import PROJECT_ROOT, ControlInputError, ControlRuntimeError
from dodge.dataset import (
    ACTION_CHOICES,
    DEVELOPMENT_VALIDATION_SEEDS,
    EVALUATION_SEEDS,
    TRAINING_SEED_MAX,
)
from dodge.imitation.board import BOARD_CHANNELS, BOARD_SHAPE, encode_board
from dodge.native.batch import NativeBatchEnvironment, NativeBatchResult
from dodge.native.differential import FRAME_HEIGHT, FRAME_WIDTH
from dodge.neat.bridge import Direction
from dodge.neat.environment import DodgeEnv, Observation, Transition
from dodge.neat.state import RawState

PPO_CHECKPOINT_VERSION = 1
PPO_RUN_VERSION = 1
PPO_MODEL_TYPE = "DodgeActorCriticCNN"
PIXEL_PPO_MODEL_TYPE = "DodgePixelActorCriticCNN"
PIXEL_PALETTE_MAX = 15.0
DEFAULT_RUN_ROOT = PROJECT_ROOT / "history" / "dodge" / "ppo"
PPO_RUNTIME_DIRECTORY_NAME = ".runtime"
MIN_RUNTIME_FREE_BYTES = 512 * 1024 * 1024
TRAINING_SEEDS = tuple(
    seed
    for seed in range(TRAINING_SEED_MAX + 1)
    if seed not in DEVELOPMENT_VALIDATION_SEEDS
)

PPOBackend = Literal["python", "native"]
NativeExecution = Literal["serial", "parallel"]
NativeObservationMode = Literal["board", "pixels"]
PixelArchitecture = Literal["fast", "small", "current"]


class Environment(Protocol):
    def reset(self, seed: int | None = None) -> Observation: ...

    def step(self, action: Direction) -> Transition: ...

    def close(self) -> None: ...


EnvironmentFactory = Callable[..., Environment]


@dataclass(frozen=True, slots=True)
class PPOConfig:
    """PPO hyperparameters and the Dodge interaction contract."""

    updates: int = 1_000
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 2.5e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    neutral_bonus: float = 0.02
    stability_bonus_cap: float = 1.0
    step_frames: int = 4
    max_episode_steps: int = 2_000
    environment_restarts_per_rollout: int = 3
    checkpoint_every: int = 10
    eval_every: int = 10
    seed: int = 42
    device: str = "auto"
    backend: PPOBackend = "python"
    native_lanes: int = 32
    native_execution: NativeExecution = "parallel"
    observation_mode: NativeObservationMode = "board"
    pixel_stack: int = 4
    pixel_architecture: PixelArchitecture = "small"
    training_seeds: tuple[int, ...] = ()
    training_seed_manifest: str | None = None

    def validate(self) -> None:
        if any(
            value < 1
            for value in (
                self.updates,
                self.rollout_steps,
                self.update_epochs,
                self.minibatch_size,
                self.max_episode_steps,
                self.checkpoint_every,
            )
        ):
            raise ValueError(
                "updates, rollout, epochs, batches, limits must be positive"
            )
        if self.environment_restarts_per_rollout < 0:
            raise ValueError("environment restarts must not be negative")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("GAE lambda must be between 0 and 1")
        if self.clip_coef <= 0:
            raise ValueError("PPO clip coefficient must be positive")
        if self.entropy_coef < 0 or self.value_coef < 0:
            raise ValueError("entropy and value coefficients must not be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("maximum gradient norm must be positive")
        if self.neutral_bonus < 0 or self.stability_bonus_cap < 0:
            raise ValueError("neutral bonus and stability cap must not be negative")
        if not 3 <= self.step_frames <= 5:
            raise ValueError("step frames must be between 3 and 5")
        if self.eval_every < 0:
            raise ValueError("evaluation interval must not be negative")
        if self.backend not in {"python", "native"}:
            raise ValueError("backend must be 'python' or 'native'")
        if self.native_lanes < 1:
            raise ValueError("native lane count must be positive")
        if self.native_execution not in {"serial", "parallel"}:
            raise ValueError("native execution must be 'serial' or 'parallel'")
        if self.observation_mode not in {"board", "pixels"}:
            raise ValueError("observation mode must be 'board' or 'pixels'")
        if not 1 <= self.pixel_stack <= 8:
            raise ValueError("pixel stack must be between 1 and 8")
        if self.pixel_architecture not in {"fast", "small", "current"}:
            raise ValueError("pixel architecture must be 'fast', 'small', or 'current'")
        if self.observation_mode == "pixels" and self.backend != "native":
            raise ValueError("pixel PPO requires the native backend")
        if self.training_seed_manifest is not None and not isinstance(
            self.training_seed_manifest, str
        ):
            raise ValueError("training seed manifest must be a string or null")
        if self.training_seeds:
            if len(set(self.training_seeds)) != len(self.training_seeds):
                raise ValueError("training seeds must be unique")
            if any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in self.training_seeds
            ):
                raise ValueError("training seeds must be nonnegative integers")
            if any(seed > 32_767 for seed in self.training_seeds):
                raise ValueError("training seeds must fit the native seed range")
        if self.backend == "native":
            if self.native_lanes > self.rollout_steps:
                raise ValueError("native lane count must not exceed rollout steps")
            if self.rollout_steps % self.native_lanes:
                raise ValueError(
                    "native rollout steps must be divisible by native lane count"
                )

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        value["training_seeds"] = list(self.training_seeds)
        return value


class BoardFeatureEncoder(nn.Module):
    """Extract spatial features from the complete raw Dodge board."""

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden size must be positive")
        self.convolution = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(128 * 2 * 2, hidden_size),
            nn.ReLU(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        _validate_board_batch(observations)
        return self.projection(self.convolution(observations))


class DodgeActorCriticCNN(nn.Module):
    """CNN policy plus value function for direct survival optimization."""

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.features = BoardFeatureEncoder(hidden_size)
        self.policy_head = nn.Linear(hidden_size, len(ACTION_CHOICES))
        self.value_head = nn.Linear(hidden_size, 1)
        self._initialize_weights()

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        features = self.features(observations)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def get_action_and_value(
        self, observations: Tensor, *, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, values = self(observations)
        distribution = Categorical(logits=logits)
        actions = logits.argmax(dim=1) if deterministic else distribution.sample()
        return (
            actions,
            distribution.log_prob(actions),
            distribution.entropy(),
            values,
        )

    def evaluate_actions(
        self, observations: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits, values = self(observations)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), values

    def load_actor_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        """Load compatible actor weights while resetting the value head."""
        current = self.state_dict()
        actor_keys = {
            name
            for name in current
            if name.startswith("features.") or name.startswith("policy_head.")
        }
        supplied = {
            name: value for name, value in state_dict.items() if name in actor_keys
        }
        missing = actor_keys - supplied.keys()
        if missing:
            raise ValueError(
                f"actor warm start is missing compatible weights: {sorted(missing)[:3]}"
            )
        try:
            self.load_state_dict(supplied, strict=False)
        except (RuntimeError, TypeError) as error:
            raise ValueError(
                f"actor warm start has incompatible weights: {error}"
            ) from error
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        if self.value_head.bias is not None:
            nn.init.zeros_(self.value_head.bias)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)


class PixelFeatureEncoder(nn.Module):
    """Extract temporal-spatial features from native indexed pixels."""

    def __init__(
        self,
        stack_size: int = 4,
        hidden_size: int = 128,
        architecture: PixelArchitecture = "small",
    ) -> None:
        super().__init__()
        if not 1 <= stack_size <= 8:
            raise ValueError("pixel stack must be between 1 and 8")
        if hidden_size < 1:
            raise ValueError("hidden size must be positive")
        if architecture not in {"fast", "small", "current"}:
            raise ValueError("pixel architecture must be 'fast', 'small', or 'current'")
        self.stack_size = stack_size
        self.architecture = architecture
        if architecture == "fast":
            first_channels, second_channels, third_channels = (16, 32, 64)
            self.convolution = nn.Sequential(
                nn.Conv2d(stack_size, first_channels, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(first_channels, second_channels, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(second_channels, third_channels, kernel_size=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
            )
        else:
            first_channels, second_channels, third_channels = (
                (16, 32, 64) if architecture == "small" else (32, 64, 128)
            )
            self.convolution = nn.Sequential(
                nn.Conv2d(
                    stack_size,
                    first_channels,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(first_channels, second_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(second_channels, third_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
            )
        self.projection = nn.Sequential(
            nn.Linear(third_channels * 2 * 2, hidden_size),
            nn.ReLU(),
        )

    def forward(self, observations: Tensor) -> Tensor:
        _validate_pixel_batch(observations, self.stack_size)
        normalized = observations.float() / PIXEL_PALETTE_MAX
        return self.projection(self.convolution(normalized))


class PixelActorCriticCNN(nn.Module):
    """CNN policy plus value function over an exact indexed-pixel stack."""

    def __init__(
        self,
        stack_size: int = 4,
        hidden_size: int = 128,
        architecture: PixelArchitecture = "small",
    ) -> None:
        super().__init__()
        self.stack_size = stack_size
        self.architecture = architecture
        self.features = PixelFeatureEncoder(stack_size, hidden_size, architecture)
        self.policy_head = nn.Linear(hidden_size, len(ACTION_CHOICES))
        self.value_head = nn.Linear(hidden_size, 1)
        self._initialize_weights()

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        features = self.features(observations)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def get_action_and_value(
        self, observations: Tensor, *, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, values = self(observations)
        distribution = Categorical(logits=logits)
        actions = logits.argmax(dim=1) if deterministic else distribution.sample()
        return (
            actions,
            distribution.log_prob(actions),
            distribution.entropy(),
            values,
        )

    def evaluate_actions(
        self, observations: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits, values = self(observations)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), values

    def load_actor_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        """Load compatible actor weights while resetting the value head."""
        current = self.state_dict()
        actor_keys = {
            name
            for name in current
            if name.startswith("features.") or name.startswith("policy_head.")
        }
        supplied = {
            name: value for name, value in state_dict.items() if name in actor_keys
        }
        missing = actor_keys - supplied.keys()
        if missing:
            raise ValueError(
                f"actor warm start is missing compatible weights: {sorted(missing)[:3]}"
            )
        try:
            self.load_state_dict(supplied, strict=False)
        except (RuntimeError, TypeError) as error:
            raise ValueError(
                f"actor warm start has incompatible weights: {error}"
            ) from error
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        if self.value_head.bias is not None:
            nn.init.zeros_(self.value_head.bias)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)


def _validate_board_batch(observations: Tensor) -> None:
    if observations.ndim != 4 or tuple(observations.shape[1:]) != BOARD_SHAPE:
        raise ValueError(
            "observations must have shape "
            f"(N, {BOARD_SHAPE[0]}, {BOARD_SHAPE[1]}, {BOARD_SHAPE[2]})"
        )


def _validate_pixel_batch(observations: Tensor, stack_size: int) -> None:
    expected_shape = (stack_size, FRAME_HEIGHT, FRAME_WIDTH)
    if observations.ndim != 4 or tuple(observations.shape[1:]) != expected_shape:
        raise ValueError(
            "pixel observations must have shape "
            f"(N, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]})"
        )


def _model_for_config(config: PPOConfig) -> nn.Module:
    if config.observation_mode == "pixels":
        return PixelActorCriticCNN(
            stack_size=config.pixel_stack,
            architecture=config.pixel_architecture,
        )
    return DodgeActorCriticCNN()


def _model_type_for_config(config: PPOConfig) -> str:
    return (
        PIXEL_PPO_MODEL_TYPE if config.observation_mode == "pixels" else PPO_MODEL_TYPE
    )


def _observation_shape_for_config(config: PPOConfig) -> tuple[int, int, int]:
    if config.observation_mode == "pixels":
        return (config.pixel_stack, FRAME_HEIGHT, FRAME_WIDTH)
    return BOARD_SHAPE


def board_tensor(state: RawState, device: torch.device | None = None) -> Tensor:
    tensor = torch.from_numpy(encode_board(state))
    return tensor if device is None else tensor.to(device)


def action_from_index(index: int) -> Direction:
    if not 0 <= index < len(ACTION_CHOICES):
        raise ValueError(
            f"action index must be between 0 and {len(ACTION_CHOICES) - 1}"
        )
    return cast(Direction, ACTION_CHOICES[index])


@dataclass(slots=True)
class StabilityReward:
    """Small capped neutral preference that cannot outweigh survival."""

    neutral_bonus: float
    cap: float
    used: float = 0.0

    def __post_init__(self) -> None:
        if self.neutral_bonus < 0 or self.cap < 0:
            raise ValueError("neutral bonus and cap must not be negative")

    def reset(self) -> None:
        self.used = 0.0

    def apply(self, survival_reward: float, action: Direction) -> float:
        if action != "neutral" or self.used >= self.cap:
            return float(survival_reward)
        bonus = min(self.neutral_bonus, self.cap - self.used)
        self.used += bonus
        return float(survival_reward) + bonus


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    observations: Tensor
    actions: Tensor
    old_log_probs: Tensor
    values: Tensor
    rewards: Tensor
    next_values: Tensor
    terminated: Tensor
    episode_ends: Tensor
    advantages: Tensor
    returns: Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    episode_ends: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    tensors = (rewards, values, next_values, terminated, episode_ends)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("GAE inputs must be one-dimensional")
    if len({len(tensor) for tensor in tensors}) != 1:
        raise ValueError("GAE inputs must have equal lengths")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 1.0 - terminated[index].to(rewards.dtype)
        delta = (
            rewards[index] + gamma * nonterminal * next_values[index] - values[index]
        )
        continue_gae = 1.0 - episode_ends[index].to(rewards.dtype)
        running = delta + gamma * gae_lambda * continue_gae * running
        advantages[index] = running
    return advantages, advantages + values


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    seed: int
    survival_frames: int
    steps: int
    neutral_actions: int
    terminated: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    seeds: tuple[int, ...]
    survival_frames: tuple[int, ...]
    terminated: tuple[bool, ...]

    @property
    def mean_survival_frames(self) -> float:
        return sum(self.survival_frames) / len(self.survival_frames)

    @property
    def best_survival_frames(self) -> int:
        return max(self.survival_frames)

    def to_json(self) -> dict[str, object]:
        return {
            "seeds": list(self.seeds),
            "survival_frames": list(self.survival_frames),
            "terminated": list(self.terminated),
            "mean_survival_frames": self.mean_survival_frames,
            "best_survival_frames": self.best_survival_frames,
        }


class TrainingSeedStream:
    def __init__(self, seed: int, candidates: Sequence[int] = TRAINING_SEEDS) -> None:
        self._random = random.Random(seed)
        self._candidates = tuple(candidates)
        if not self._candidates:
            raise ValueError("training seed candidates must not be empty")

    def next(self) -> int:
        return self._random.choice(self._candidates)

    def getstate(self) -> object:
        return self._random.getstate()

    def setstate(self, state: object) -> None:
        self._random.setstate(cast(tuple[object, ...], state))


class PPOTrainer:
    """Collect real Dodge transitions and optimize a CNN actor-critic."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        environment_factory: EnvironmentFactory = DodgeEnv,
        checkpoint: Path | None = None,
        runtime_directory: Path | None = None,
        initial_actor_state: Mapping[str, Tensor] | None = None,
        initialization: Mapping[str, object] | None = None,
    ) -> None:
        config.validate()
        if config.backend != "python":
            raise ValueError("native PPO configs require NativePPOTrainer")
        self.config = config
        self.device = _resolve_device(config.device)
        _seed_torch(config.seed)
        self.model = _model_for_config(config).to(self.device)
        self.initialization = (
            dict(initialization) if initialization is not None else None
        )
        if initial_actor_state is not None:
            self.model.load_actor_state_dict(initial_actor_state)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self._environment_factory = environment_factory
        self.runtime_directory = runtime_directory
        self._environment = _make_environment(
            environment_factory,
            step_frames=config.step_frames,
            temporary_root=runtime_directory,
        )
        self._seed_stream = TrainingSeedStream(
            config.seed, config.training_seeds or TRAINING_SEEDS
        )
        self._observation: Observation | None = None
        self._episode_seed: int | None = None
        self._episode_steps = 0
        self._episode_survival = 0.0
        self._episode_neutral_actions = 0
        self._stability_reward = StabilityReward(
            config.neutral_bonus, config.stability_bonus_cap
        )
        self.updates_completed = 0
        self.global_step = 0
        self.episodes_completed = 0
        self.environment_errors = 0
        self.best_validation: dict[str, object] | None = None
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

    def collect_rollout(self) -> tuple[RolloutBatch, tuple[EpisodeSummary, ...]]:
        observations: list[Tensor] = []
        actions: list[int] = []
        old_log_probs: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        next_values: list[float] = []
        terminated: list[bool] = []
        episode_ends: list[bool] = []
        episodes: list[EpisodeSummary] = []
        restarts = 0
        self.model.eval()
        while len(rewards) < self.config.rollout_steps:
            self._ensure_episode()
            assert self._observation is not None
            current_board = board_tensor(self._observation.raw_state, self.device)
            try:
                with torch.inference_mode():
                    action_tensor, log_prob, _, value = self.model.get_action_and_value(
                        current_board.unsqueeze(0)
                    )
                action_index = int(action_tensor.item())
                action = action_from_index(action_index)
                transition = self._environment.step(action)
            except ControlRuntimeError:
                self.environment_errors += 1
                restarts += 1
                self._restart_episode()
                if restarts > self.config.environment_restarts_per_rollout:
                    raise
                continue

            restarts = 0
            self._episode_steps += 1
            self._episode_survival += transition.reward
            self._episode_neutral_actions += int(action == "neutral")
            shaped_reward = self._stability_reward.apply(transition.reward, action)
            actual_terminal = transition.done
            truncated = (
                not actual_terminal
                and self._episode_steps >= self.config.max_episode_steps
            )
            episode_end = actual_terminal or truncated
            if actual_terminal:
                bootstrap_value = 0.0
            else:
                next_board = board_tensor(transition.observation.raw_state, self.device)
                with torch.inference_mode():
                    _, next_value_tensor = self.model(next_board.unsqueeze(0))
                bootstrap_value = float(next_value_tensor.item())

            observations.append(board_tensor(self._observation.raw_state))
            actions.append(action_index)
            old_log_probs.append(float(log_prob.item()))
            values.append(float(value.item()))
            rewards.append(shaped_reward)
            next_values.append(bootstrap_value)
            terminated.append(actual_terminal)
            episode_ends.append(episode_end)
            self.global_step += 1

            if episode_end:
                if transition.result is not None:
                    survival_frames = transition.result.survival_frames
                else:
                    survival_frames = int(round(self._episode_survival))
                episodes.append(
                    EpisodeSummary(
                        seed=self._episode_seed_or_error(),
                        survival_frames=survival_frames,
                        steps=self._episode_steps,
                        neutral_actions=self._episode_neutral_actions,
                        terminated=actual_terminal,
                    )
                )
                self.episodes_completed += 1
                self._begin_episode()
            else:
                self._observation = transition.observation

        self.model.train()
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        value_tensor = torch.tensor(values, dtype=torch.float32)
        next_value_tensor = torch.tensor(next_values, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminated, dtype=torch.bool)
        episode_end_tensor = torch.tensor(episode_ends, dtype=torch.bool)
        advantages, returns = compute_gae(
            reward_tensor,
            value_tensor,
            next_value_tensor,
            terminated_tensor,
            episode_end_tensor,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        return (
            RolloutBatch(
                observations=torch.stack(observations),
                actions=torch.tensor(actions, dtype=torch.long),
                old_log_probs=torch.tensor(old_log_probs, dtype=torch.float32),
                values=value_tensor,
                rewards=reward_tensor,
                next_values=next_value_tensor,
                terminated=terminated_tensor,
                episode_ends=episode_end_tensor,
                advantages=advantages,
                returns=returns,
            ),
            tuple(episodes),
        )

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        self.model.train()
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        size = len(batch.actions)
        minibatch_size = min(self.config.minibatch_size, size)
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        minibatches = 0
        for _ in range(self.config.update_epochs):
            indices = torch.randperm(size)
            for start in range(0, size, minibatch_size):
                selected = indices[start : start + minibatch_size]
                observations = batch.observations[selected].to(self.device)
                selected_actions = batch.actions[selected].to(self.device)
                selected_old_log_probs = batch.old_log_probs[selected].to(self.device)
                selected_old_values = batch.values[selected].to(self.device)
                selected_advantages = advantages[selected].to(self.device)
                selected_returns = batch.returns[selected].to(self.device)
                log_probs, entropy, values = self.model.evaluate_actions(
                    observations, selected_actions
                )
                log_ratio = log_probs - selected_old_log_probs
                ratio = log_ratio.exp()
                unclipped = ratio * selected_advantages
                clipped = (
                    ratio.clamp(
                        1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef
                    )
                    * selected_advantages
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_delta = values - selected_old_values
                clipped_values = selected_old_values + value_delta.clamp(
                    -self.config.clip_coef, self.config.clip_coef
                )
                value_losses = (values - selected_returns).square()
                clipped_value_losses = (clipped_values - selected_returns).square()
                value_loss = (
                    0.5 * torch.maximum(value_losses, clipped_value_losses).mean()
                )
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_mean
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                totals["policy_loss"] += float(policy_loss.detach().cpu())
                totals["value_loss"] += float(value_loss.detach().cpu())
                totals["entropy"] += float(entropy_mean.detach().cpu())
                totals["approx_kl"] += float(
                    ((ratio - 1.0) - log_ratio).mean().detach().cpu()
                )
                totals["clip_fraction"] += float(
                    ((ratio - 1.0).abs() > self.config.clip_coef)
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )
                minibatches += 1
        result = {key: value / minibatches for key, value in totals.items()}
        with torch.inference_mode():
            _, predicted_values = self.model(batch.observations.to(self.device))
        result["explained_variance"] = _explained_variance(
            batch.returns, predicted_values.cpu()
        )
        result["advantage_mean"] = float(batch.advantages.mean())
        result["return_mean"] = float(batch.returns.mean())
        return result

    def pause_environment(self) -> None:
        self._environment.close()
        self._observation = None

    def save_checkpoint(self, path: Path) -> None:
        payload = self._checkpoint_payload()
        _atomic_torch_save(payload, path)

    def close(self) -> None:
        self._environment.close()

    def _ensure_episode(self) -> None:
        if self._observation is None:
            self._begin_episode()

    def _begin_episode(self) -> None:
        seed = self._seed_stream.next()
        self._episode_seed = seed
        self._observation = self._environment.reset(seed=seed)
        self._episode_steps = 0
        self._episode_survival = 0.0
        self._episode_neutral_actions = 0
        self._stability_reward.reset()

    def _restart_episode(self) -> None:
        self._environment.close()
        self._observation = None
        self._begin_episode()

    def _episode_seed_or_error(self) -> int:
        if self._episode_seed is None:
            raise RuntimeError("episode seed is not initialized")
        return self._episode_seed

    def _checkpoint_payload(self) -> dict[str, object]:
        observation_shape = _observation_shape_for_config(self.config)
        pixels_mode = self.config.observation_mode == "pixels"
        return {
            "version": PPO_CHECKPOINT_VERSION,
            "model_type": _model_type_for_config(self.config),
            "observation_mode": self.config.observation_mode,
            "observation_shape": list(observation_shape),
            "board_shape": list(BOARD_SHAPE) if not pixels_mode else None,
            "pixel_shape": list(observation_shape) if pixels_mode else None,
            "actions": list(ACTION_CHOICES),
            "config": self.config.to_json(),
            "updates_completed": self.updates_completed,
            "global_step": self.global_step,
            "episodes_completed": self.episodes_completed,
            "environment_errors": self.environment_errors,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in self.model.state_dict().items()
            },
            "optimizer_state_dict": _to_cpu(self.optimizer.state_dict()),
            "seed_stream_state": self._seed_stream.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "best_validation": self.best_validation,
            "initialization": self.initialization,
        }

    def _load_checkpoint(self, path: Path) -> None:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except (OSError, RuntimeError) as error:
            raise ControlRuntimeError(
                f"could not load PPO checkpoint: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise ControlRuntimeError("PPO checkpoint must be an object")
        expected_shape = _observation_shape_for_config(self.config)
        expected_pixels = self.config.observation_mode == "pixels"
        stored_mode = payload.get("observation_mode", "board")
        stored_shape = payload.get("observation_shape")
        mode_metadata_matches = stored_mode == self.config.observation_mode
        shape_metadata_matches = stored_shape is None or tuple(stored_shape) == (
            expected_shape
        )
        if expected_pixels:
            shape_metadata_matches = (
                tuple(payload.get("pixel_shape", ())) == (expected_shape)
                and shape_metadata_matches
            )
        else:
            shape_metadata_matches = (
                tuple(payload.get("board_shape", ())) == (BOARD_SHAPE)
                and shape_metadata_matches
            )
        if (
            payload.get("version") != PPO_CHECKPOINT_VERSION
            or payload.get("model_type") != _model_type_for_config(self.config)
            or not mode_metadata_matches
            or not shape_metadata_matches
            or tuple(payload.get("actions", ())) != ACTION_CHOICES
        ):
            raise ControlRuntimeError("PPO checkpoint has incompatible model metadata")
        stored_config = payload.get("config")
        if not isinstance(stored_config, Mapping):
            raise ControlRuntimeError("PPO checkpoint is missing its configuration")
        _validate_resume_config(stored_config, self.config)
        try:
            self.model.load_state_dict(payload["model_state_dict"])
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self._seed_stream.setstate(payload["seed_stream_state"])
            torch.set_rng_state(payload["torch_rng_state"])
            cuda_rng_state = payload.get("cuda_rng_state")
            if cuda_rng_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_rng_state)
            self.updates_completed = int(payload["updates_completed"])
            self.global_step = int(payload["global_step"])
            self.episodes_completed = int(payload["episodes_completed"])
            self.environment_errors = int(payload.get("environment_errors", 0))
            best_validation = payload.get("best_validation")
            self.best_validation = (
                dict(best_validation) if isinstance(best_validation, Mapping) else None
            )
            initialization = payload.get("initialization")
            self.initialization = (
                dict(initialization) if isinstance(initialization, Mapping) else None
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise ControlRuntimeError(
                f"PPO checkpoint state is invalid: {error}"
            ) from error
        _optimizer_to_device(self.optimizer, self.device)


class NativePPOTrainer(PPOTrainer):
    """PPO trainer using one Rust batch call for every environment horizon."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        checkpoint: Path | None = None,
        runtime_directory: Path | None = None,
        initial_actor_state: Mapping[str, Tensor] | None = None,
        initialization: Mapping[str, object] | None = None,
    ) -> None:
        config.validate()
        if config.backend != "native":
            raise ValueError("NativePPOTrainer requires backend='native'")
        self.config = config
        self.device = _resolve_device(config.device)
        _seed_torch(config.seed)
        self.model = _model_for_config(config).to(self.device)
        self.initialization = (
            dict(initialization) if initialization is not None else None
        )
        if initial_actor_state is not None:
            self.model.load_actor_state_dict(initial_actor_state)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.runtime_directory = runtime_directory
        self._environment = self._new_environment()
        self._seed_stream = TrainingSeedStream(
            config.seed, config.training_seeds or TRAINING_SEEDS
        )
        self._boards: np.ndarray | None = None
        self._pixels: np.ndarray | None = None
        self._lane_seeds: list[int] = []
        self._episode_steps: list[int] = []
        self._episode_survival: list[float] = []
        self._episode_neutral_actions: list[int] = []
        self._stability_rewards: list[StabilityReward] = []
        self.updates_completed = 0
        self.global_step = 0
        self.episodes_completed = 0
        self.environment_errors = 0
        self.best_validation: dict[str, object] | None = None
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

    def collect_rollout(self) -> tuple[RolloutBatch, tuple[EpisodeSummary, ...]]:
        observations: list[Tensor] = []
        actions: list[int] = []
        old_log_probs: list[float] = []
        values: list[float] = []
        rewards: list[float] = []
        next_values: list[float] = []
        terminated: list[bool] = []
        episode_ends: list[bool] = []
        episodes: list[EpisodeSummary] = []
        restarts = 0
        self.model.eval()
        while len(rewards) < self.config.rollout_steps:
            self._ensure_lanes()
            current_observations = self._current_observations()
            if current_observations is None:
                raise ControlRuntimeError("native PPO lanes were not initialized")
            current_tensor = torch.from_numpy(current_observations)
            with torch.inference_mode():
                action_tensor, log_prob_tensor, _, value_tensor = (
                    self.model.get_action_and_value(current_tensor.to(self.device))
                )
            action_indices = (
                action_tensor.detach().cpu().numpy().astype(np.uint8, copy=True)
            )
            try:
                result = self._environment.step_batch(action_indices)
            except ControlRuntimeError:
                self.environment_errors += 1
                restarts += 1
                self._restart_native_environment()
                if restarts > self.config.environment_restarts_per_rollout:
                    raise
                continue

            restarts = 0
            next_observations = self._next_observations(result)
            next_tensor = torch.from_numpy(next_observations)
            with torch.inference_mode():
                _, next_value_tensor = self.model(next_tensor.to(self.device))
            ended_lanes: list[int] = []
            for lane in range(len(self._lane_seeds)):
                action_index = int(action_indices[lane])
                action = action_from_index(action_index)
                survival_reward = float(result.rewards[lane])
                self._episode_steps[lane] += 1
                self._episode_survival[lane] += survival_reward
                self._episode_neutral_actions[lane] += int(action == "neutral")
                shaped_reward = self._stability_rewards[lane].apply(
                    survival_reward, action
                )
                actual_terminal = bool(result.done[lane])
                truncated = (
                    not actual_terminal
                    and self._episode_steps[lane] >= self.config.max_episode_steps
                )
                episode_end = actual_terminal or truncated
                bootstrap_value = (
                    0.0 if actual_terminal else float(next_value_tensor[lane].item())
                )

                observations.append(current_tensor[lane].clone())
                actions.append(action_index)
                old_log_probs.append(float(log_prob_tensor[lane].item()))
                values.append(float(value_tensor[lane].item()))
                rewards.append(shaped_reward)
                next_values.append(bootstrap_value)
                terminated.append(actual_terminal)
                episode_ends.append(episode_end)
                if episode_end:
                    episodes.append(
                        EpisodeSummary(
                            seed=self._lane_seeds[lane],
                            survival_frames=int(round(self._episode_survival[lane])),
                            steps=self._episode_steps[lane],
                            neutral_actions=self._episode_neutral_actions[lane],
                            terminated=actual_terminal,
                        )
                    )
                    self.episodes_completed += 1
                    ended_lanes.append(lane)

            self.global_step += len(self._lane_seeds)
            self._set_current_observations(next_observations)
            if ended_lanes:
                self._reset_lanes(ended_lanes)

        self.model.train()
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        value_tensor = torch.tensor(values, dtype=torch.float32)
        next_value_tensor = torch.tensor(next_values, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminated, dtype=torch.bool)
        episode_end_tensor = torch.tensor(episode_ends, dtype=torch.bool)
        advantages, returns = compute_gae(
            reward_tensor,
            value_tensor,
            next_value_tensor,
            terminated_tensor,
            episode_end_tensor,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        return (
            RolloutBatch(
                observations=torch.stack(observations),
                actions=torch.tensor(actions, dtype=torch.long),
                old_log_probs=torch.tensor(old_log_probs, dtype=torch.float32),
                values=value_tensor,
                rewards=reward_tensor,
                next_values=next_value_tensor,
                terminated=terminated_tensor,
                episode_ends=episode_end_tensor,
                advantages=advantages,
                returns=returns,
            ),
            tuple(episodes),
        )

    def pause_environment(self) -> None:
        self._environment.close()
        self._boards = None
        self._pixels = None

    def close(self) -> None:
        self._environment.close()

    def _new_environment(self) -> NativeBatchEnvironment:
        return NativeBatchEnvironment(
            step_frames=self.config.step_frames,
            execution=self.config.native_execution,
            full_state=False,
            pixels=self.config.observation_mode == "pixels",
            board=self.config.observation_mode == "board",
        )

    def _ensure_lanes(self) -> None:
        if self._current_observations() is not None:
            return
        if self._environment.closed:
            self._environment = self._new_environment()
        seeds = [self._seed_stream.next() for _ in range(self.config.native_lanes)]
        result = self._environment.reset_batch(np.asarray(seeds, dtype=np.uint32))
        if result.lane_ids.tolist() != list(range(len(seeds))):
            raise ControlRuntimeError("native reset returned unexpected lane order")
        if self.config.observation_mode == "pixels":
            self._pixels = _initial_pixel_stack(
                _native_pixels(result), self.config.pixel_stack
            )
            self._boards = None
        else:
            self._boards = np.array(_native_board(result), dtype=np.float32, copy=True)
            self._pixels = None
        self._lane_seeds = seeds
        self._episode_steps = [0] * len(seeds)
        self._episode_survival = [0.0] * len(seeds)
        self._episode_neutral_actions = [0] * len(seeds)
        self._stability_rewards = [
            StabilityReward(self.config.neutral_bonus, self.config.stability_bonus_cap)
            for _ in seeds
        ]

    def _reset_lanes(self, lanes: Sequence[int]) -> None:
        seeds = [self._seed_stream.next() for _ in lanes]
        result = self._environment.reset_lanes(
            np.asarray(lanes, dtype=np.uint32),
            np.asarray(seeds, dtype=np.uint32),
        )
        if self._current_observations() is None:
            raise ControlRuntimeError("native reset lost the active observations")
        reset_pixels = (
            _native_pixels(result) if self.config.observation_mode == "pixels" else None
        )
        reset_boards = (
            _native_board(result) if self.config.observation_mode == "board" else None
        )
        for position, lane_value in enumerate(result.lane_ids.tolist()):
            lane = int(lane_value)
            if self.config.observation_mode == "pixels":
                if self._pixels is None:
                    raise ControlRuntimeError("native reset lost the pixel batch")
                assert reset_pixels is not None
                self._pixels[lane] = reset_pixels[position]
            else:
                if self._boards is None:
                    raise ControlRuntimeError("native reset lost the board batch")
                assert reset_boards is not None
                self._boards[lane] = reset_boards[position]
            self._lane_seeds[lane] = seeds[position]
            self._episode_steps[lane] = 0
            self._episode_survival[lane] = 0.0
            self._episode_neutral_actions[lane] = 0
            self._stability_rewards[lane].reset()

    def _restart_native_environment(self) -> None:
        self._environment.close()
        self._environment = self._new_environment()
        self._boards = None
        self._pixels = None

    def _current_observations(self) -> np.ndarray | None:
        return (
            self._pixels if self.config.observation_mode == "pixels" else self._boards
        )

    def _next_observations(self, result: NativeBatchResult) -> np.ndarray:
        if self.config.observation_mode == "pixels":
            if self._pixels is None:
                raise ControlRuntimeError("native PPO lost the current pixel stack")
            return _advance_pixel_stack(self._pixels, _native_pixels(result))
        return np.array(_native_board(result), dtype=np.float32, copy=True)

    def _set_current_observations(self, observations: np.ndarray) -> None:
        if self.config.observation_mode == "pixels":
            self._pixels = np.array(observations, dtype=np.uint8, copy=True)
            self._boards = None
        else:
            self._boards = np.array(observations, dtype=np.float32, copy=True)
            self._pixels = None


def _native_board(result: NativeBatchResult) -> np.ndarray:
    if result.board is None:
        raise ControlRuntimeError("native PPO requires board observations")
    return result.board


def _native_pixels(result: NativeBatchResult) -> np.ndarray:
    if result.pixels is None:
        raise ControlRuntimeError("native PPO requires pixel observations")
    pixels = result.pixels
    expected_shape = (pixels.shape[0], FRAME_HEIGHT, FRAME_WIDTH)
    if pixels.shape != expected_shape or pixels.dtype != np.uint8:
        raise ControlRuntimeError(
            "native PPO pixel observations have unexpected shape or dtype: "
            f"expected {expected_shape} and uint8, got {pixels.shape} and "
            f"{pixels.dtype}"
        )
    if pixels.size and (int(pixels.min()) < 0 or int(pixels.max()) > 15):
        raise ControlRuntimeError("native PPO pixels contain an invalid palette index")
    return pixels


def _initial_pixel_stack(pixels: np.ndarray, stack_size: int) -> np.ndarray:
    return np.repeat(pixels[:, None, :, :], stack_size, axis=1)


def _advance_pixel_stack(
    current_stack: np.ndarray, next_pixels: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        (current_stack[:, 1:, :, :], next_pixels[:, None, :, :]), axis=1
    )


def _to_cpu(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _validate_resume_config(stored: Mapping[str, object], current: PPOConfig) -> None:
    current_values = current.to_json()
    legacy_defaults = {
        "backend": "python",
        "native_lanes": 32,
        "native_execution": "parallel",
        "observation_mode": "board",
        "pixel_stack": 4,
        "pixel_architecture": "small",
        "training_seeds": [],
        "training_seed_manifest": None,
    }
    mismatches = []
    for key, value in current_values.items():
        if key in {"updates", "device"}:
            continue
        if key not in stored and legacy_defaults.get(key) == value:
            continue
        if stored.get(key) != value:
            mismatches.append(key)
    completed = stored.get("updates", 0)
    if not isinstance(completed, int) or current.updates < completed:
        mismatches.append("updates")
    if mismatches:
        raise ControlInputError(
            "PPO resume configuration differs: " + ", ".join(mismatches)
        )


def _explained_variance(returns: Tensor, predictions: Tensor) -> float:
    variance = torch.var(returns, unbiased=False)
    if float(variance) == 0:
        return 0.0
    return float(1.0 - torch.var(returns - predictions, unbiased=False) / variance)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ControlInputError(
                "CUDA is unavailable; use --device cpu or --device auto"
            )
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device: {device}")


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_policy(
    model: nn.Module,
    config: PPOConfig,
    seeds: Sequence[int],
    *,
    environment_factory: EnvironmentFactory = DodgeEnv,
    temporary_root: Path | None = None,
) -> EvaluationResult:
    if not seeds:
        raise ValueError("evaluation requires at least one seed")
    if config.backend == "native":
        return _evaluate_native_policy(model, config, seeds)
    device = next(model.parameters()).device
    survival_frames: list[int] = []
    terminated: list[bool] = []
    model.eval()
    for seed in seeds:
        environment = _make_environment(
            environment_factory,
            step_frames=config.step_frames,
            temporary_root=temporary_root,
        )
        try:
            observation = environment.reset(seed=seed)
            accumulated_survival = 0.0
            episode_terminated = False
            for _ in range(config.max_episode_steps):
                board = board_tensor(observation.raw_state, device)
                with torch.inference_mode():
                    logits, _ = model(board.unsqueeze(0))
                action = action_from_index(int(logits.argmax(dim=1).item()))
                transition = environment.step(action)
                accumulated_survival += transition.reward
                observation = transition.observation
                if transition.done:
                    episode_terminated = True
                    accumulated_survival = (
                        transition.result.survival_frames
                        if transition.result is not None
                        else accumulated_survival
                    )
                    break
            survival_frames.append(int(round(accumulated_survival)))
            terminated.append(episode_terminated)
        finally:
            environment.close()
    return EvaluationResult(tuple(seeds), tuple(survival_frames), tuple(terminated))


def _evaluate_native_policy(
    model: nn.Module,
    config: PPOConfig,
    seeds: Sequence[int],
) -> EvaluationResult:
    device = next(model.parameters()).device
    environment = NativeBatchEnvironment(
        step_frames=config.step_frames,
        execution=config.native_execution,
        full_state=False,
        pixels=config.observation_mode == "pixels",
        board=config.observation_mode == "board",
    )
    try:
        result = environment.reset_batch(np.asarray(seeds, dtype=np.uint32))
        if config.observation_mode == "pixels":
            pixels = _initial_pixel_stack(_native_pixels(result), config.pixel_stack)
        else:
            boards = np.array(_native_board(result), dtype=np.float32, copy=True)
        active = [True] * len(seeds)
        survival_frames = [0.0] * len(seeds)
        terminated = [False] * len(seeds)
        model.eval()
        for _ in range(config.max_episode_steps):
            if not any(active):
                break
            if config.observation_mode == "pixels":
                observation_array = pixels
            else:
                observation_array = boards
            observation_tensor = torch.from_numpy(observation_array).to(device)
            with torch.inference_mode():
                logits, _ = model(observation_tensor)
            action_indices = (
                logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8, copy=True)
            )
            for lane, is_active in enumerate(active):
                if not is_active:
                    action_indices[lane] = 0
            result = environment.step_batch(action_indices)
            if config.observation_mode == "pixels":
                next_observations = _advance_pixel_stack(pixels, _native_pixels(result))
            else:
                next_observations = np.array(
                    _native_board(result), dtype=np.float32, copy=True
                )
            done_lanes = [lane for lane, done in enumerate(result.done) if bool(done)]
            completed: list[int] = []
            for lane, is_active in enumerate(active):
                if not is_active:
                    continue
                survival_frames[lane] += float(result.rewards[lane])
                if bool(result.done[lane]):
                    active[lane] = False
                    terminated[lane] = True
                    completed.append(lane)
            if config.observation_mode == "pixels":
                pixels = next_observations
            else:
                boards = next_observations
            # The batch API requires every lane to be live on the next call.
            # A lane that has already finished the measured episode still
            # receives masked actions until the other lanes finish, so reset
            # every completed native lane, not only newly measured lanes.
            if done_lanes:
                reset = environment.reset_lanes(
                    np.asarray(done_lanes, dtype=np.uint32),
                    np.zeros(len(done_lanes), dtype=np.uint32),
                )
                if config.observation_mode == "pixels":
                    reset_pixels = _native_pixels(reset)
                    for position, lane_value in enumerate(reset.lane_ids.tolist()):
                        pixels[int(lane_value)] = reset_pixels[position]
                else:
                    reset_boards = _native_board(reset)
                    for position, lane_value in enumerate(reset.lane_ids.tolist()):
                        boards[int(lane_value)] = reset_boards[position]
        return EvaluationResult(
            tuple(seeds),
            tuple(int(round(value)) for value in survival_frames),
            tuple(terminated),
        )
    finally:
        environment.close()


def train_ppo(
    config: PPOConfig,
    run_directory: Path,
    *,
    resume: bool = False,
    environment_factory: EnvironmentFactory = DodgeEnv,
    validation_seeds: Sequence[int] = DEVELOPMENT_VALIDATION_SEEDS,
    evaluation_seeds: Sequence[int] = EVALUATION_SEEDS,
    training_evaluation_seeds: Sequence[int] | None = None,
    initial_actor_state: Mapping[str, Tensor] | None = None,
    initialization: Mapping[str, object] | None = None,
) -> dict[str, object]:
    config.validate()
    if resume and (initial_actor_state is not None or initialization is not None):
        raise ControlInputError("PPO warm start cannot be combined with --resume")
    if resume:
        checkpoint = run_directory / "checkpoint-latest.pt"
        if not checkpoint.is_file():
            raise ControlInputError(f"PPO checkpoint does not exist: {checkpoint}")
    else:
        checkpoint = None
        if run_directory.exists() and any(run_directory.iterdir()):
            raise ControlInputError(
                f"PPO run directory is not empty; use --resume: {run_directory}"
            )
    run_directory.mkdir(parents=True, exist_ok=True)
    runtime_directory = run_directory / PPO_RUNTIME_DIRECTORY_NAME
    _prepare_runtime_directory(runtime_directory)
    if config.backend == "native":
        trainer = NativePPOTrainer(
            config,
            checkpoint=checkpoint,
            runtime_directory=runtime_directory,
            initial_actor_state=initial_actor_state,
            initialization=initialization,
        )
    else:
        trainer = PPOTrainer(
            config,
            environment_factory=environment_factory,
            checkpoint=checkpoint,
            runtime_directory=runtime_directory,
            initial_actor_state=initial_actor_state,
            initialization=initialization,
        )
    metrics_path = run_directory / "metrics.jsonl"
    try:
        _write_run_record(run_directory, config, trainer)
        while trainer.updates_completed < config.updates:
            batch, episodes = trainer.collect_rollout()
            update_metrics = trainer.update(batch)
            trainer.updates_completed += 1
            metrics: dict[str, object] = {
                "update": trainer.updates_completed,
                "global_step": trainer.global_step,
                "episodes_completed": trainer.episodes_completed,
                "environment_errors": trainer.environment_errors,
                "rollout_reward": float(batch.rewards.sum()),
                "rollout_neutral_fraction": float(
                    batch.actions.eq(ACTION_CHOICES.index("neutral")).float().mean()
                ),
                "episodes": [episode.to_json() for episode in episodes],
                **update_metrics,
            }
            trainer.save_checkpoint(run_directory / "checkpoint-latest.pt")
            if trainer.updates_completed % config.checkpoint_every == 0:
                trainer.save_checkpoint(
                    run_directory / f"checkpoint-{trainer.updates_completed:06d}.pt"
                )
            if config.eval_every and trainer.updates_completed % config.eval_every == 0:
                trainer.pause_environment()
                validation = evaluate_policy(
                    trainer.model,
                    config,
                    validation_seeds,
                    environment_factory=environment_factory,
                    temporary_root=runtime_directory,
                )
                metrics["validation"] = validation.to_json()
                if training_evaluation_seeds is not None:
                    training_evaluation = evaluate_policy(
                        trainer.model,
                        config,
                        training_evaluation_seeds,
                        environment_factory=environment_factory,
                        temporary_root=runtime_directory,
                    )
                    metrics["training_evaluation"] = training_evaluation.to_json()
                if (
                    trainer.best_validation is None
                    or validation.mean_survival_frames
                    > float(trainer.best_validation["mean_survival_frames"])
                ):
                    trainer.best_validation = validation.to_json()
                    trainer.save_checkpoint(run_directory / "checkpoint-best.pt")
            _append_jsonl(metrics_path, metrics)
            _write_run_record(run_directory, config, trainer)
            _print_update(metrics, trainer.device.type)
        trainer.pause_environment()
        final_validation = evaluate_policy(
            trainer.model,
            config,
            validation_seeds,
            environment_factory=environment_factory,
            temporary_root=runtime_directory,
        )
        final_training_evaluation = None
        if training_evaluation_seeds is not None:
            final_training_evaluation = evaluate_policy(
                trainer.model,
                config,
                training_evaluation_seeds,
                environment_factory=environment_factory,
                temporary_root=runtime_directory,
            )
        final_evaluation = evaluate_policy(
            trainer.model,
            config,
            evaluation_seeds,
            environment_factory=environment_factory,
            temporary_root=runtime_directory,
        )
        trainer.save_checkpoint(run_directory / "checkpoint-latest.pt")
        record = _write_run_record(
            run_directory,
            config,
            trainer,
            final_validation=final_validation,
            final_training_evaluation=final_training_evaluation,
            final_evaluation=final_evaluation,
        )
        return record
    finally:
        trainer.close()


def _write_run_record(
    run_directory: Path,
    config: PPOConfig,
    trainer: PPOTrainer,
    *,
    final_validation: EvaluationResult | None = None,
    final_training_evaluation: EvaluationResult | None = None,
    final_evaluation: EvaluationResult | None = None,
) -> dict[str, object]:
    observation_shape = _observation_shape_for_config(config)
    pixels_mode = config.observation_mode == "pixels"
    record: dict[str, object] = {
        "version": PPO_RUN_VERSION,
        "kind": "dodge_ppo_run",
        "model_type": _model_type_for_config(config),
        "observation_mode": config.observation_mode,
        "observation_shape": list(observation_shape),
        "board_shape": list(BOARD_SHAPE) if not pixels_mode else None,
        "pixel_shape": list(observation_shape) if pixels_mode else None,
        "actions": list(ACTION_CHOICES),
        "config": config.to_json(),
        "updates_completed": trainer.updates_completed,
        "global_step": trainer.global_step,
        "episodes_completed": trainer.episodes_completed,
        "environment_errors": trainer.environment_errors,
        "latest_checkpoint": "checkpoint-latest.pt",
        "runtime_directory": PPO_RUNTIME_DIRECTORY_NAME,
        "best_validation": trainer.best_validation,
        "initialization": trainer.initialization,
    }
    if (
        trainer.best_validation is not None
        and (run_directory / "checkpoint-best.pt").is_file()
    ):
        record["best_checkpoint"] = "checkpoint-best.pt"
    if final_validation is not None:
        record["final_validation"] = final_validation.to_json()
    if final_training_evaluation is not None:
        record["final_training_evaluation"] = final_training_evaluation.to_json()
    if final_evaluation is not None:
        record["final_evaluation"] = final_evaluation.to_json()
    _atomic_write_json(run_directory / "run.json", record)
    return record


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    try:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
    except OSError as error:
        raise ControlRuntimeError(
            f"could not append PPO metrics {path}: {error}"
        ) from error


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary: Path | None = None
    try:
        temporary = _temporary_path(path)
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    except OSError as error:
        raise ControlRuntimeError(
            f"could not write PPO run record {path}: {error}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_torch_save(value: object, path: Path) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = _temporary_path(path)
        torch.save(value, temporary)
        temporary.replace(path)
    except (OSError, RuntimeError) as error:
        raise ControlRuntimeError(
            f"could not save PPO checkpoint {path}: {error}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _temporary_path(path: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def _prepare_runtime_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK | os.X_OK):
            raise ControlInputError(f"PPO runtime directory is not writable: {path}")
        for child in path.iterdir():
            if child.is_dir() and child.name.startswith("dodge-neat-"):
                shutil.rmtree(child)
        usage = shutil.disk_usage(path)
    except ControlInputError:
        raise
    except OSError as error:
        raise ControlRuntimeError(
            f"could not prepare PPO runtime directory {path}: {error}"
        ) from error
    if usage.free < MIN_RUNTIME_FREE_BYTES:
        raise ControlInputError(
            f"PPO runtime directory {path} has {usage.free} bytes of free space; "
            f"need at least {MIN_RUNTIME_FREE_BYTES}"
        )


def _make_environment(
    factory: EnvironmentFactory,
    *,
    step_frames: int,
    temporary_root: Path | None,
) -> Environment:
    arguments: dict[str, object] = {"step_frames": step_frames}
    if temporary_root is not None:
        arguments["temporary_root"] = temporary_root
    return factory(**arguments)


def _print_update(metrics: Mapping[str, object], device: str) -> None:
    validation = metrics.get("validation")
    validation_text = ""
    if isinstance(validation, Mapping):
        validation_text = (
            f" validation_mean={float(validation['mean_survival_frames']):.1f}"
        )
    print(
        f"update={metrics['update']} steps={metrics['global_step']} "
        f"reward={float(metrics['rollout_reward']):.2f} "
        f"neutral={float(metrics['rollout_neutral_fraction']):.3f} "
        f"policy={float(metrics['policy_loss']):.4f} "
        f"value={float(metrics['value_loss']):.4f} "
        f"entropy={float(metrics['entropy']):.4f} device={device}{validation_text}",
        flush=True,
    )


def _new_run_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%S.%fZ")
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}-{suffix}"
        suffix += 1
    return candidate


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ppo-train")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--updates", type=_positive_int, default=1_000)
    parser.add_argument("--rollout-steps", type=_positive_int, default=256)
    parser.add_argument("--update-epochs", type=_positive_int, default=4)
    parser.add_argument("--minibatch-size", type=_positive_int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--neutral-bonus", type=float, default=0.02)
    parser.add_argument("--stability-bonus-cap", type=float, default=1.0)
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=_positive_int, default=2_000)
    parser.add_argument("--environment-restarts", type=_nonnegative_int, default=3)
    parser.add_argument("--checkpoint-every", type=_positive_int, default=10)
    parser.add_argument("--eval-every", type=_nonnegative_int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--backend", choices=("python", "native"), default="python")
    parser.add_argument("--native-lanes", type=_positive_int, default=32)
    parser.add_argument(
        "--native-execution", choices=("serial", "parallel"), default="parallel"
    )
    parser.add_argument(
        "--observation-mode", choices=("board", "pixels"), default="board"
    )
    parser.add_argument("--pixel-stack", type=_positive_int, default=4)
    parser.add_argument(
        "--pixel-architecture", choices=("fast", "small", "current"), default="small"
    )
    arguments = parser.parse_args(argv)
    run_directory = arguments.run_dir or _new_run_directory(DEFAULT_RUN_ROOT)
    config = PPOConfig(
        updates=arguments.updates,
        rollout_steps=arguments.rollout_steps,
        update_epochs=arguments.update_epochs,
        minibatch_size=arguments.minibatch_size,
        learning_rate=arguments.learning_rate,
        gamma=arguments.gamma,
        gae_lambda=arguments.gae_lambda,
        clip_coef=arguments.clip_coef,
        entropy_coef=arguments.entropy_coef,
        value_coef=arguments.value_coef,
        max_grad_norm=arguments.max_grad_norm,
        neutral_bonus=arguments.neutral_bonus,
        stability_bonus_cap=arguments.stability_bonus_cap,
        step_frames=arguments.step_frames,
        max_episode_steps=arguments.max_episode_steps,
        environment_restarts_per_rollout=arguments.environment_restarts,
        checkpoint_every=arguments.checkpoint_every,
        eval_every=arguments.eval_every,
        seed=arguments.seed,
        device=arguments.device,
        backend=arguments.backend,
        native_lanes=arguments.native_lanes,
        native_execution=arguments.native_execution,
        observation_mode=arguments.observation_mode,
        pixel_stack=arguments.pixel_stack,
        pixel_architecture=arguments.pixel_architecture,
    )
    try:
        record = train_ppo(config, run_directory, resume=arguments.resume)
    except (ControlInputError, ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-ppo-train: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"run": str(run_directory), **record}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
