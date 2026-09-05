"""PyTorch side of the isolated Mojo/Python waypoint-DQN experiment."""

from __future__ import annotations

from contextlib import suppress

import numpy as np
import torch
from torch import Tensor, nn

OBSERVATION_SIZE = 225
ACTION_COUNT = 9
HIDDEN_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 1e-4
# The current Python DQN constructs AdamW without a weight_decay argument.
# Keep the isolated learner behaviorally aligned for the speed comparison.
WEIGHT_DECAY = 0.0
DEFAULT_TORCH_THREADS = 2


class DuelingWaypointDQN(nn.Module):
    """The same dueling MLP topology as the Python waypoint learner."""

    def __init__(
        self,
        input_size: int = OBSERVATION_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        action_count: int = ACTION_COUNT,
    ) -> None:
        super().__init__()
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
        features = self.features(observations)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class TorchWaypointLearner:
    """Persistent PyTorch learner called by Mojo at coarse batch boundaries."""

    def __init__(
        self,
        seed: int,
        lanes: int,
        threads: int = DEFAULT_TORCH_THREADS,
        hidden_size: int = HIDDEN_SIZE,
        validate_inputs: bool = True,
    ) -> None:
        if lanes < 1:
            raise ValueError("learner lane count must be positive")
        if threads < 1:
            raise ValueError("PyTorch thread count must be positive")
        torch.set_num_threads(threads)
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        torch.manual_seed(seed)
        self.lanes = lanes
        self.validate_inputs = validate_inputs
        self.online = DuelingWaypointDQN(hidden_size=hidden_size)
        self.target = DuelingWaypointDQN(hidden_size=hidden_size)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    @staticmethod
    def _observations(values: object, *, validate: bool) -> np.ndarray:
        observations = np.asarray(values, dtype=np.float32)
        if observations.ndim == 1:
            if observations.size % OBSERVATION_SIZE != 0:
                raise ValueError("flat observations have an invalid size")
            observations = observations.reshape(-1, OBSERVATION_SIZE)
        if observations.ndim != 2 or observations.shape[1] != OBSERVATION_SIZE:
            raise ValueError("observations have an invalid shape")
        if validate and not np.isfinite(observations).all():
            raise ValueError("observations must be finite")
        return observations

    def choose_actions(self, values: object, epsilon: float) -> np.ndarray:
        observations = self._observations(values, validate=self.validate_inputs)
        if len(observations) != self.lanes:
            raise ValueError("observation lane count does not match learner")
        with torch.inference_mode():
            q_values = self.online(torch.from_numpy(observations))
        actions = q_values.argmax(dim=1).numpy().astype(np.uint8, copy=False)
        random_mask = self.rng.random(len(actions)) < epsilon
        random_actions = self.rng.integers(
            0,
            ACTION_COUNT,
            size=len(actions),
            dtype=np.uint8,
        )
        actions[random_mask] = random_actions[random_mask]
        return np.ascontiguousarray(actions)

    def learn(
        self,
        observations: object,
        actions: object,
        rewards: object,
        next_observations: object,
        discounts: object,
    ) -> np.ndarray:
        current = self._observations(observations, validate=self.validate_inputs)
        following = self._observations(next_observations, validate=self.validate_inputs)
        actions_array = np.asarray(actions, dtype=np.int64).reshape(-1)
        rewards_array = np.asarray(rewards, dtype=np.float32).reshape(-1)
        discounts_array = np.asarray(discounts, dtype=np.float32).reshape(-1)
        batch_size = len(current)
        if (
            len(following) != batch_size
            or len(actions_array) != batch_size
            or len(rewards_array) != batch_size
            or len(discounts_array) != batch_size
        ):
            raise ValueError("learner batch fields have different lengths")
        if not batch_size >= 1:
            raise ValueError("learner batch must not be empty")
        if (
            not np.isfinite(rewards_array).all()
            or not np.isfinite(discounts_array).all()
        ):
            raise ValueError("learner targets must be finite")

        current_tensor = torch.from_numpy(current)
        next_tensor = torch.from_numpy(following)
        action_tensor = torch.from_numpy(actions_array)
        reward_tensor = torch.from_numpy(rewards_array)
        discount_tensor = torch.from_numpy(discounts_array)

        q_values = (
            self.online(current_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        )
        with torch.no_grad():
            next_actions = self.online(next_tensor).argmax(dim=1)
            next_values = (
                self.target(next_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            )
            targets = reward_tensor + discount_tensor * next_values
        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), max_norm=10.0
        )
        self.optimizer.step()
        self.updates += 1
        return np.asarray(
            [
                loss.item(),
                q_values.detach().mean().item(),
                targets.mean().item(),
                (q_values.detach() - targets).abs().mean().item(),
                float(gradient_norm),
            ],
            dtype=np.float32,
        )

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save_checkpoint(self, path: str) -> None:
        """Save the trained network for the separate Python evaluator."""
        checkpoint = {
            "kind": "dodge_mojo_hybrid_waypoint_dqn",
            "model_state_dict": self.online.state_dict(),
            "target_model_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "updates": self.updates,
        }
        torch.save(
            checkpoint,
            path,
        )
