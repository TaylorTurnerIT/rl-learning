"""Focused tests for the isolated Mojo/PyTorch learner boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_waypoint_learner import TorchWaypointLearner


def _batch() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    observations = rng.random((4, 225), dtype=np.float32)
    next_observations = rng.random((4, 225), dtype=np.float32)
    actions = rng.integers(0, 9, size=4, dtype=np.uint8)
    rewards = rng.random(4, dtype=np.float32)
    discounts = np.full(4, 0.99, dtype=np.float32)
    return observations, actions, rewards, next_observations, discounts


def test_fast_path_preserves_learning_contract() -> None:
    learner = TorchWaypointLearner(7, lanes=4, threads=2, validate_inputs=False)
    observations, actions, rewards, next_observations, discounts = _batch()
    metrics = learner.learn(
        observations, actions, rewards, next_observations, discounts
    )
    assert metrics.shape == (5,)
    assert np.isfinite(metrics).all()
    assert learner.updates == 1


def test_checked_path_rejects_nonfinite_observations() -> None:
    learner = TorchWaypointLearner(7, lanes=4, threads=2)
    observations, actions, rewards, next_observations, discounts = _batch()
    observations[0, 0] = np.nan
    with pytest.raises(ValueError, match="observations must be finite"):
        learner.learn(observations, actions, rewards, next_observations, discounts)


def test_checkpoint_contains_model_and_optimizer_state(tmp_path: Path) -> None:
    learner = TorchWaypointLearner(7, lanes=4, threads=2)
    checkpoint = tmp_path / "checkpoint.pt"
    learner.save_checkpoint(str(checkpoint))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["kind"] == "dodge_mojo_hybrid_waypoint_dqn"
    assert payload["updates"] == 0
    assert payload["model_state_dict"]
    assert payload["optimizer_state_dict"]
