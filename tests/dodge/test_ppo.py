from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from dodge.neat.environment import EpisodeResult, Observation, Transition
from dodge.neat.state import (
    EntityState,
    PlayerState,
    RawState,
    project_state,
)
from dodge.rl.ppo import (
    BOARD_SHAPE,
    TRAINING_SEEDS,
    DodgeActorCriticCNN,
    PPOConfig,
    PPOTrainer,
    StabilityReward,
    TrainingSeedStream,
    compute_gae,
    train_ppo,
)


def _observation(frame: int) -> Observation:
    raw_state = RawState(
        frame=frame,
        player=PlayerState(64, 64, 0, 0, 4),
        enemies=(EntityState(32, 32, 0, 1, 8, 8, "enemy", 1),),
        aoes=(),
    )
    return Observation(
        raw_state=raw_state,
        projected=project_state(raw_state),
    )


class FakePPOEnvironment:
    def __init__(self, **_: object) -> None:
        self.seed = 0
        self.steps = 0

    def reset(self, seed: int | None = None) -> Observation:
        self.seed = 0 if seed is None else seed
        self.steps = 0
        return _observation(0)

    def step(self, _: str) -> Transition:
        self.steps += 1
        observation = _observation(self.steps * 4)
        if self.steps < 3:
            return Transition(observation, 4.0, False, None)
        result = EpisodeResult(
            score=0,
            frames=self.steps * 4,
            survival_frames=self.steps * 4,
            seed=self.seed,
        )
        return Transition(observation, 4.0, True, result)

    def close(self) -> None:
        pass


def _config(**overrides: object) -> PPOConfig:
    values: dict[str, object] = {
        "updates": 1,
        "rollout_steps": 6,
        "update_epochs": 1,
        "minibatch_size": 3,
        "checkpoint_every": 1,
        "eval_every": 0,
        "device": "cpu",
    }
    values.update(overrides)
    return PPOConfig(**values)  # type: ignore[arg-type]


def test_actor_critic_returns_nine_logits_and_one_value() -> None:
    model = DodgeActorCriticCNN(hidden_size=16)

    logits, values = model(torch.zeros((2, *BOARD_SHAPE)))

    assert logits.shape == (2, 9)
    assert values.shape == (2,)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_stability_bonus_is_capped_below_survival_reward() -> None:
    reward = StabilityReward(neutral_bonus=0.2, cap=0.5)

    total_bonus = sum(reward.apply(0.0, "neutral") for _ in range(10))

    assert total_bonus == pytest.approx(0.5)
    assert reward.apply(1.0, "left") == 1.0
    assert total_bonus < 1.0


def test_v37_training_seed_stream_excludes_reserved_validation_seeds() -> None:
    stream = TrainingSeedStream(42)

    sampled = {stream.next() for _ in range(1_000)}

    assert len(TRAINING_SEEDS) == 29_991
    assert sampled.isdisjoint(set(range(29_991, 30_001)))


def test_v33_gae_resets_at_episode_boundaries_and_bootstraps_truncation() -> None:
    advantages, returns = compute_gae(
        torch.tensor([1.0, 1.0, 1.0]),
        torch.zeros(3),
        torch.tensor([5.0, 7.0, 11.0]),
        torch.tensor([False, True, False]),
        torch.tensor([False, True, False]),
        gamma=0.9,
        gae_lambda=0.5,
    )

    assert advantages.tolist() == pytest.approx([5.95, 1.0, 10.9])
    assert returns.tolist() == pytest.approx(advantages.tolist())


def test_ppo_trainer_collects_real_shaped_rollout_and_updates() -> None:
    trainer = PPOTrainer(_config(), environment_factory=FakePPOEnvironment)
    try:
        batch, episodes = trainer.collect_rollout()
        metrics = trainer.update(batch)
    finally:
        trainer.close()

    assert batch.observations.shape == (6, *BOARD_SHAPE)
    assert batch.actions.shape == (6,)
    assert len(episodes) == 2
    assert all(episode.terminated for episode in episodes)
    assert all(math.isfinite(value) for value in metrics.values())
    assert trainer.global_step == 6


def test_ppo_single_transition_update_has_finite_metrics() -> None:
    trainer = PPOTrainer(
        _config(rollout_steps=1, minibatch_size=1),
        environment_factory=FakePPOEnvironment,
    )
    try:
        batch, _ = trainer.collect_rollout()
        metrics = trainer.update(batch)
    finally:
        trainer.close()

    assert all(math.isfinite(value) for value in metrics.values())


def test_ppo_run_checkpoints_and_resumes(tmp_path: Path) -> None:
    run_directory = tmp_path / "ppo-run"
    first = train_ppo(
        _config(),
        run_directory,
        environment_factory=FakePPOEnvironment,
        validation_seeds=(1,),
        evaluation_seeds=(2,),
    )

    assert first["updates_completed"] == 1
    assert (run_directory / "checkpoint-latest.pt").is_file()
    assert len((run_directory / "metrics.jsonl").read_text().splitlines()) == 1
    assert json.loads((run_directory / "run.json").read_text())["model_type"] == (
        "DodgeActorCriticCNN"
    )

    resumed = train_ppo(
        _config(updates=2),
        run_directory,
        resume=True,
        environment_factory=FakePPOEnvironment,
        validation_seeds=(1,),
        evaluation_seeds=(2,),
    )

    assert resumed["updates_completed"] == 2
    assert len((run_directory / "metrics.jsonl").read_text().splitlines()) == 2
