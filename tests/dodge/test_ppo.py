from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dodge.control import ControlInputError, ControlRuntimeError
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
    NativePPOTrainer,
    PPOConfig,
    PPOTrainer,
    StabilityReward,
    TrainingSeedStream,
    _atomic_torch_save,
    _prepare_runtime_directory,
    compute_gae,
    evaluate_policy,
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


def test_explicit_training_seed_stream_is_reproducible_and_scoped() -> None:
    candidates = (30_100, 30_101, 30_102)
    first = TrainingSeedStream(42, candidates)
    second = TrainingSeedStream(42, candidates)

    first_values = [first.next() for _ in range(100)]
    second_values = [second.next() for _ in range(100)]

    assert first_values == second_values
    assert set(first_values) <= set(candidates)


def test_explicit_training_seed_config_is_json_serializable() -> None:
    config = _config(
        training_seeds=(30_100, 30_101),
        training_seed_manifest="manifest-hash",
    )

    assert config.to_json()["training_seeds"] == [30_100, 30_101]
    assert config.to_json()["training_seed_manifest"] == "manifest-hash"


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


def test_native_ppo_trainer_collects_batched_board_rollout() -> None:
    pytest.importorskip("dodge_native")
    trainer = NativePPOTrainer(
        _config(
            backend="native",
            native_lanes=2,
            native_execution="serial",
            rollout_steps=6,
        )
    )
    try:
        batch, episodes = trainer.collect_rollout()
        metrics = trainer.update(batch)
    finally:
        trainer.close()

    assert batch.observations.shape == (6, *BOARD_SHAPE)
    assert batch.rewards.tolist() == [4.0] * 6
    assert not episodes
    assert trainer.global_step == 6
    assert all(math.isfinite(value) for value in metrics.values())


def test_v14_native_evaluation_resets_inactive_completed_lanes(monkeypatch) -> None:
    class FakeBatchEnvironment:
        thresholds = (1, 2, 3)

        def __init__(self, **_: object) -> None:
            self.steps = [0, 0, 0]
            self.done = [False, False, False]

        @staticmethod
        def _result(lanes: list[int], done: list[bool]):
            return SimpleNamespace(
                board=np.zeros((len(lanes), *BOARD_SHAPE), dtype=np.float32),
                done=np.asarray(done, dtype=bool),
                lane_ids=np.asarray(lanes, dtype=np.uint32),
                rewards=np.ones(len(lanes), dtype=np.float32),
            )

        def reset_batch(self, seeds: object):
            lanes = list(range(3))
            self.steps = [0, 0, 0]
            self.done = [False, False, False]
            return self._result(lanes, self.done)

        def step_batch(self, actions: object):
            assert not any(self.done)
            done = []
            for lane in range(3):
                self.steps[lane] += 1
                done.append(self.steps[lane] >= self.thresholds[lane])
            self.done = done
            return self._result(list(range(3)), done)

        def reset_lanes(self, lanes: object, seeds: object):
            values = [int(lane) for lane in lanes]
            for lane in values:
                self.steps[lane] = 0
                self.done[lane] = False
            return self._result(values, [False] * len(values))

        def close(self) -> None:
            pass

    class FixedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, boards: torch.Tensor):
            return (
                torch.zeros((boards.shape[0], 9), device=boards.device),
                torch.zeros(boards.shape[0], device=boards.device),
            )

    monkeypatch.setattr("dodge.rl.ppo.NativeBatchEnvironment", FakeBatchEnvironment)
    result = evaluate_policy(
        FixedModel(),
        _config(backend="native", max_episode_steps=10),
        seeds=(1, 2, 3),
    )

    assert result.survival_frames == (1, 2, 3)
    assert result.terminated == (True, True, True)


def test_native_ppo_run_records_backend_and_observation_mode(tmp_path: Path) -> None:
    pytest.importorskip("dodge_native")
    run_directory = tmp_path / "native-ppo-run"
    record = train_ppo(
        _config(
            backend="native",
            native_lanes=2,
            native_execution="parallel",
            rollout_steps=2,
            max_episode_steps=1,
        ),
        run_directory,
        validation_seeds=(1,),
        evaluation_seeds=(2,),
    )

    stored = json.loads((run_directory / "run.json").read_text())
    assert record["updates_completed"] == 1
    assert stored["config"]["backend"] == "native"
    assert stored["config"]["observation_mode"] == "board"


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


def test_ppo_run_records_training_side_evaluation(tmp_path: Path) -> None:
    run_directory = tmp_path / "ppo-run"
    train_ppo(
        _config(
            eval_every=1,
            training_seeds=(30_100, 30_101),
            training_seed_manifest="manifest-hash",
        ),
        run_directory,
        environment_factory=FakePPOEnvironment,
        validation_seeds=(30_100,),
        training_evaluation_seeds=(30_100, 30_101),
        evaluation_seeds=(30_102,),
    )

    metrics = json.loads(
        (run_directory / "metrics.jsonl").read_text().splitlines()[0]
    )
    record = json.loads((run_directory / "run.json").read_text())
    assert metrics["training_evaluation"]["seeds"] == [30_100, 30_101]
    assert record["final_training_evaluation"]["seeds"] == [30_100, 30_101]


def test_v39_ppo_run_passes_run_scoped_temporary_root(tmp_path: Path) -> None:
    run_directory = tmp_path / "ppo-run"
    temporary_roots: list[Path | None] = []

    def environment_factory(**kwargs: object) -> FakePPOEnvironment:
        temporary_root = kwargs.get("temporary_root")
        temporary_roots.append(temporary_root)  # type: ignore[arg-type]
        return FakePPOEnvironment(**kwargs)

    train_ppo(
        _config(),
        run_directory,
        environment_factory=environment_factory,
        validation_seeds=(1,),
        evaluation_seeds=(2,),
    )

    assert temporary_roots
    assert all(root == run_directory / ".runtime" for root in temporary_roots)


def test_v40_runtime_preflight_cleans_owned_stale_workspaces_only(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    stale = runtime_root / "dodge-neat-stale"
    stale.mkdir(parents=True)
    (stale / "cartridge.p8").write_text("stale")
    unrelated = runtime_root / "keep.txt"
    unrelated.write_text("keep")

    _prepare_runtime_directory(runtime_root)

    assert not stale.exists()
    assert unrelated.read_text() == "keep"


def test_v40_runtime_preflight_rejects_insufficient_free_space(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "dodge.rl.ppo.shutil.disk_usage",
        lambda _: SimpleNamespace(free=0),
    )

    with pytest.raises(ControlInputError, match="free space"):
        _prepare_runtime_directory(tmp_path / "runtime")


def test_v41_checkpoint_error_names_operation_and_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"

    def fail_save(*_: object, **__: object) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr("dodge.rl.ppo.torch.save", fail_save)

    with pytest.raises(ControlRuntimeError) as error:
        _atomic_torch_save({}, checkpoint)

    assert "checkpoint" in str(error.value)
    assert str(checkpoint) in str(error.value)
