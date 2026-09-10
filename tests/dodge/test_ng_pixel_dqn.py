from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import NativeBatchEnvironment
from dodge.ng.manifest import SeedManifest
from dodge.ng.pixel_dqn import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    DuelingPixelDQN,
    PixelDQNConfig,
    PixelNStepAccumulator,
    PixelReplayBuffer,
    _checkpoint_config_matches,
    _checkpoint_contract,
    _collect_macro_transition,
    _controller_for_config,
    _PixelProfiler,
    _restore_replay_snapshot,
    _save_checkpoint,
    _step_pixel_batch,
    _train_pixel_dqn_impl,
    _validate_pixel_result,
)

pytest.importorskip("dodge_native")


def _stack(value: int, stack_size: int = 2) -> np.ndarray:
    return np.full(
        (1, stack_size, FRAME_HEIGHT, FRAME_WIDTH),
        value,
        dtype=np.uint8,
    )


@pytest.mark.parametrize("architecture", ["fast", "palette-spatial"])
def test_pixel_dqn_accepts_only_indexed_pixel_tensors(architecture) -> None:
    model = DuelingPixelDQN(
        stack_size=2,
        hidden_size=16,
        architecture=architecture,
    )
    observations = torch.zeros((3, 2, FRAME_HEIGHT, FRAME_WIDTH), dtype=torch.uint8)

    values = model(observations)

    assert values.shape == (3, len(ACTION_CHOICES))
    assert torch.isfinite(values).all()
    with pytest.raises(ValueError, match="uint8"):
        model(observations.float())
    with pytest.raises(ValueError, match="pixel observations must have shape"):
        model(torch.zeros((3, 1, FRAME_HEIGHT, FRAME_WIDTH), dtype=torch.uint8))


def test_pixel_profiler_writes_bounded_cpu_trace(tmp_path) -> None:
    profiler = _PixelProfiler(tmp_path / "profile", torch.device("cpu"), 1)
    profiler.start()
    torch.zeros((4, 4)).sum()
    profiler.step()

    metadata = json.loads(
        (tmp_path / "profile" / "profile.json").read_text(encoding="utf-8")
    )
    assert metadata["captured_steps"] == 1
    assert (tmp_path / "profile" / "trace.json").is_file()
    assert (tmp_path / "profile" / "summary.txt").is_file()


def test_pixel_replay_frame_ring_and_n_step_boundary(tmp_path) -> None:
    replay = PixelReplayBuffer(
        capacity=8,
        stack_size=2,
        lane_count=1,
        run_directory=tmp_path,
    )
    accumulator = PixelNStepAccumulator(1, 3, 0.9, replay)
    references = [replay.store_stacks(_stack(value))[0] for value in (1, 2, 3)]

    accumulator.append(0, references[0], 0, (1, 1), 1.0, references[1], False, False)
    accumulator.append(0, references[1], 1, (1, 2), 2.0, references[2], True, False)

    assert replay.size == 2
    assert replay.rewards[0] == pytest.approx(1.0 + 0.9 * 2.0)
    assert replay.rewards[1] == pytest.approx(2.0)
    assert np.all(replay.discounts[:2] == 0.0)
    assert replay.terminated[:2].tolist() == [True, True]
    sample = replay.sample(2, np.random.default_rng(7))
    assert sample.observations.shape == (2, 2, FRAME_HEIGHT, FRAME_WIDTH)
    assert sample.observations.dtype == np.uint8
    replay.close()


def test_checkpoint_restores_saved_pixels_after_live_ring_is_overwritten(
    tmp_path,
) -> None:
    replay = PixelReplayBuffer(8, 2, 1, tmp_path)
    refs = replay.store_stacks(_stack(3))[0]
    replay.add(refs, 0, (0, 0), 1.0, refs, 0.0, True, False, 1)
    path = tmp_path / "checkpoint-latest.pt"
    _save_checkpoint(path, {"training_state": {"replay": replay.state_dict()}})
    original = replay.frames.copy()
    for _ in range(40):
        replay.store_stacks(_stack(9))
    replay.close()
    assert not np.array_equal(replay.frames, original)

    assert _restore_replay_snapshot(path, replay.frame_path)
    restored = PixelReplayBuffer(8, 2, 1, tmp_path, resume=True)
    restored.load_state_dict(
        torch.load(path, weights_only=False)["training_state"]["replay"]
    )
    np.testing.assert_array_equal(restored.frames, original)
    assert np.all(restored.sample(1, np.random.default_rng(7)).observations == 3)
    restored.close()


def test_failed_checkpoint_commit_preserves_previous_snapshot(
    tmp_path, monkeypatch
) -> None:
    replay = PixelReplayBuffer(8, 2, 1, tmp_path)
    replay.store_stacks(_stack(3))
    path = tmp_path / "checkpoint-latest.pt"
    _save_checkpoint(path, {"training_state": {"replay": replay.state_dict()}})
    old_checkpoint = path.read_bytes()
    old_snapshots = set(tmp_path.glob("replay-*.u8.gz"))

    def fail_save(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        _save_checkpoint(path, {"training_state": {"replay": replay.state_dict()}})
    assert path.read_bytes() == old_checkpoint
    assert set(tmp_path.glob("replay-*.u8.gz")) == old_snapshots
    replay.close()


def test_v94_corrupt_snapshot_cannot_replace_live_replay(tmp_path):
    replay = PixelReplayBuffer(8, 2, 1, tmp_path)
    replay.store_stacks(_stack(3))
    path = tmp_path / "checkpoint-latest.pt"
    _save_checkpoint(path, {"training_state": {"replay": replay.state_dict()}})
    payload = torch.load(path, weights_only=False)
    payload["replay_snapshot"]["raw_sha256"] = "0" * 64
    torch.save(payload, path)
    replay.store_stacks(_stack(9))
    replay.close()
    original = replay.frame_path.read_bytes()
    with pytest.raises(ValueError, match="hash or size"):
        _restore_replay_snapshot(path, replay.frame_path)
    assert replay.frame_path.read_bytes() == original


@pytest.mark.parametrize("execution", ["serial", "parallel"])
def test_active_pixel_lanes_match_independent_games(execution) -> None:
    options = dict(pixels=True, board=False, full_state=False, execution=execution)
    with (
        NativeBatchEnvironment(**options) as batch,
        NativeBatchEnvironment(**options) as single,
    ):
        batch.reset_batch_with_startup([30101, 30102])
        single.reset_batch_with_startup([30101])
        for step in range(120):
            # Lane 1 is frozen throughout. Lane 0 must be identical to one game.
            actual = batch.step_pixels_active([step % 9, 0], [True, False])
            expected = single.step_pixels([step % 9])
            assert actual.lane_ids.tolist() == [0]
            np.testing.assert_array_equal(actual.pixels, expected.pixels)
            np.testing.assert_array_equal(actual.frames, expected.frames)
            np.testing.assert_array_equal(actual.done, expected.done)
            if actual.done[0]:
                # Excluding a terminal lane must be permitted and leave it unchanged.
                assert batch.step_pixels_active([0, 0], [False, False]).lane_count == 0
                batch.reset_lanes_with_startup([0], [30101])
                single.reset_batch_with_startup([30101])
        untouched = batch.step_pixels_active([0, 0], [False, True])
        single.reset_batch_with_startup([30102])
        expected = single.step_pixels([0])
        np.testing.assert_array_equal(untouched.pixels, expected.pixels)


def test_pixel_checkpoint_contract_is_pixel_only_and_lives_are_single_ablation() -> (
    None
):
    one_life = PixelDQNConfig()
    three_lives = replace(one_life, training_lives=3)

    assert _checkpoint_contract(one_life)["observation_source"] == (
        "native_indexed_pixels_only"
    )
    assert one_life.to_json() | {"training_lives": 3} == three_lives.to_json()
    left = one_life.to_json()
    right = three_lives.to_json()
    left.pop("training_lives")
    right.pop("training_lives")
    assert left == right


def test_pixel_boundary_is_opt_in_and_old_checkpoints_remain_resumable() -> None:
    legacy = PixelDQNConfig().to_json()
    legacy.pop("native_pixel_boundary")
    assert _checkpoint_config_matches(legacy, PixelDQNConfig())
    assert not _checkpoint_config_matches(
        legacy,
        replace(PixelDQNConfig(), native_pixel_boundary="fast"),
    )

    class StubEnvironment:
        def step_batch(self, actions: np.ndarray) -> str:
            del actions
            return "legacy"

        def step_pixels(self, actions: np.ndarray) -> str:
            del actions
            return "fast"

    actions = np.zeros(1, dtype=np.uint8)
    environment = StubEnvironment()
    assert _step_pixel_batch(environment, actions, PixelDQNConfig()) == "legacy"
    assert (
        _step_pixel_batch(
            environment,
            actions,
            replace(PixelDQNConfig(), native_pixel_boundary="fast"),
        )
        == "fast"
    )


def test_v104_checkpoint_is_published_before_evaluation(tmp_path, monkeypatch) -> None:
    config = replace(
        PixelDQNConfig(),
        total_steps=1,
        batch_size=1,
        replay_capacity=8,
        warmup_steps=2,
        native_lanes=1,
        eval_every=1,
        checkpoint_every=99,
        pixel_stack=1,
        torch_threads=1,
    )
    manifest = SeedManifest.fresh_default(seed_start=30_200, seed_count=10)
    events: list[tuple[str, str | int]] = []

    class Telemetry:
        def __init__(self, _run_directory):
            pass

        def consume_control(self):
            return None

        def publish(self, _value):
            pass

        def close(self):
            pass

    class Environment:
        def close(self):
            pass

    def reset(_environment, seeds, reset_config):
        del seeds
        return (
            np.zeros(
                (1, reset_config.pixel_stack, FRAME_HEIGHT, FRAME_WIDTH),
                dtype=np.uint8,
            ),
            np.full((1, 2), 64, dtype=np.float32),
        )

    def collect(*args):
        events.append(("collect", args[-1]))
        return (
            args[1],
            args[3],
            args[2],
            args[11],
            1,
            {"game_frames_collected": 1.0},
        )

    def save(path, _payload):
        events.append(("save", path.name))

    def evaluate(_model, _seeds, _config):
        events.append(("evaluate", "inner"))
        return {"summary": {"mean_survival_frames": 1.0}}

    monkeypatch.setattr("dodge.ng.pixel_dqn.DashboardTelemetry", Telemetry)
    monkeypatch.setattr(
        "dodge.ng.pixel_dqn._new_pixel_environment", lambda _: Environment()
    )
    monkeypatch.setattr("dodge.ng.pixel_dqn._reset_pixels", reset)
    monkeypatch.setattr("dodge.ng.pixel_dqn._collect_macro_transition", collect)
    monkeypatch.setattr("dodge.ng.pixel_dqn._save_checkpoint", save)
    monkeypatch.setattr("dodge.ng.pixel_dqn.evaluate_pixel_dqn", evaluate)

    _train_pixel_dqn_impl(
        config,
        tmp_path,
        manifest,
        resume=False,
        evaluate_holdout=False,
        evaluate_training=False,
        stop_requested=[False],
    )

    first_evaluation = next(
        index for index, event in enumerate(events) if event[0] == "evaluate"
    )
    assert events.index(("save", "checkpoint-latest.pt")) < first_evaluation
    metric = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    for field in (
        "step_wall_seconds",
        "phase_overhead_seconds",
        "replay_sample_seconds",
        "host_to_device_seconds",
        "learner_compute_seconds",
        "evaluation_seconds",
        "checkpoint_seconds",
    ):
        assert field in metric


def test_native_pixel_result_has_exact_raster_and_no_snapshot_payload() -> None:
    with NativeBatchEnvironment(
        step_frames=4,
        execution="serial",
        full_state=False,
        pixels=True,
        board=False,
        ml=True,
        ml_grid_spacing=32,
    ) as environment:
        result = environment.reset_batch([30_200, 30_201])
        pixels, positions = _validate_pixel_result(result)

    assert pixels.shape == (2, FRAME_HEIGHT, FRAME_WIDTH)
    assert pixels.dtype == np.uint8
    assert int(pixels.max()) <= 15
    assert positions.shape == (2, 2)
    assert result.snapshot_bytes == (None, None)


@pytest.mark.parametrize("lives", [1, 3])
def test_frozen_lanes_preserve_survivor_cadence_and_terminal_replay(
    tmp_path, monkeypatch, lives
) -> None:
    config = replace(
        PixelDQNConfig(),
        native_lanes=2,
        pixel_stack=2,
        n_step=1,
        training_lives=lives,
        hold_decisions=8,
    )
    replay = PixelReplayBuffer(32, 2, 2, tmp_path)
    stacks = np.zeros((2, 2, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
    references = replay.store_stacks(stacks)
    stepped = np.zeros(2, dtype=np.int64)
    reset_at = []

    class Environment:
        def step_pixels_active(self, actions, active):
            del actions
            lanes = np.flatnonzero(active)
            stepped[lanes] += 1
            return SimpleNamespace(
                lane_count=len(lanes),
                lane_ids=lanes,
                pixels=np.full((len(lanes), 128, 128), 3, dtype=np.uint8),
                player_positions=np.full((len(lanes), 2), 64, dtype=np.float32),
                done=lanes == 0,
                rewards=np.full(len(lanes), 4, dtype=np.float32),
                frames_advanced=np.full(len(lanes), 4, dtype=np.uint32),
            )

    def reset(_environment, lanes, seeds, _config):
        del seeds
        reset_at.append(stepped.copy())
        assert lanes.tolist() == [0]
        return np.zeros((1, 2, 128, 128), dtype=np.uint8), np.full(
            (1, 2), 64, dtype=np.float32
        )

    monkeypatch.setattr("dodge.ng.pixel_dqn._reset_pixel_lanes", reset)
    monkeypatch.setattr(
        "dodge.ng.pixel_dqn._choose_actions",
        lambda *args: (np.zeros(2, dtype=np.uint8), {}),
    )
    result = _collect_macro_transition(
        Environment(),
        stacks,
        references,
        np.full((2, 2), 64, dtype=np.float32),
        _controller_for_config(config),
        None,
        config,
        np.zeros(2, dtype=np.int64),
        np.asarray([30101, 30102]),
        np.full(2, lives),
        (30101, 30102),
        0,
        PixelNStepAccumulator(2, 1, 0.99, replay),
        replay,
        np.random.default_rng(42),
        torch.device("cpu"),
        0,
    )
    assert stepped.tolist() == [1, 8]
    assert reset_at[0].tolist() == [1, 8]
    assert replay.terminated[:2].tolist() == [True, False]
    assert replay.discounts[0] == 0
    assert replay.rewards[:2].tolist() == [-60, 32]
    assert result[-1]["game_frames_collected"] == 36
    replay.close()
