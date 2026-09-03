from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dodge.control import PEMSA_PATH
from dodge.native.batch import NativeBatchEnvironment, NativeBatchResult, NativeDodgeEnv
from dodge.neat.environment import DodgeEnv
from dodge.rl.ppo import NativePPOTrainer, PPOConfig

PROJECT_ROOT = Path(__file__).parents[2]


def test_native_training_modules_have_no_forbidden_ipc_boundary() -> None:
    native_batch_source = (
        PROJECT_ROOT / "src" / "dodge" / "native" / "batch.py"
    ).read_text(encoding="utf-8")
    forbidden_tokens = ("subprocess", "xdotool", "Xvfb", "DISPLAY", "json")
    assert not any(token in native_batch_source for token in forbidden_tokens)

    tree = ast.parse(native_batch_source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "subprocess" not in imported_modules


def test_native_ppo_does_not_spawn_child_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("dodge_native")

    def forbidden_process(*_: object, **__: object) -> None:
        raise AssertionError("native PPO attempted to spawn a child process")

    monkeypatch.setattr(subprocess, "Popen", forbidden_process)
    monkeypatch.setattr(subprocess, "run", forbidden_process)
    trainer = NativePPOTrainer(
        PPOConfig(
            updates=1,
            rollout_steps=2,
            update_epochs=1,
            minibatch_size=1,
            checkpoint_every=1,
            eval_every=0,
            max_episode_steps=1,
            device="cpu",
            backend="native",
            native_lanes=2,
        )
    )
    try:
        batch, _ = trainer.collect_rollout()
    finally:
        trainer.close()
    assert len(batch.rewards) == 2


def test_native_repeated_full_observation_batches_are_byte_identical() -> None:
    pytest.importorskip("dodge_native")
    seeds = np.asarray([3, 17, 41, 89, 13, 27, 58, 101], dtype=np.uint32)
    actions = [
        np.asarray(
            [(decision * 7 + lane * 3) % 9 for lane in range(len(seeds))],
            dtype=np.uint8,
        )
        for decision in range(32)
    ]

    first = _collect_full_observation_batches(seeds, actions)
    second = _collect_full_observation_batches(seeds, actions)

    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        _assert_batch_result_equal(left, right)


def _collect_full_observation_batches(
    seeds: np.ndarray, actions: list[np.ndarray]
) -> list[NativeBatchResult]:
    results: list[NativeBatchResult] = []
    with NativeBatchEnvironment(
        step_frames=4,
        execution="parallel",
        full_state=True,
        pixels=True,
        board=True,
    ) as environment:
        result = environment.reset_batch(seeds)
        results.append(result)
        for decision, action_batch in enumerate(actions):
            result = environment.step_batch(action_batch)
            results.append(result)
            done_lanes = np.flatnonzero(result.done).astype(np.uint32)
            if done_lanes.size:
                replacement_seeds = np.asarray(
                    [
                        (13 + decision * 31 + int(lane) * 17) % 32_768
                        for lane in done_lanes
                    ],
                    dtype=np.uint32,
                )
                results.append(environment.reset_lanes(done_lanes, replacement_seeds))
    return results


def _assert_batch_result_equal(
    left: NativeBatchResult, right: NativeBatchResult
) -> None:
    fields = (
        "lane_ids",
        "frames",
        "frames_advanced",
        "rewards",
        "done",
        "seeds",
        "state_hashes",
        "pixel_hashes",
        "modes",
        "event_flags",
    )
    for field in fields:
        assert np.array_equal(getattr(left, field), getattr(right, field))
    for field in ("pixels", "board"):
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        assert (left_value is None) is (right_value is None)
        if left_value is not None and right_value is not None:
            assert np.array_equal(left_value, right_value)
    assert left.snapshot_bytes == right.snapshot_bytes


@pytest.mark.skipif(
    not PEMSA_PATH.is_file() or shutil.which("Xvfb") is None,
    reason="requires the checked-in Pemsa runtime and Xvfb",
)
def test_native_and_pemsa_short_trajectory_matches() -> None:
    pytest.importorskip("dodge_native")
    actions = (
        "right",
        "up_left",
        "neutral",
        "down",
        "left",
        "up_right",
        "neutral",
        "down_left",
    )
    with DodgeEnv(step_frames=4) as fallback, NativeDodgeEnv(step_frames=4) as native:
        fallback_observation = fallback.reset(seed=42)
        native_observation = native.reset(seed=42)
        _assert_observation_match(
            fallback_observation, native_observation, frame_offset=13
        )
        for action in actions:
            fallback_transition = fallback.step(action)
            native_transition = native.step(action)
            assert native_transition.reward == pytest.approx(fallback_transition.reward)
            assert native_transition.done is fallback_transition.done
            _assert_observation_match(
                fallback_transition.observation,
                native_transition.observation,
                frame_offset=13,
            )
            if fallback_transition.done:
                break


def _assert_observation_match(
    left: object, right: object, *, frame_offset: int
) -> None:
    left_state = left.raw_state  # type: ignore[attr-defined]
    right_state = right.raw_state  # type: ignore[attr-defined]
    assert right_state.frame + frame_offset == left_state.frame
    for left_value, right_value in zip(
        (
            left_state.player.x,
            left_state.player.y,
            left_state.player.vx,
            left_state.player.vy,
            left_state.player.size,
        ),
        (
            right_state.player.x,
            right_state.player.y,
            right_state.player.vx,
            right_state.player.vy,
            right_state.player.size,
        ),
        strict=True,
    ):
        assert right_value == pytest.approx(left_value, abs=0.0002)
    assert len(left_state.enemies) == len(right_state.enemies)
    assert len(left_state.aoes) == len(right_state.aoes)
    for left_entity, right_entity in zip(
        (*left_state.enemies, *left_state.aoes),
        (*right_state.enemies, *right_state.aoes),
        strict=True,
    ):
        for left_value, right_value in zip(
            (
                left_entity.x,
                left_entity.y,
                left_entity.vx,
                left_entity.vy,
                left_entity.width,
                left_entity.height,
                left_entity.stage,
            ),
            (
                right_entity.x,
                right_entity.y,
                right_entity.vx,
                right_entity.vy,
                right_entity.width,
                right_entity.height,
                right_entity.stage,
            ),
            strict=True,
        ):
            assert right_value == pytest.approx(left_value, abs=0.0002)
        assert right_entity.kind == left_entity.kind
