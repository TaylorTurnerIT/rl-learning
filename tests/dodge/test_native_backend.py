from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from dodge.control import PEMSA_PATH
from dodge.native.batch import NativeDodgeEnv
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
    with DodgeEnv(step_frames=4) as fallback, NativeDodgeEnv(
        step_frames=4
    ) as native:
        fallback_observation = fallback.reset(seed=42)
        native_observation = native.reset(seed=42)
        _assert_observation_match(
            fallback_observation, native_observation, frame_offset=13
        )
        for action in actions:
            fallback_transition = fallback.step(action)
            native_transition = native.step(action)
            assert native_transition.reward == pytest.approx(
                fallback_transition.reward
            )
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
