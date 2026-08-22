from __future__ import annotations

import pytest

from dodge.control import ControlRuntimeError
from dodge.neat.environment import EpisodeResult, EpisodeTrace
from dodge.neat.replay import replay_episode, trace_commands


def _trace() -> EpisodeTrace:
    return EpisodeTrace(
        seed=42,
        step_frames=4,
        enemy_slots=16,
        aoe_slots=8,
        actions=("right", "up_left"),
        result=EpisodeResult(3, 34, 8, 42),
        max_visible_enemies=2,
        max_visible_aoes=1,
        enemy_overflow_frames=0,
        aoe_overflow_frames=0,
    )


def test_trace_commands_include_menu_x_and_exact_step_durations() -> None:
    assert [
        (command.move, command.duration_ms) for command in trace_commands(_trace())
    ] == [("x", 66), ("right", 66), ("up_left", 66)]


def test_replay_requires_the_recorded_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dodge.neat.replay.replay_commands",
        lambda _commands, seed: {
            "score": 3,
            "frames": 34,
            "survival_frames": 8,
            "seed": seed,
            "started": True,
            "died": True,
        },
    )
    assert replay_episode(_trace())["survival_frames"] == 8

    monkeypatch.setattr(
        "dodge.neat.replay.replay_commands",
        lambda _commands, seed: {
            "score": 3,
            "frames": 34,
            "survival_frames": 7,
            "seed": seed,
            "started": True,
            "died": True,
        },
    )
    with pytest.raises(ControlRuntimeError, match="diverged"):
        replay_episode(_trace())
