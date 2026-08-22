from __future__ import annotations

from dataclasses import replace

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


@pytest.mark.parametrize("step_frames", (3, 4, 5))
def test_v20_menu_start_is_always_three_frames(step_frames: int) -> None:
    trace = replace(_trace(), step_frames=step_frames)
    assert [
        (command.move, command.duration_ms) for command in trace_commands(trace)
    ] == [
        ("x", 50),
        ("right", (step_frames * 1_000) // 60),
        ("up_left", (step_frames * 1_000) // 60),
    ]


def test_replay_requires_the_recorded_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dodge.neat.replay.replay_commands",
        lambda _commands, seed, **_kwargs: {
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
        lambda _commands, seed, **_kwargs: {
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


def test_v20_legacy_episode_restores_its_mouse_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def replay(commands: object, **kwargs: object) -> dict[str, object]:
        captured["commands"] = commands
        captured.update(kwargs)
        return {
            "score": 3,
            "frames": 34,
            "survival_frames": 8,
            "seed": 42,
            "started": True,
            "died": True,
        }

    monkeypatch.setattr("dodge.neat.replay.replay_commands", replay)

    replay_episode(replace(_trace(), input_mode="legacy_mouse"))

    assert captured["wait_for_game_start"] is True
    assert captured["legacy_mouse_input"] is True
