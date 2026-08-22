from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dodge.control import ControlInputError, ControlRuntimeError
from dodge.neat.environment import EpisodeResult, EpisodeTrace, save_episode_trace
from dodge.neat.replay import (
    generation_winner_trace,
    latest_run,
    replay_episode,
    replay_latest_run_main,
    trace_commands,
)


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


def test_v25_replay_latest_selects_newest_run_generation_winner_and_best_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _save_generation(
        tmp_path / "run-20260822T010000.000000Z",
        epoch=1,
        genome_id=3,
        survivals={11: 100},
    )
    newest = tmp_path / "run-20260822T020000.000000Z"
    _save_generation(
        newest,
        epoch=2,
        genome_id=7,
        survivals={21: 100, 22: 300, 23: 300},
    )
    captured: list[EpisodeTrace] = []
    monkeypatch.setattr(
        "dodge.neat.replay.replay_episode",
        lambda trace: (
            captured.append(trace)
            or {
                "score": trace.result.score,
                "frames": trace.result.frames,
                "survival_frames": trace.result.survival_frames,
                "seed": trace.seed,
                "started": True,
                "died": True,
            }
        ),
    )

    assert latest_run(tmp_path) == newest
    assert generation_winner_trace(newest, 2).seed == 22
    assert replay_latest_run_main(["2", "--history-dir", str(tmp_path)]) == 0
    assert captured[0].seed == 22
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["seed"] == 22


def test_v25_replay_latest_fails_before_replay_for_missing_history(
    tmp_path: Path,
) -> None:
    with pytest.raises(ControlInputError, match="no saved runs"):
        latest_run(tmp_path)

    run = tmp_path / "run-20260822T020000.000000Z"
    _save_generation(run, epoch=1, genome_id=7, survivals={21: 100})
    with pytest.raises(ControlInputError, match="contains no generation 2"):
        generation_winner_trace(run, 2)


def _save_generation(
    directory: Path, *, epoch: int, genome_id: int, survivals: dict[int, int]
) -> None:
    directory.mkdir(parents=True)
    directory.joinpath("run.json").write_text(
        json.dumps(
            {
                "version": 2,
                "kind": "neat_run",
                "generations": [
                    {
                        "generation": epoch,
                        "best_genome_id": genome_id,
                        "seed_bank": list(survivals),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for seed, survival in survivals.items():
        save_episode_trace(
            replace(
                _trace(),
                seed=seed,
                result=EpisodeResult(3, survival + 10, survival, seed),
            ),
            directory / f"generation-{epoch:04d}",
            filename=f"genome-{genome_id:04d}-seed-{seed}.json",
        )
