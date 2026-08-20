from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from dodge.control import ControlInputError, MovementCommand
from dodge.history import (
    HISTORY_VERSION,
    create_run,
    latest_run,
    load_epoch,
    load_winner,
    main,
    replay_latest_run_main,
    replay_run_main,
    save_epoch,
    save_winner,
)

COMMANDS = [
    MovementCommand("x", 50),
    MovementCommand("neutral", 750),
    MovementCommand("left", 100),
]
REPLAY_RESULT = {
    "score": 2,
    "frames": 120,
    "survival_frames": 90,
    "seed": 42,
    "started": True,
    "died": True,
}


def test_save_winner_writes_replayable_history(tmp_path) -> None:
    path = save_winner(
        COMMANDS,
        seed=42,
        fitness=90,
        epochs=50,
        replay_result=REPLAY_RESULT,
        directory=tmp_path,
        created_at=datetime(2026, 8, 19, 12, 0, tzinfo=UTC),
    )

    assert path.name == "winner-20260819T120000.000000Z.json"
    assert load_winner(path) == (COMMANDS, 42)
    assert json.loads(path.read_text()) == {
        "version": HISTORY_VERSION,
        "kind": "winner",
        "seed": 42,
        "fitness": 90,
        "epochs": 50,
        "commands": [
            {"move": "x", "duration_ms": 50},
            {"move": "neutral", "duration_ms": 750},
            {"move": "left", "duration_ms": 100},
        ],
        "replay_result": REPLAY_RESULT,
    }


def test_load_winner_rejects_invalid_commands(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"version": HISTORY_VERSION, "seed": 42, "commands": []})
    )

    with pytest.raises(ControlInputError, match="invalid history"):
        load_winner(path)


def test_history_cli_replays_saved_commands(monkeypatch, tmp_path, capsys) -> None:
    path = save_winner(
        COMMANDS,
        seed=42,
        fitness=90,
        epochs=50,
        replay_result=REPLAY_RESULT,
        directory=tmp_path,
    )
    monkeypatch.setattr(
        "dodge.history.replay_commands", lambda commands, **_: REPLAY_RESULT
    )

    assert main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == REPLAY_RESULT


def test_epoch_history_records_each_epoch_and_replays_in_order(
    monkeypatch, tmp_path, capsys
) -> None:
    run_directory = create_run(
        seed=42,
        population=2,
        mutation_chance=0.1,
        max_epochs=2,
        directory=tmp_path,
        created_at=datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
    )
    first = {**REPLAY_RESULT, "survival_frames": 10}
    second = {**REPLAY_RESULT, "survival_frames": 20}
    save_epoch(
        COMMANDS,
        epoch=1,
        seed=42,
        fitness=10,
        global_best_fitness=10,
        headless_result=first,
        directory=run_directory,
    )
    save_epoch(
        COMMANDS,
        epoch=2,
        seed=42,
        fitness=20,
        global_best_fitness=20,
        headless_result=second,
        directory=run_directory,
    )

    assert run_directory.name == "run-20260820T120000.000000Z"
    assert json.loads((run_directory / "run.json").read_text()) == {
        "version": HISTORY_VERSION,
        "kind": "run",
        "seed": 42,
        "population": 2,
        "mutation_chance": 0.1,
        "max_epochs": 2,
    }
    assert load_epoch(run_directory / "epoch-0001.json") == (1, COMMANDS, 42, first)
    replayed = iter([first, second])
    monkeypatch.setattr(
        "dodge.history.replay_commands", lambda commands, **_: next(replayed)
    )

    assert replay_run_main([str(run_directory)]) == 0
    output = capsys.readouterr().out.splitlines()
    assert output[:2] == ["replaying epoch 1 (1/2)", "replaying epoch 2 (2/2)"]
    assert json.loads(output[2]) == [first, second]


def test_replay_latest_run_selects_newest_run(monkeypatch, tmp_path, capsys) -> None:
    older = create_run(
        seed=42,
        population=2,
        mutation_chance=0.1,
        max_epochs=2,
        directory=tmp_path,
        created_at=datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
    )
    newer = create_run(
        seed=42,
        population=2,
        mutation_chance=0.1,
        max_epochs=2,
        directory=tmp_path,
        created_at=datetime(2026, 8, 20, 13, 0, tzinfo=UTC),
    )
    replayed: list[object] = []
    monkeypatch.setattr(
        "dodge.history.replay_run",
        lambda directory: replayed.append(directory) or [REPLAY_RESULT],
    )

    assert latest_run(tmp_path) == newer
    assert replay_latest_run_main(["--history-dir", str(tmp_path)]) == 0
    assert replayed == [newer]
    assert json.loads(capsys.readouterr().out) == [REPLAY_RESULT]

    older.joinpath("run.json").unlink()
    newer.joinpath("run.json").unlink()
    with pytest.raises(ControlInputError, match="no saved runs"):
        latest_run(tmp_path)
