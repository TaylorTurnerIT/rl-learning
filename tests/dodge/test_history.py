from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from dodge.control import ControlInputError, MovementCommand
from dodge.history import HISTORY_VERSION, load_winner, main, save_winner

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
