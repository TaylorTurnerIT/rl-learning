from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from dodge.control import (
    PROJECT_ROOT,
    ControlInputError,
    ControlRuntimeError,
    MovementCommand,
    parse_commands,
    parse_seed,
)
from dodge.headless import HeadlessResult, replay_commands

HISTORY_DIRECTORY = PROJECT_ROOT / "history" / "dodge"
HISTORY_VERSION = 1


def save_winner(
    commands: list[MovementCommand],
    *,
    seed: int,
    fitness: int,
    epochs: int,
    replay_result: HeadlessResult,
    directory: Path = HISTORY_DIRECTORY,
    created_at: datetime | None = None,
) -> Path:
    timestamp = created_at or datetime.now(UTC)
    record = {
        "version": HISTORY_VERSION,
        "seed": seed,
        "fitness": fitness,
        "epochs": epochs,
        "commands": [
            {"move": command.move, "duration_ms": command.duration_ms}
            for command in commands
        ],
        "replay_result": replay_result,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"winner-{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def load_winner(path: Path) -> tuple[list[MovementCommand], int]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlInputError(f"could not read history: {error}") from error

    if not isinstance(value, dict):
        raise ControlInputError("history must be a JSON object")
    if value.get("version") != HISTORY_VERSION:
        raise ControlInputError(f"history version must be {HISTORY_VERSION}")

    try:
        commands = parse_commands(value["commands"])
        seed = parse_seed(str(value["seed"]))
    except (KeyError, ControlInputError) as error:
        raise ControlInputError(f"invalid history: {error}") from error

    return commands, seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-replay",
        description="Replay a saved Dodge winner without live controls.",
    )
    parser.add_argument("history", type=Path, help="winner history JSON file")
    arguments = parser.parse_args(argv)

    try:
        commands, seed = load_winner(arguments.history)
        result = replay_commands(commands, seed=seed)
    except (ControlInputError, ControlRuntimeError) as error:
        print(f"dodge-replay: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
