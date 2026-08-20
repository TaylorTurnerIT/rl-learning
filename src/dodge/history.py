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


def create_run(
    *,
    seed: int,
    population: int,
    mutation_chance: float,
    max_epochs: int,
    directory: Path = HISTORY_DIRECTORY,
    created_at: datetime | None = None,
) -> Path:
    timestamp = created_at or datetime.now(UTC)
    run_directory = directory / f"run-{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}"
    run_directory.mkdir(parents=True, exist_ok=False)
    _write_json(
        run_directory / "run.json",
        {
            "version": HISTORY_VERSION,
            "kind": "run",
            "seed": seed,
            "population": population,
            "mutation_chance": mutation_chance,
            "max_epochs": max_epochs,
        },
    )
    return run_directory


def save_winner(
    commands: list[MovementCommand],
    *,
    seed: int,
    fitness: int,
    epochs: int,
    replay_result: HeadlessResult,
    directory: Path = HISTORY_DIRECTORY,
    created_at: datetime | None = None,
    filename: str | None = None,
) -> Path:
    timestamp = created_at or datetime.now(UTC)
    record = {
        "version": HISTORY_VERSION,
        "kind": "winner",
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
    name = filename or f"winner-{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}.json"
    path = directory / name
    _write_json(path, record)
    return path


def save_epoch(
    commands: list[MovementCommand],
    *,
    epoch: int,
    seed: int,
    fitness: int,
    global_best_fitness: int,
    headless_result: HeadlessResult,
    directory: Path,
) -> Path:
    if epoch < 1:
        raise ValueError("epoch must be positive")
    path = directory / f"epoch-{epoch:04d}.json"
    _write_json(
        path,
        {
            "version": HISTORY_VERSION,
            "kind": "epoch",
            "epoch": epoch,
            "seed": seed,
            "fitness": fitness,
            "global_best_fitness": global_best_fitness,
            "commands": _commands_json(commands),
            "headless_result": headless_result,
        },
    )
    return path


def load_winner(path: Path) -> tuple[list[MovementCommand], int]:
    value = _load_record(path)

    try:
        commands = parse_commands(value["commands"])
        seed = parse_seed(str(value["seed"]))
    except (KeyError, ControlInputError) as error:
        raise ControlInputError(f"invalid history: {error}") from error

    return commands, seed


def load_epoch(path: Path) -> tuple[int, list[MovementCommand], int, HeadlessResult]:
    value = _load_record(path)
    try:
        epoch = value["epoch"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            raise ControlInputError("epoch must be a positive integer")
        commands = parse_commands(value["commands"])
        seed = parse_seed(str(value["seed"]))
        result = value["headless_result"]
        if not isinstance(result, dict):
            raise ControlInputError("headless_result must be an object")
    except (KeyError, ControlInputError) as error:
        raise ControlInputError(f"invalid epoch history: {error}") from error
    return epoch, commands, seed, result


def replay_run(directory: Path) -> list[HeadlessResult]:
    paths = sorted(directory.glob("epoch-*.json"))
    if not paths:
        raise ControlInputError("run history contains no epoch records")

    results: list[HeadlessResult] = []
    for index, path in enumerate(paths, start=1):
        epoch, commands, seed, expected = load_epoch(path)
        print(f"replaying epoch {epoch} ({index}/{len(paths)})")
        result = replay_commands(commands, seed=seed)
        if result != expected:
            raise ControlRuntimeError(
                f"epoch {epoch} replay diverged: expected {expected}, got {result}"
            )
        results.append(result)
    return results


def latest_run(directory: Path = HISTORY_DIRECTORY) -> Path:
    runs = sorted(
        path
        for path in directory.glob("run-*")
        if path.is_dir() and (path / "run.json").is_file()
    )
    if not runs:
        raise ControlInputError("history contains no saved runs")
    return runs[-1]


def replay_latest_run(directory: Path = HISTORY_DIRECTORY) -> list[HeadlessResult]:
    return replay_run(latest_run(directory))


def replay_run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-replay-run",
        description="Replay every saved epoch winner in order.",
    )
    parser.add_argument("history", type=Path, help="run history directory")
    arguments = parser.parse_args(argv)

    try:
        results = replay_run(arguments.history)
    except (ControlInputError, ControlRuntimeError) as error:
        print(f"dodge-replay-run: {error}", file=sys.stderr)
        return 1

    print(json.dumps(results, separators=(",", ":")))
    return 0


def replay_latest_run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-replay-latest",
        description="Replay every epoch from the most recent saved Dodge run.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=HISTORY_DIRECTORY,
        help="directory containing run-* history directories",
    )
    arguments = parser.parse_args(argv)

    try:
        results = replay_latest_run(arguments.history_dir)
    except (ControlInputError, ControlRuntimeError) as error:
        print(f"dodge-replay-latest: {error}", file=sys.stderr)
        return 1

    print(json.dumps(results, separators=(",", ":")))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "replay-run":
        return replay_run_main(arguments[1:])
    if arguments and arguments[0] == "replay-latest":
        return replay_latest_run_main(arguments[1:])

    parser = argparse.ArgumentParser(
        prog="dodge-replay",
        description="Replay a saved Dodge winner without live controls.",
    )
    parser.add_argument("history", type=Path, help="winner history JSON file")
    arguments = parser.parse_args(arguments)

    try:
        commands, seed = load_winner(arguments.history)
        result = replay_commands(commands, seed=seed)
    except (ControlInputError, ControlRuntimeError) as error:
        print(f"dodge-replay: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, separators=(",", ":")))
    return 0


def _commands_json(commands: list[MovementCommand]) -> list[dict[str, str | int]]:
    return [
        {"move": command.move, "duration_ms": command.duration_ms}
        for command in commands
    ]


def _load_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlInputError(f"could not read history: {error}") from error

    if not isinstance(value, dict):
        raise ControlInputError("history must be a JSON object")
    if value.get("version") != HISTORY_VERSION:
        raise ControlInputError(f"history version must be {HISTORY_VERSION}")
    return value


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError as error:
        raise ControlRuntimeError(f"could not write history: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
