from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dodge.control import (
    ControlInputError,
    ControlRuntimeError,
    MovementCommand,
    parse_seed,
)
from dodge.headless import HeadlessResult, replay_commands
from dodge.neat.environment import NEAT_HISTORY_DIRECTORY, EpisodeTrace, load_episode

_MENU_START_FRAMES = 3


def trace_commands(trace: EpisodeTrace) -> list[MovementCommand]:
    return [
        MovementCommand("x", _frames_to_milliseconds(_MENU_START_FRAMES)),
        *[
            MovementCommand(action, _frames_to_milliseconds(trace.step_frames))
            for action in trace.actions
        ],
    ]


def replay_episode(trace: EpisodeTrace) -> HeadlessResult:
    result = replay_commands(
        trace_commands(trace),
        seed=trace.seed,
        wait_for_game_start=True,
        legacy_mouse_input=trace.input_mode == "legacy_mouse",
    )
    expected = trace.result
    actual = {
        "score": result["score"],
        "frames": result["frames"],
        "survival_frames": result["survival_frames"],
        "seed": result["seed"],
    }
    wanted = {
        "score": expected.score,
        "frames": expected.frames,
        "survival_frames": expected.survival_frames,
        "seed": expected.seed,
    }
    if actual != wanted:
        raise ControlRuntimeError(
            f"NEAT replay diverged: expected {wanted}, got {actual}"
        )
    return result


def latest_run(directory: Path = NEAT_HISTORY_DIRECTORY) -> Path:
    runs = sorted(
        path
        for path in directory.glob("run-*")
        if path.is_dir() and (path / "run.json").is_file()
    )
    if not runs:
        raise ControlInputError("NEAT history contains no saved runs")
    return runs[-1]


def replay_generation(directory: Path, epoch: int) -> HeadlessResult:
    trace = generation_winner_trace(directory, epoch)
    print(f"replaying generation {epoch} (genome winner, seed {trace.seed})")
    return replay_episode(trace)


def replay_latest_run(
    epoch: int, directory: Path = NEAT_HISTORY_DIRECTORY
) -> HeadlessResult:
    return replay_generation(latest_run(directory), epoch)


def generation_winner_trace(directory: Path, epoch: int) -> EpisodeTrace:
    if epoch < 1:
        raise ControlInputError("epoch must be a positive integer")
    record = _load_run_record(directory / "run.json")
    generation = _generation_record(record, epoch)
    genome_id = _nonnegative_integer(generation.get("best_genome_id"), "best_genome_id")
    seeds = _seed_bank(generation.get("seed_bank"))
    episode_directory = directory / f"generation-{epoch:04d}"
    paths = [
        episode_directory / f"genome-{genome_id:04d}-seed-{seed}.json" for seed in seeds
    ]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ControlInputError(
            f"generation {epoch} winner history is missing: {', '.join(missing)}"
        )
    traces = [load_episode(path) for path in paths]
    return max(traces, key=lambda trace: (trace.result.survival_frames, -trace.seed))


def replay_latest_run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-neat-replay-latest",
        description="Replay one generation winner from the most recent saved NEAT run.",
    )
    parser.add_argument(
        "epoch", type=_positive_epoch, help="generation number to replay"
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=NEAT_HISTORY_DIRECTORY,
        help="directory containing NEAT run-* history directories",
    )
    arguments = parser.parse_args(argv)
    try:
        result = replay_latest_run(arguments.epoch, arguments.history_dir)
    except (ControlInputError, ControlRuntimeError, ValueError) as error:
        print(f"dodge-neat-replay-latest: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "replay-latest":
        return replay_latest_run_main(arguments[1:])

    parser = argparse.ArgumentParser(
        prog="dodge-neat-replay",
        description="Replay a saved NEAT Dodge episode without live controls.",
    )
    parser.add_argument("episode", type=Path, help="saved NEAT episode JSON")
    arguments = parser.parse_args(arguments)
    try:
        result = replay_episode(load_episode(arguments.episode))
    except (ControlInputError, ControlRuntimeError, ValueError) as error:
        print(f"dodge-neat-replay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


def _frames_to_milliseconds(frames: int) -> int:
    return (frames * 1_000) // 60


def _load_run_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlInputError(f"could not read NEAT run record: {error}") from error
    if not isinstance(value, dict) or value.get("kind") != "neat_run":
        raise ControlInputError("unsupported NEAT run record")
    return value


def _generation_record(record: dict[str, object], epoch: int) -> dict[str, object]:
    generations = record.get("generations")
    if not isinstance(generations, list):
        raise ControlInputError("NEAT run record has invalid generation history")
    for generation in generations:
        if isinstance(generation, dict) and generation.get("generation") == epoch:
            return generation
    raise ControlInputError(f"NEAT run contains no generation {epoch}")


def _seed_bank(value: object) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ControlInputError("generation has no recorded seed bank")
    try:
        return [parse_seed(str(seed)) for seed in value]
    except ControlInputError as error:
        raise ControlInputError("generation has an invalid seed bank") from error


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ControlInputError(f"generation has an invalid {name}")
    return value


def _positive_epoch(value: str) -> int:
    try:
        epoch = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("epoch must be a positive integer") from error
    if epoch < 1:
        raise argparse.ArgumentTypeError("epoch must be a positive integer")
    return epoch


if __name__ == "__main__":
    raise SystemExit(main())
