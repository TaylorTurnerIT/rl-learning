from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from dodge.neat.environment import NEAT_HISTORY_DIRECTORY
from dodge.neat.evaluator import DodgeEvaluator

DEFAULT_CONFIG = Path(__file__).with_name("config-dodge")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-neat-train",
        description="Train NEAT on hidden live Dodge episodes.",
    )
    parser.add_argument("--generations", type=_positive, default=100)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--history-dir", type=Path, default=NEAT_HISTORY_DIRECTORY)
    parser.add_argument("--step-frames", type=_step_frames, default=4)
    parser.add_argument("--enemy-slots", type=_positive, default=16)
    parser.add_argument("--aoe-slots", type=_positive, default=8)
    arguments = parser.parse_args(argv)

    import neat

    run_directory = _create_run_directory(arguments.history_dir)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        arguments.config,
    )
    evaluator = DodgeEvaluator(
        step_frames=arguments.step_frames,
        enemy_slots=arguments.enemy_slots,
        aoe_slots=arguments.aoe_slots,
        history_directory=run_directory,
        progress=print,
    )
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    winner = population.run(evaluator, arguments.generations)
    _write_run_record(run_directory, arguments, evaluator, winner)
    print(json.dumps({"run": str(run_directory), "winner": str(winner)}))
    return 0


def _create_run_directory(history_directory: Path) -> Path:
    name = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%S.%fZ")
    run_directory = history_directory / name
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _write_run_record(
    run_directory: Path,
    arguments: argparse.Namespace,
    evaluator: DodgeEvaluator,
    winner: object,
) -> None:
    last_generation = evaluator.last_generation
    record = {
        "kind": "neat_run",
        "config": str(arguments.config),
        "generations": arguments.generations,
        "step_frames": arguments.step_frames,
        "enemy_slots": arguments.enemy_slots,
        "aoe_slots": arguments.aoe_slots,
        "last_seed_bank": list(last_generation.seeds) if last_generation else [],
        "winner": str(winner),
    }
    (run_directory / "run.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def _positive(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _step_frames(value: str) -> int:
    parsed = _positive(value)
    if not 3 <= parsed <= 5:
        raise argparse.ArgumentTypeError("must be between 3 and 5")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
