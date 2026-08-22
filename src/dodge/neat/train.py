from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from dodge.neat.environment import NEAT_HISTORY_DIRECTORY
from dodge.neat.evaluator import (
    DodgeEvaluator,
    GenerationEvaluation,
    default_worker_count,
)

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
    parser.add_argument(
        "--workers",
        type=_positive,
        default=default_worker_count(),
        help="concurrent genome workers (default: up to 8 CPU cores)",
    )
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
        workers=arguments.workers,
    )
    population = neat.Population(config)
    population.run(evaluator, arguments.generations)
    record = _write_run_record(run_directory, arguments, evaluator)
    print("\nGeneration results")
    print(format_generation_table(evaluator.generation_history))
    print("\nTraining complete")
    print(json.dumps(record["final_generation"], indent=2))
    print(f"run: {run_directory}")
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
) -> dict[str, object]:
    generations = [
        generation_summary(result) for result in evaluator.generation_history
    ]
    record = {
        "kind": "neat_run",
        "config": str(arguments.config),
        "requested_generations": arguments.generations,
        "step_frames": arguments.step_frames,
        "enemy_slots": arguments.enemy_slots,
        "aoe_slots": arguments.aoe_slots,
        "workers": arguments.workers,
        "generations": generations,
        "final_generation": generations[-1] if generations else None,
    }
    (run_directory / "run.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def generation_summary(result: GenerationEvaluation) -> dict[str, object]:
    fitness = tuple(result.mean_survival_frames.values())
    average = sum(fitness) / len(fitness) if fitness else 0.0
    return {
        "generation": result.generation,
        "population": len(fitness),
        "average_survival_frames": average,
        "best_survival_frames": result.best_fitness,
        "best_genome_id": result.best_genome_id,
        "seed_bank": list(result.seeds),
        "network_visualization": (
            str(result.network_visualization)
            if result.network_visualization is not None
            else None
        ),
    }


def format_generation_table(generations: Iterable[GenerationEvaluation]) -> str:
    rows = [generation_summary(generation) for generation in generations]
    if not rows:
        return "(no completed generations)"
    headers = ("Gen", "Population", "Average", "Best", "Best ID", "Seeds")
    values = [
        (
            str(row["generation"]),
            str(row["population"]),
            f"{float(row['average_survival_frames']):.1f}",
            f"{float(row['best_survival_frames']):.1f}",
            str(row["best_genome_id"]),
            ",".join(str(seed) for seed in row["seed_bank"]),
        )
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in values))
        for index, header in enumerate(headers)
    ]
    header = "  ".join(
        value.ljust(widths[index]) for index, value in enumerate(headers)
    )
    separator = "  ".join("-" * width for width in widths)
    body = "\n".join(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in values
    )
    return "\n".join((header, separator, body))


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
