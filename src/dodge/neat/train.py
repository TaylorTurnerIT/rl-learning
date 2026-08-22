from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from dodge.neat.checkpoint import (
    CHECKPOINT_RETENTION,
    RunCheckpointer,
    latest_checkpoint,
)
from dodge.neat.environment import NEAT_HISTORY_DIRECTORY
from dodge.neat.evaluator import (
    SEED_BANK_GENERATIONS,
    DodgeEvaluator,
    GenerationEvaluation,
    SeedBankSchedule,
    default_worker_count,
)
from dodge.neat.speciation import (
    SpeciesMetrics,
    ensure_species_monitor,
    format_species_metrics,
)

DEFAULT_CONFIG = Path(__file__).with_name("config-dodge-v3")
DEFAULT_STEP_FRAMES = 3
RUN_VERSION = 3
RESUMABLE_RUN_VERSIONS = (2, RUN_VERSION)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-neat-train",
        description="Train NEAT on hidden live Dodge episodes.",
    )
    parser.add_argument("--generations", type=_positive, default=100)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--history-dir", type=Path, default=NEAT_HISTORY_DIRECTORY)
    parser.add_argument(
        "--resume",
        type=Path,
        help="existing checkpointed run directory; generations are additional",
    )
    parser.add_argument("--step-frames", type=_step_frames)
    parser.add_argument(
        "--time-to-intersection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include bounded per-hazard time-to-intersection features",
    )
    parser.add_argument("--enemy-slots", type=_positive, default=16)
    parser.add_argument("--aoe-slots", type=_positive, default=8)
    parser.add_argument(
        "--workers",
        type=_positive,
        default=default_worker_count(),
        help="concurrent genome workers (default: up to 8 CPU cores)",
    )
    parser.add_argument(
        "--evolution-seed",
        type=_nonnegative,
        help="population RNG seed; omitted new runs receive and record entropy",
    )
    arguments = parser.parse_args(argv)

    existing: dict[str, object] | None = None
    if arguments.resume is None:
        arguments.config = (
            DEFAULT_CONFIG if arguments.config is None else arguments.config
        )
        arguments.step_frames = (
            DEFAULT_STEP_FRAMES
            if arguments.step_frames is None
            else arguments.step_frames
        )
        schedule_seed = secrets.randbits(64)
        evolution_seed = (
            secrets.randbits(64)
            if arguments.evolution_seed is None
            else arguments.evolution_seed
        )
    else:
        existing = _load_run_record(arguments.resume)
        _apply_resume_defaults(arguments, existing)
        schedule_seed = _schedule_seed(existing)
        evolution_seed = _evolution_seed(existing, arguments.evolution_seed)

    import neat

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        arguments.config,
    )
    configured_time_to_intersection = _configured_time_to_intersection(
        config,
        enemy_slots=arguments.enemy_slots,
        aoe_slots=arguments.aoe_slots,
    )
    if arguments.time_to_intersection is None:
        arguments.time_to_intersection = configured_time_to_intersection
    elif arguments.time_to_intersection != configured_time_to_intersection:
        raise ValueError(
            "time_to_intersection does not match the NEAT config input width"
        )
    if arguments.resume is None:
        run_directory = _create_run_directory(arguments.history_dir)
        summaries: list[dict[str, object]] = []
        _write_run_record(
            run_directory,
            arguments,
            summaries=summaries,
            schedule_seed=schedule_seed,
            evolution_seed=evolution_seed,
        )
        population = neat.Population(config, seed=evolution_seed)
    else:
        run_directory = arguments.resume
        assert existing is not None
        _validate_resume(existing, arguments)
        summaries = _summaries(existing)
        checkpoint = latest_checkpoint(run_directory)
        population = neat.Checkpointer.restore_checkpoint(checkpoint)

    species_monitor = ensure_species_monitor(population)

    evaluator = DodgeEvaluator(
        step_frames=arguments.step_frames,
        enemy_slots=arguments.enemy_slots,
        aoe_slots=arguments.aoe_slots,
        include_time_to_intersection=arguments.time_to_intersection,
        history_directory=run_directory,
        seed_bank_schedule=SeedBankSchedule(schedule_seed),
        progress=print,
        workers=arguments.workers,
    )
    evaluator.generation = population.generation

    def checkpoint_saved(generation: int, checkpoint: Path) -> None:
        result = evaluator.last_generation
        if result is None or result.generation != generation:
            raise RuntimeError("checkpoint did not follow a completed generation")
        metrics = species_monitor.latest
        summaries.append(generation_summary(result, species_metrics=metrics))
        _write_run_record(
            run_directory,
            arguments,
            summaries=summaries,
            checkpoint=checkpoint,
            schedule_seed=schedule_seed,
            evolution_seed=evolution_seed,
        )
        print(f"generation {generation} complete: {format_species_metrics(metrics)}")

    population.add_reporter(RunCheckpointer(run_directory, on_saved=checkpoint_saved))
    population.run(evaluator, arguments.generations)
    record = _write_run_record(
        run_directory,
        arguments,
        summaries=summaries,
        checkpoint=_latest_checkpoint_or_none(run_directory),
        schedule_seed=schedule_seed,
        evolution_seed=evolution_seed,
    )
    print("\nGeneration results")
    print(format_generation_summaries(summaries))
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
    evaluator: DodgeEvaluator | None = None,
    *,
    summaries: list[dict[str, object]] | None = None,
    checkpoint: Path | None = None,
    schedule_seed: int | None = None,
    evolution_seed: int | None = None,
) -> dict[str, object]:
    generations = summaries
    if generations is None:
        if evaluator is None:
            generations = []
        else:
            generations = [
                generation_summary(result) for result in evaluator.generation_history
            ]
    record = {
        "version": RUN_VERSION,
        "kind": "neat_run",
        "config": str(arguments.config.resolve()),
        "config_sha256": _config_sha256(arguments.config),
        "requested_generations": arguments.generations,
        "step_frames": arguments.step_frames,
        "time_to_intersection": getattr(arguments, "time_to_intersection", False),
        "enemy_slots": arguments.enemy_slots,
        "aoe_slots": arguments.aoe_slots,
        "workers": arguments.workers,
        "seed_bank_generations": SEED_BANK_GENERATIONS,
        "seed_schedule_seed": schedule_seed,
        "evolution_seed": evolution_seed,
        "checkpoint_retention": CHECKPOINT_RETENTION,
        "latest_checkpoint": checkpoint.name if checkpoint is not None else None,
        "completed_generations": len(generations),
        "generations": generations,
        "final_generation": generations[-1] if generations else None,
    }
    _write_json(run_directory / "run.json", record)
    return record


def _load_run_record(run_directory: Path) -> dict[str, object]:
    try:
        value = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read NEAT run record: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("NEAT run record must be an object")
    return value


def _validate_resume(
    record: Mapping[str, object], arguments: argparse.Namespace
) -> None:
    if (
        record.get("version") not in RESUMABLE_RUN_VERSIONS
        or record.get("kind") != "neat_run"
    ):
        raise ValueError("run does not support checkpoint resume")
    expected = {
        "config_sha256": _config_sha256(arguments.config),
        "step_frames": arguments.step_frames,
        "time_to_intersection": getattr(arguments, "time_to_intersection", False),
        "enemy_slots": arguments.enemy_slots,
        "aoe_slots": arguments.aoe_slots,
    }
    for key, value in expected.items():
        recorded_value = (
            record.get(key, False) if key == "time_to_intersection" else record.get(key)
        )
        if recorded_value != value:
            raise ValueError(f"resume setting differs: {key}")


def _apply_resume_defaults(
    arguments: argparse.Namespace, record: Mapping[str, object]
) -> None:
    if arguments.config is None:
        config = record.get("config")
        if not isinstance(config, str):
            raise ValueError("run record has no config path")
        arguments.config = Path(config)
    if arguments.step_frames is None:
        arguments.step_frames = _record_step_frames(record)
    if arguments.time_to_intersection is None:
        arguments.time_to_intersection = record.get("time_to_intersection", False)
    if not isinstance(arguments.time_to_intersection, bool):
        raise ValueError("run record has invalid time_to_intersection setting")


def _schedule_seed(record: Mapping[str, object]) -> int:
    seed = record.get("seed_schedule_seed")
    if isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0:
        return seed
    return secrets.randbits(64)


def _evolution_seed(record: Mapping[str, object], requested: int | None) -> int | None:
    stored = record.get("evolution_seed")
    if stored is None:
        if requested is not None:
            raise ValueError("legacy run has no reproducible evolution seed")
        return None
    if isinstance(stored, bool) or not isinstance(stored, int) or stored < 0:
        raise ValueError("run record has invalid evolution_seed")
    if requested is not None and requested != stored:
        raise ValueError("resume setting differs: evolution_seed")
    return stored


def _record_step_frames(record: Mapping[str, object]) -> int:
    value = record.get("step_frames")
    if isinstance(value, bool) or not isinstance(value, int) or not 3 <= value <= 5:
        raise ValueError("run record has invalid step_frames setting")
    return value


def _configured_time_to_intersection(
    config: object, *, enemy_slots: int, aoe_slots: int
) -> bool:
    input_keys = tuple(
        getattr(getattr(config, "genome_config", None), "input_keys", ())
    )
    legacy_size = 5 + (enemy_slots + aoe_slots) * 8
    enhanced_size = 5 + (enemy_slots + aoe_slots) * 9
    if len(input_keys) == legacy_size:
        return False
    if len(input_keys) == enhanced_size:
        return True
    raise ValueError(
        f"NEAT config expects {len(input_keys)} inputs, but Dodge projection has "
        f"{legacy_size} or {enhanced_size}"
    )


def _summaries(record: Mapping[str, object]) -> list[dict[str, object]]:
    value = record.get("generations")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("NEAT run record has invalid generation history")
    return [dict(item) for item in value]


def _config_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"could not read NEAT config: {error}") from error


def _latest_checkpoint_or_none(run_directory: Path) -> Path | None:
    try:
        return latest_checkpoint(run_directory)
    except FileNotFoundError:
        return None


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def generation_summary(
    result: GenerationEvaluation,
    *,
    species_metrics: SpeciesMetrics | None = None,
) -> dict[str, object]:
    fitness = tuple(result.mean_survival_frames.values())
    average = sum(fitness) / len(fitness) if fitness else 0.0
    return {
        "generation": result.generation,
        "population": len(fitness),
        "average_survival_frames": average,
        "best_survival_frames": result.best_fitness,
        "best_genome_id": result.best_genome_id,
        "seed_bank": list(result.seeds),
        "validation_survival_frames": result.validation_fitness,
        "validation_seed_bank": list(result.validation_seeds),
        "benchmark_survival_frames": result.benchmark_fitness,
        "benchmark_seed_bank": list(result.benchmark_seeds),
        "species": (
            species_metrics.as_json()
            if species_metrics is not None
            and species_metrics.generation == result.generation
            else None
        ),
        "network_visualization": (
            str(result.network_visualization)
            if result.network_visualization is not None
            else None
        ),
    }


def format_generation_table(generations: Iterable[GenerationEvaluation]) -> str:
    return format_generation_summaries(
        generation_summary(generation) for generation in generations
    )


def format_generation_summaries(rows: Iterable[Mapping[str, object]]) -> str:
    rows = list(rows)
    if not rows:
        return "(no completed generations)"
    headers = (
        "Gen",
        "Population",
        "Average",
        "Best",
        "Validation",
        "Benchmark",
        "Species",
        "Best ID",
        "Seeds",
    )
    values = [
        (
            str(row["generation"]),
            str(row["population"]),
            f"{float(row['average_survival_frames']):.1f}",
            f"{float(row['best_survival_frames']):.1f}",
            (
                "-"
                if row.get("validation_survival_frames") is None
                else f"{float(row['validation_survival_frames']):.1f}"
            ),
            (
                "-"
                if row.get("benchmark_survival_frames") is None
                else f"{float(row['benchmark_survival_frames']):.1f}"
            ),
            _species_cell(row.get("species")),
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


def _nonnegative(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _species_cell(value: object) -> str:
    if not isinstance(value, Mapping):
        return "-"
    count = value.get("count")
    sizes = value.get("sizes")
    if not isinstance(count, int) or not isinstance(sizes, (list, tuple)):
        return "-"
    return f"{count}[{','.join(str(size) for size in sizes)}]"


if __name__ == "__main__":
    raise SystemExit(main())
