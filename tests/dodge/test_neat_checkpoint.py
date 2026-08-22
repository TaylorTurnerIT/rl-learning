from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import neat

from dodge.control import PROJECT_ROOT
from dodge.neat.checkpoint import (
    CHECKPOINT_RETENTION,
    RunCheckpointer,
    checkpoint_paths,
    latest_checkpoint,
)


def test_v22_checkpoint_retention_keeps_only_the_newest_five(tmp_path: Path) -> None:
    saved: list[tuple[int, Path]] = []
    reporter = RunCheckpointer(
        tmp_path,
        on_saved=lambda generation, path: saved.append((generation, path)),
    )

    for generation in range(CHECKPOINT_RETENTION + 2):
        reporter.start_generation(generation)
        reporter.end_generation({"config": generation}, {"population": generation}, {})

    assert [path.name for path in checkpoint_paths(tmp_path)] == [
        "checkpoint-000003.gz",
        "checkpoint-000004.gz",
        "checkpoint-000005.gz",
        "checkpoint-000006.gz",
        "checkpoint-000007.gz",
    ]
    assert saved[-1] == (7, latest_checkpoint(tmp_path))
    assert not list(tmp_path.glob(".*.tmp"))

    with gzip.open(latest_checkpoint(tmp_path), "rb") as saved_state:
        generation, config, population, species, _random_state = pickle.load(
            saved_state
        )
    assert (generation, config, population, species) == (
        7,
        {"config": 6},
        {"population": 6},
        {},
    )


def test_v22_checkpoint_restores_neat_population_state(tmp_path: Path) -> None:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        PROJECT_ROOT / "src/dodge/neat/config-dodge",
    )
    population = neat.Population(config)
    reporter = RunCheckpointer(tmp_path, on_saved=lambda _generation, _path: None)

    reporter.start_generation(0)
    reporter.end_generation(config, population.population, population.species)
    restored = neat.Checkpointer.restore_checkpoint(latest_checkpoint(tmp_path))

    assert restored.generation == 1
    assert set(restored.population) == set(population.population)
