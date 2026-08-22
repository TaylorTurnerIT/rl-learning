from __future__ import annotations

import gzip
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

import neat
import pytest
from neat.reporting import ReporterSet

from dodge.control import PEMSA_PATH, PROJECT_ROOT
from dodge.neat.checkpoint import (
    CHECKPOINT_RETENTION,
    RunCheckpointer,
    checkpoint_paths,
    latest_checkpoint,
)
from dodge.neat.train import main as train_main


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


def test_v24_checkpoint_excludes_live_save_callback_from_pickle(tmp_path: Path) -> None:
    reporter = RunCheckpointer(tmp_path, on_saved=lambda _generation, _path: None)
    reporters = ReporterSet()
    reporters.add(reporter)
    species = SimpleNamespace(reporters=reporters)

    reporter.start_generation(0)
    reporter.end_generation({}, {}, species)

    assert latest_checkpoint(tmp_path).is_file()


def test_v23_checkpoint_reporter_accepts_every_neat_lifecycle_hook(
    tmp_path: Path,
) -> None:
    reporter = RunCheckpointer(tmp_path, on_saved=lambda _generation, _path: None)
    reporters = ReporterSet()
    reporters.add(reporter)

    reporters.start_generation(0)
    reporters.post_evaluate({}, {}, {}, object())
    reporters.post_reproduction({}, {}, {})
    reporters.species_stagnant(1, object())
    reporters.info("checkpoint test")
    reporters.complete_extinction()
    reporters.found_solution({}, 0, object())
    reporters.end_generation({}, {}, {})


@pytest.mark.skipif(
    not PEMSA_PATH.is_file() or shutil.which("Xvfb") is None,
    reason="requires the checked-in Pemsa runtime and Xvfb",
)
def test_v24_tiny_live_population_checkpoints_after_generation_one(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "tiny-config-dodge"
    config_path.write_text(
        (PROJECT_ROOT / "src/dodge/neat/config-dodge")
        .read_text(encoding="utf-8")
        .replace("pop_size              = 50", "pop_size              = 2")
        .replace("min_species_size   = 2", "min_species_size   = 1"),
        encoding="utf-8",
    )

    assert (
        train_main(
            [
                "--config",
                str(config_path),
                "--history-dir",
                str(tmp_path / "history"),
                "--generations",
                "1",
                "--workers",
                "1",
            ]
        )
        == 0
    )

    run_directory = next((tmp_path / "history").iterdir())
    assert latest_checkpoint(run_directory).name == "checkpoint-000001.gz"
