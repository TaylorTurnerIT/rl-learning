from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dodge.neat.evaluator import GenerationEvaluation
from dodge.neat.train import (
    DEFAULT_CONFIG,
    RUN_VERSION,
    _config_sha256,
    _validate_resume,
    _write_run_record,
    format_generation_table,
    generation_summary,
    main,
)


def _generation(number: int, *, first: float, second: float) -> GenerationEvaluation:
    return GenerationEvaluation(
        generation=number,
        seeds=(number, number + 1, number + 2),
        mean_survival_frames={10: first, 11: second},
        traces={},
        best_genome_id=11,
        best_fitness=second,
        network_summary="network=test",
        network_visualization=Path(f"/tmp/generation-{number:04d}/network.html"),
    )


def test_final_generation_table_shows_average_and_best_fitness() -> None:
    table = format_generation_table(
        (_generation(1, first=100, second=300), _generation(2, first=200, second=500))
    )

    assert "Gen  Population  Average  Best   Validation  Best ID  Seeds" in table
    assert "1    2           200.0    300.0  -           11       1,2,3" in table
    assert "2    2           350.0    500.0  -           11       2,3,4" in table


def test_run_record_keeps_concise_generation_results(tmp_path: Path) -> None:
    evaluator = argparse.Namespace(
        generation_history=[_generation(1, first=100, second=300)]
    )
    arguments = argparse.Namespace(
        config=DEFAULT_CONFIG,
        generations=100,
        step_frames=4,
        enemy_slots=16,
        aoe_slots=8,
        workers=8,
    )

    record = _write_run_record(tmp_path, arguments, evaluator)  # type: ignore[arg-type]

    saved = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert record["final_generation"] == generation_summary(
        evaluator.generation_history[0]
    )
    assert saved["final_generation"]["average_survival_frames"] == 200
    assert saved["final_generation"]["best_survival_frames"] == 300
    assert "winner" not in saved
    assert saved["version"] == RUN_VERSION
    assert saved["checkpoint_retention"] == 5


def test_v26_default_config_is_sparse_three_frame_v2() -> None:
    import neat

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        DEFAULT_CONFIG,
    )

    assert config.genome_config.num_inputs == 221
    assert config.genome_config.initial_connection == "partial_direct"
    assert config.genome_config.connection_fraction == 0.15


def test_resume_rejects_changed_observation_settings() -> None:
    arguments = argparse.Namespace(
        config=DEFAULT_CONFIG,
        step_frames=4,
        time_to_intersection=False,
        enemy_slots=16,
        aoe_slots=8,
    )
    record = {
        "version": RUN_VERSION,
        "kind": "neat_run",
        "config_sha256": _config_sha256(DEFAULT_CONFIG),
        "step_frames": 4,
        "time_to_intersection": False,
        "enemy_slots": 12,
        "aoe_slots": 8,
    }

    with pytest.raises(ValueError, match="enemy_slots"):
        _validate_resume(record, arguments)


def test_v26_v1_run_record_resumes_without_time_to_intersection_flag() -> None:
    legacy_config = DEFAULT_CONFIG.with_name("config-dodge")
    arguments = argparse.Namespace(
        config=legacy_config,
        step_frames=4,
        time_to_intersection=False,
        enemy_slots=16,
        aoe_slots=8,
    )
    record = {
        "version": RUN_VERSION,
        "kind": "neat_run",
        "config_sha256": _config_sha256(legacy_config),
        "step_frames": 4,
        "enemy_slots": 16,
        "aoe_slots": 8,
    }

    _validate_resume(record, arguments)


class _FakeEvaluator:
    def __init__(self, **_: object) -> None:
        self.generation = 0
        self.last_generation: GenerationEvaluation | None = None


class _FakePopulation:
    def __init__(
        self, _config: object, state: tuple[object, ...] | None = None
    ) -> None:
        self.population = {1: "genome"}
        self.species = {"species": 1}
        self.generation = 0
        if state is not None:
            self.population, self.species, self.generation = state
        self._reporters: list[object] = []

    def add_reporter(self, reporter: object) -> None:
        self._reporters.append(reporter)

    def run(self, evaluator: _FakeEvaluator, generations: int) -> None:
        for _ in range(generations):
            for reporter in self._reporters:
                reporter.start_generation(self.generation)  # type: ignore[attr-defined]
            evaluator.generation = self.generation + 1
            evaluator.last_generation = _generation(
                evaluator.generation, first=10, second=20
            )
            for reporter in self._reporters:
                reporter.end_generation(  # type: ignore[attr-defined]
                    {"generation": self.generation}, self.population, self.species
                )
            self.generation += 1


class _FakeCheckpointer:
    restores: list[Path] = []

    @classmethod
    def restore_checkpoint(cls, path: Path) -> _FakePopulation:
        cls.restores.append(path)
        with gzip.open(path, "rb") as input_file:
            generation, config, population, species, _random_state = pickle.load(
                input_file
            )
        return _FakePopulation(config, (population, species, generation))


def test_v22_main_creates_and_resumes_the_same_checkpointed_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_neat = SimpleNamespace(
        DefaultGenome=object,
        DefaultReproduction=object,
        DefaultSpeciesSet=object,
        DefaultStagnation=object,
        Config=lambda *_args: {"config": "fake"},
        Population=_FakePopulation,
        Checkpointer=_FakeCheckpointer,
    )
    _FakeCheckpointer.restores.clear()
    monkeypatch.setitem(sys.modules, "neat", fake_neat)
    monkeypatch.setattr("dodge.neat.train.DodgeEvaluator", _FakeEvaluator)

    assert main(["--history-dir", str(tmp_path), "--generations", "1"]) == 0
    run_directory = next(tmp_path.iterdir())
    assert main(["--resume", str(run_directory), "--generations", "1"]) == 0

    record = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    assert record["completed_generations"] == 2
    assert [row["generation"] for row in record["generations"]] == [1, 2]
    assert record["latest_checkpoint"] == "checkpoint-000002.gz"
    assert record["config"] == str(DEFAULT_CONFIG.resolve())
    assert record["step_frames"] == 3
    assert record["time_to_intersection"] is True
    assert record["seed_bank_generations"] == 5
    assert _FakeCheckpointer.restores == [run_directory / "checkpoint-000001.gz"]
