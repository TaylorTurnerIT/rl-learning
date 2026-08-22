from __future__ import annotations

import argparse
import json
from pathlib import Path

from dodge.neat.evaluator import GenerationEvaluation
from dodge.neat.train import (
    _write_run_record,
    format_generation_table,
    generation_summary,
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

    assert "Gen  Population  Average  Best   Best ID  Seeds" in table
    assert "1    2           200.0    300.0  11       1,2,3" in table
    assert "2    2           350.0    500.0  11       2,3,4" in table


def test_run_record_keeps_concise_generation_results(tmp_path: Path) -> None:
    evaluator = argparse.Namespace(
        generation_history=[_generation(1, first=100, second=300)]
    )
    arguments = argparse.Namespace(
        config=Path("config-dodge"),
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
