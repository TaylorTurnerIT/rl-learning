from __future__ import annotations

import json
from pathlib import Path

import pytest

from dodge.ng.hpo_report import build_hpo_report
from dodge.ng.manifest import SeedManifest
from dodge.ng.report import summarize_evaluation


def _evaluation(seeds: tuple[int, ...], survival: int) -> dict[str, object]:
    return {
        "seeds": list(seeds),
        "survival_frames": [survival + index for index, _ in enumerate(seeds)],
        "terminated": [False] * len(seeds),
    }


def test_build_hpo_report_aggregates_trials_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    manifest = SeedManifest.fresh_default()
    run_directory = tmp_path / "hpo"
    trial_directory = run_directory / "trial-0000"
    trial_directory.mkdir(parents=True)
    inner = _evaluation(manifest.training_seeds[:10], 800)
    inner_summary = summarize_evaluation(inner)
    (trial_directory / "run.json").write_text(
        json.dumps(
            {
                "config": {
                    "observation_mode": "legacy",
                    "observation_size": 225,
                    "native_lanes": 2,
                    "hold_decisions": 2,
                    "step_frames": 4,
                    "max_episode_steps": 10,
                },
                "updates_completed": 20,
                "best_inner": {"mean_survival_frames": 805, "step": 20},
                "selected_model": "best_inner",
                "holdout_evaluated": False,
                "training_evaluated": False,
                "final_validation": inner,
            }
        ),
        encoding="utf-8",
    )
    (trial_directory / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "step": step,
                    "collection_seconds": 2.0,
                    "learning_seconds": 1.0,
                    "evaluation_seconds": 0.0,
                    "checkpoint_seconds": 0.0,
                    "loss": 4.0,
                    "waypoint_action_counts": [1, 0, 0, 0, 0, 0, 0, 0, 1],
                    "inner_validation": inner_summary,
                }
            )
            for step in (10, 20)
        )
        + "\n",
        encoding="utf-8",
    )
    selected = {
        "number": 0,
        "state": "COMPLETE",
        "value": 900.0,
        "params": {
            "learning_rate": 1e-4,
            "batch_size": 64,
            "n_step": 3,
            "target_update_interval": 1000,
        },
        "user_attrs": {"rung_metrics": []},
        "intermediate_values": {"20": 900.0},
    }
    hpo = {
        "kind": "dodge_ng_waypoint_dqn_hpo",
        "manifest_sha256": manifest.sha256,
        "training_seeds": list(manifest.training_seeds),
        "inner_validation_seeds": list(manifest.training_seeds[:10]),
        "holdout_seeds": list(manifest.holdout_seeds),
        "holdout_evaluated_once_after_selection": True,
        "selected_trial": selected,
        "trials": [selected],
        "training_evaluation": _evaluation(manifest.training_seeds, 850),
        "holdout_evaluation": _evaluation(manifest.holdout_seeds, 700),
    }
    run_directory.mkdir(exist_ok=True)
    (run_directory / "hpo.json").write_text(
        json.dumps(hpo),
        encoding="utf-8",
    )

    report = build_hpo_report(run_directory, manifest)

    assert report["routing"]["trial_objectives_holdout_free"] is True  # type: ignore[index]
    assert report["trials"][0]["metrics"]["action_counts"] == [
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        2,
    ]  # type: ignore[index]
    assert report["trials"][0]["metrics"]["phase_timing"]["available"] is True  # type: ignore[index]
    findings = {item["name"]: item for item in report["failure_modes"]}  # type: ignore[index]
    assert findings["relevance_gate"]["finding"] == (
        "training-side gate passed; holdout below gate"
    )
    assert findings["best_checkpoint_timing"]["finding"] == (
        "best inner checkpoint was final step"
    )
    assert (run_directory / "HPO_PERFORMANCE_REPORT.md").is_file()
    assert (run_directory / "hpo_performance_report.json").is_file()
    assert all((run_directory / name).is_file() for name in report["plots"])


def test_hpo_report_rejects_holdout_inner_validation(tmp_path: Path) -> None:
    manifest = SeedManifest.fresh_default()
    run_directory = tmp_path / "hpo"
    run_directory.mkdir()
    bad = {
        "manifest_sha256": manifest.sha256,
        "training_seeds": list(manifest.training_seeds),
        "inner_validation_seeds": [manifest.holdout_seeds[0]],
        "holdout_seeds": list(manifest.holdout_seeds),
        "holdout_evaluated_once_after_selection": True,
    }
    (run_directory / "hpo.json").write_text(json.dumps(bad), encoding="utf-8")

    with pytest.raises(ValueError, match="holdout seed"):
        build_hpo_report(run_directory, manifest)
