from __future__ import annotations

import json
from pathlib import Path

from dodge.ng.manifest import SeedManifest
from dodge.ng.report import build_report, summarize_evaluation


def _evaluation(seeds: tuple[int, ...], offset: int) -> dict[str, object]:
    return {
        "seeds": list(seeds),
        "survival_frames": [offset + index for index, _ in enumerate(seeds)],
        "terminated": [False] * len(seeds),
    }


def test_summarize_evaluation_reports_distribution_statistics() -> None:
    summary = summarize_evaluation(
        {
            "seeds": [1, 2, 3, 4, 5],
            "survival_frames": [10, 20, 30, 40, 50],
            "terminated": [True, True, False, False, False],
        }
    )

    assert summary["mean_survival_frames"] == 30.0
    assert summary["median_survival_frames"] == 30
    assert summary["p10_survival_frames"] == 10
    assert summary["worst_survival_frames"] == 10
    assert summary["best_survival_frames"] == 50
    assert summary["horizon_completion_fraction"] == 0.6


def test_build_report_validates_splits_and_writes_plots(tmp_path: Path) -> None:
    manifest = SeedManifest.fresh_default()
    run_directory = tmp_path / "baseline"
    run_directory.mkdir()
    run_record = {
        "config": {
            "backend": "native",
            "observation_mode": "board",
            "training_seeds": list(manifest.training_seeds),
            "training_seed_manifest": manifest.sha256,
        },
        "updates_completed": 2,
        "global_step": 20,
        "final_training_evaluation": _evaluation(manifest.training_seeds, 100),
        "final_validation": _evaluation(manifest.training_seeds[:10], 90),
        "final_evaluation": _evaluation(manifest.holdout_seeds, 50),
    }
    (run_directory / "run.json").write_text(json.dumps(run_record))
    (run_directory / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "update": 1,
                        "training_evaluation": {"mean_survival_frames": 100.0},
                        "validation": {"mean_survival_frames": 90.0},
                        "rollout_reward": 20.0,
                        "rollout_neutral_fraction": 0.2,
                        "entropy": 1.0,
                        "policy_loss": 0.1,
                        "value_loss": 0.2,
                        "approx_kl": 0.01,
                    }
                ),
                json.dumps(
                    {
                        "update": 2,
                        "training_evaluation": {"mean_survival_frames": 110.0},
                        "validation": {"mean_survival_frames": 95.0},
                        "rollout_reward": 22.0,
                        "rollout_neutral_fraction": 0.8,
                        "entropy": 0.9,
                        "policy_loss": 0.08,
                        "value_loss": 0.18,
                        "approx_kl": 0.02,
                    }
                ),
            ]
        )
        + "\n"
    )

    report = build_report(run_directory, manifest)

    comparison = report["comparison"]
    assert isinstance(comparison, dict)
    assert comparison["mean_train_minus_holdout"] == 70.0
    assert report["trend"]["training_mean_survival"]["gain"] == 10.0  # type: ignore[index]
    assert report["trend"]["rollout_neutral_fraction"]["last_value"] == 0.8  # type: ignore[index]
    assert (run_directory / "REPORT.md").is_file()
    assert (run_directory / "report.json").is_file()
    assert all(
        (run_directory / name).is_file()
        for name in (
            "survival_curves.png",
            "split_comparison.png",
            "per_seed_survival.png",
            "training_diagnostics.png",
        )
    )
