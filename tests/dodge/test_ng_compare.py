from __future__ import annotations

import json
from pathlib import Path

import pytest

from dodge.ng.compare import compare_ppo_runs
from dodge.ng.manifest import SeedManifest, save_manifest
from dodge.rl.ppo import PPOConfig


def _evaluation(mean: float, p10: float) -> dict[str, object]:
    return {
        "mean_survival_frames": mean,
        "median_survival_frames": mean,
        "p10_survival_frames": p10,
        "worst_survival_frames": p10,
        "best_survival_frames": mean + 10,
        "horizon_completion_fraction": 0.0,
    }


def test_compare_writes_selected_checkpoint_comparison_and_efficiency_curve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = SeedManifest.fresh_default(seed_start=30_200, seed_count=100)
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest_path, manifest)
    config = PPOConfig(
        updates=200,
        rollout_steps=256,
        update_epochs=4,
        minibatch_size=64,
        backend="native",
        native_lanes=32,
        training_seeds=manifest.training_seeds,
        training_seed_manifest=manifest.sha256,
    ).to_json()

    def fake_load(run_directory: Path, loaded_manifest: SeedManifest):
        assert loaded_manifest == manifest
        warm = "warm" in run_directory.name
        inner = 394.0 if warm else 195.9
        return {
            "run_directory": str(run_directory),
            "run_kind": "dodge_ng_bc_to_ppo_run" if warm else "dodge_ng_baseline_run",
            "initialization": {"kind": "bc"} if warm else None,
            "checkpoint": "checkpoint-best.pt",
            "selected_update": 75 if warm else 25,
            "best_inner_validation": {"mean_survival_frames": inner},
            "selected": {
                "inner_validation": _evaluation(inner, inner - 5),
                "training": _evaluation(inner - 15, inner - 20),
                "holdout": _evaluation(inner - 34, inner - 45),
            },
            "latest": {"training": None, "holdout": None},
            "curve": (
                [{"update": 25.0, "value": 248.5}, {"update": 75.0, "value": inner}]
                if warm
                else [{"update": 25.0, "value": 195.9}]
            ),
            "global_step": 51_200,
            "config": config,
        }

    monkeypatch.setattr("dodge.ng.compare._load_selected_run", fake_load)
    output_directory = tmp_path / "comparison"
    result = compare_ppo_runs(
        tmp_path / "scratch",
        tmp_path / "warm",
        manifest_path,
        output_directory,
    )

    assert result["delta"]["inner_mean_survival_frames"] == pytest.approx(198.1)  # type: ignore[index]
    assert result["delta"]["holdout_mean_survival_frames"] == pytest.approx(198.1)  # type: ignore[index]
    assert (
        result["sample_efficiency"]["scratch"][  # type: ignore[index]
            "first_inner_at_least_250"
        ]
        is None
    )
    assert (
        result["sample_efficiency"]["bc_warm_start"][  # type: ignore[index]
            "first_inner_at_least_350"
        ]
        == 75
    )
    assert (output_directory / "comparison.json").is_file()
    assert (output_directory / "COMPARISON.md").is_file()
    assert (output_directory / "ppo_comparison.png").is_file()
    json.loads((output_directory / "comparison.json").read_text())
