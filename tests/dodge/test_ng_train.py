from __future__ import annotations

import json
from pathlib import Path

from dodge.ng.manifest import SeedManifest, save_manifest
from dodge.ng.train import BaselineConfig, run_baseline


def test_baseline_config_routes_only_manifest_training_seeds(tmp_path: Path) -> None:
    manifest = SeedManifest.fresh_default()
    config = BaselineConfig(
        manifest_path=tmp_path / "manifest.json",
        run_directory=tmp_path / "run",
        updates=3,
        rollout_steps=32,
        native_lanes=8,
    )

    ppo_config = config.ppo_config(manifest)

    assert ppo_config.backend == "native"
    assert ppo_config.observation_mode == "board"
    assert ppo_config.training_seeds == manifest.training_seeds
    assert ppo_config.training_seed_manifest == manifest.sha256
    assert config.to_json()["manifest_path"] == str(config.manifest_path)


def test_run_baseline_passes_locked_split_to_ppo(monkeypatch, tmp_path: Path) -> None:
    manifest = SeedManifest.fresh_default()
    manifest_path = tmp_path / "manifest.json"
    run_directory = tmp_path / "run"
    save_manifest(manifest_path, manifest)
    calls: dict[str, object] = {}

    def fake_train(config, directory, **kwargs):
        calls["config"] = config
        calls["directory"] = directory
        calls["kwargs"] = kwargs
        directory.mkdir()
        return {"updates_completed": config.updates}

    monkeypatch.setattr("dodge.ng.train.train_ppo", fake_train)
    monkeypatch.setattr(
        "dodge.ng.train.build_report",
        lambda directory, loaded: {"run_directory": str(directory)},
    )

    result = run_baseline(
        BaselineConfig(
            manifest_path=manifest_path,
            run_directory=run_directory,
            updates=3,
            rollout_steps=32,
            native_lanes=8,
        )
    )

    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["validation_seeds"] == manifest.training_seeds[:10]
    assert kwargs["training_evaluation_seeds"] == manifest.training_seeds
    assert kwargs["evaluation_seeds"] == manifest.holdout_seeds
    stored = json.loads((run_directory / "ng-run.json").read_text())
    assert stored["legacy_inputs"] == "none"
    assert result["report"] == {"run_directory": str(run_directory)}
