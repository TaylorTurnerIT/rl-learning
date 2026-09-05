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


def test_baseline_config_routes_pixel_observation_contract() -> None:
    manifest = SeedManifest.fresh_default()
    config = BaselineConfig(
        observation_mode="pixels",
        pixel_stack=4,
        native_lanes=8,
        rollout_steps=32,
    )

    ppo_config = config.ppo_config(manifest)

    assert ppo_config.observation_mode == "pixels"
    assert ppo_config.pixel_stack == 4
    assert ppo_config.backend == "native"


def test_baseline_config_routes_offscreen_board_contract() -> None:
    manifest = SeedManifest.fresh_default()
    config = BaselineConfig(
        observation_mode="board_full",
        native_lanes=8,
        rollout_steps=32,
    )

    ppo_config = config.ppo_config(manifest)

    assert ppo_config.observation_mode == "board_full"
    assert ppo_config.backend == "native"


def test_baseline_config_routes_board_spatial_pool() -> None:
    manifest = SeedManifest.fresh_default()
    config = BaselineConfig(
        observation_mode="board_full",
        board_spatial_pool="max",
        native_lanes=8,
        rollout_steps=32,
    )

    ppo_config = config.ppo_config(manifest)

    assert ppo_config.board_spatial_pool == "max"


def test_baseline_config_routes_coordinate_board_contract() -> None:
    manifest = SeedManifest.fresh_default()
    config = BaselineConfig(
        observation_mode="board_full_coords",
        board_spatial_pool="max",
        native_lanes=8,
        rollout_steps=32,
    )

    ppo_config = config.ppo_config(manifest)

    assert ppo_config.observation_mode == "board_full_coords"
    assert ppo_config.board_spatial_pool == "max"


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
    assert stored["target"]["target_survival_frames"] == 800
    assert stored["target"]["target_decision_steps"] == 200
    assert stored["target"]["reached"] is False
    assert result["report"] == {"run_directory": str(run_directory)}


def test_run_baseline_routes_actor_only_bc_warm_start(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = SeedManifest.fresh_default()
    manifest_path = tmp_path / "manifest.json"
    run_directory = tmp_path / "run"
    checkpoint = tmp_path / "bc-checkpoint.pt"
    save_manifest(manifest_path, manifest)
    checkpoint.write_bytes(b"test checkpoint")
    calls: dict[str, object] = {}
    marker = object()

    def fake_load(path, loaded_manifest):
        assert path == checkpoint
        assert loaded_manifest == manifest
        return {"features.projection.0.weight": marker}

    def fake_train(config, directory, **kwargs):
        calls["kwargs"] = kwargs
        directory.mkdir()
        return {"updates_completed": config.updates}

    monkeypatch.setattr("dodge.ng.train.load_bc_actor_state", fake_load)
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
            initial_bc_checkpoint=checkpoint,
        )
    )

    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["initial_actor_state"] == {"features.projection.0.weight": marker}
    assert kwargs["initialization"]["kind"] == "board_behavior_cloning_actor"  # type: ignore[index]
    stored = json.loads((run_directory / "ng-run.json").read_text())
    assert stored["kind"] == "dodge_ng_bc_to_ppo_run"
    assert stored["initialization"]["checkpoint"] == str(checkpoint)
    assert result["report"] == {"run_directory": str(run_directory)}
