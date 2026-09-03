from __future__ import annotations

import json
from pathlib import Path

import pytest

from dodge.ng.manifest import (
    DEFAULT_SEED_COUNT,
    DEFAULT_SEED_START,
    LEGACY_SEED_MAX,
    SeedManifest,
    load_manifest,
    main,
    save_manifest,
)


def test_fresh_default_is_deterministic_and_exactly_seventy_thirty() -> None:
    first = SeedManifest.fresh_default()
    second = SeedManifest.fresh_default()

    assert first == second
    assert first.sample_count == DEFAULT_SEED_COUNT
    assert len(first.training_seeds) == 70
    assert len(first.holdout_seeds) == 30
    assert set(first.training_seeds).isdisjoint(first.holdout_seeds)
    assert set(first.training_seeds) | set(first.holdout_seeds) == set(
        first.sample_space
    )
    assert all(seed > LEGACY_SEED_MAX for seed in first.sample_space)
    assert min(first.sample_space) == DEFAULT_SEED_START


def test_manifest_round_trip_preserves_hash(tmp_path: Path) -> None:
    path = tmp_path / "ng-v1.json"
    manifest = SeedManifest.fresh_default()

    save_manifest(path, manifest)
    loaded = load_manifest(path)

    assert loaded == manifest
    assert json.loads(path.read_text())["manifest_sha256"] == manifest.sha256


def test_manifest_rejects_hash_tampering(tmp_path: Path) -> None:
    path = tmp_path / "ng-v1.json"
    save_manifest(path, SeedManifest.fresh_default())
    value = json.loads(path.read_text())
    value["split_seed"] += 1
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="hash"):
        load_manifest(path)


def test_manifest_rejects_legacy_seed() -> None:
    manifest = SeedManifest(
        manifest_id="dodge-ng-v1",
        schema_version=1,
        split_seed=1,
        sample_space=(
            LEGACY_SEED_MAX,
            30_011,
            30_012,
            30_013,
            30_014,
            30_015,
            30_016,
            30_017,
            30_018,
            30_019,
        ),
        training_seeds=(
            LEGACY_SEED_MAX,
            30_011,
            30_012,
            30_013,
            30_014,
            30_015,
            30_016,
        ),
        holdout_seeds=(30_017, 30_018, 30_019),
    )

    with pytest.raises(ValueError, match="legacy"):
        manifest.validate()


def test_manifest_cli_freezes_requested_file(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    assert main(["--output", str(path), "--seed-start", "30100"]) == 0
    assert load_manifest(path).sample_space
    assert main(["--output", str(path)]) == 1
