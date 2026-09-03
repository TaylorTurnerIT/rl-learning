from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import NativeBatchEnvironment
from dodge.ng.manifest import SeedManifest, save_manifest
from dodge.ng.relevance import RelevanceConfig, build_decision_relevance_audit

pytest.importorskip("dodge_native")


def _manifest() -> SeedManifest:
    return SeedManifest.fresh_default(
        seed_start=30_200,
        seed_count=10,
        split_seed=17,
    )


def test_counterfactual_scores_are_deterministic_and_nonmutating() -> None:
    with NativeBatchEnvironment(
        step_frames=4,
        execution="serial",
        full_state=True,
        pixels=True,
        board=False,
    ) as environment:
        result = environment.reset_batch([30_200])
        snapshot = result.snapshot_bytes[0]
        assert snapshot is not None
        before_frames = result.frames.copy()
        before_hashes = result.state_hashes.copy()
        before_pixels = result.pixel_hashes.copy()
        first = environment.score_actions([snapshot], lookahead_steps=8)
        second = environment.score_actions([snapshot], lookahead_steps=8)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(environment.last_result.frames, before_frames)
        np.testing.assert_array_equal(
            environment.last_result.state_hashes, before_hashes
        )
        np.testing.assert_array_equal(
            environment.last_result.pixel_hashes, before_pixels
        )
        assert environment.last_result.snapshot_bytes[0] == snapshot


def test_relevance_audit_routes_training_partition_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest_path, manifest)
    output_directory = tmp_path / "relevance"
    diagnostic = build_decision_relevance_audit(
        RelevanceConfig(
            manifest_path=manifest_path,
            output_directory=output_directory,
            action="neutral",
            lookahead_steps=(1, 2),
            gate_lookahead_steps=1,
            sample_every=2,
            max_samples_per_seed=2,
            max_episode_steps=24,
            native_lanes=2,
            native_execution="serial",
            visual_seeds=(manifest.training_seeds[0],),
        )
    )

    assert diagnostic["seeds"] == list(manifest.training_seeds)
    assert diagnostic["seed_scope"] == "training_only"
    assert diagnostic["source_state_nonmutation_verified"] is True
    assert [row["lookahead_steps"] for row in diagnostic["horizons"]] == [1, 2]
    assert (output_directory / "relevance.json").is_file()
    assert (output_directory / "RELEVANCE.md").is_file()
    assert (output_directory / "relevance_horizons.png").is_file()
    assert (output_directory / "relevance_timeline.png").is_file()
    assert (output_directory / "visuals").is_dir()
    stored = json.loads((output_directory / "relevance.json").read_text())
    assert stored["manifest_sha256"] == manifest.sha256
    assert stored["policy"]["label"] == "fixed:neutral"
    assert set(stored["seeds"]).isdisjoint(manifest.holdout_seeds)


def test_holdout_relevance_report_cannot_be_selection_gate(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest_path, manifest)
    diagnostic = build_decision_relevance_audit(
        RelevanceConfig(
            manifest_path=manifest_path,
            output_directory=tmp_path / "holdout",
            action=ACTION_CHOICES[0],
            partition="holdout",
            lookahead_steps=(1,),
            gate_lookahead_steps=1,
            max_samples_per_seed=1,
            max_episode_steps=8,
            native_lanes=2,
            native_execution="serial",
        )
    )

    gate = diagnostic["gate"]
    assert diagnostic["seeds"] == list(manifest.holdout_seeds)
    assert diagnostic["seed_scope"] == "holdout_report_only"
    assert gate["passed"] is False
    assert gate["selection_eligible"] is False
    assert gate["reason"] == "holdout-only report cannot set selection gate"
