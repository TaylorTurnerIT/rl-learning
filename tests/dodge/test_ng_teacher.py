from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dodge.ng.manifest import SeedManifest
from dodge.ng.teacher import (
    ACTION_COUNT,
    BOARD_SHAPE,
    CounterfactualCache,
    TeacherConfig,
    TeacherDataset,
    collect_teacher_dataset,
    load_teacher_dataset,
    save_teacher_dataset,
)

pytest.importorskip("dodge_native")


def _manifest() -> SeedManifest:
    return SeedManifest.fresh_default(
        seed_start=30_200,
        seed_count=10,
        split_seed=17,
    )


def _dataset(manifest: SeedManifest, seeds: np.ndarray | None = None) -> TeacherDataset:
    count = 4
    scores = np.zeros((count, ACTION_COUNT), dtype=np.float32)
    scores[np.arange(count), [1, 2, 3, 4]] = 4.0
    return TeacherDataset(
        boards=np.zeros((count, *BOARD_SHAPE), dtype=np.float32),
        actions=np.argmax(scores, axis=1).astype(np.int64),
        scores=scores,
        margins=np.full(count, 4.0, dtype=np.float32),
        seeds=np.asarray(
            manifest.training_seeds[:count] if seeds is None else seeds,
            dtype=np.uint32,
        ),
        frames=np.arange(count, dtype=np.uint32),
        state_hashes=np.arange(count, dtype=np.uint64),
        pixel_hashes=np.arange(count, dtype=np.uint64) + 100,
        metadata={
            "schema_version": 1,
            "data_version": 1,
            "kind": "dodge_ng_teacher_dataset",
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": manifest.sha256,
            "seed_scope": "training_only",
            "training_seeds": list(manifest.training_seeds),
            "holdout_examples": 0,
            "legacy_inputs": "none",
            "board_shape": list(BOARD_SHAPE),
            "actions": [
                "neutral",
                "left",
                "right",
                "up",
                "down",
                "up_left",
                "up_right",
                "down_left",
                "down_right",
            ],
        },
    )


def test_teacher_dataset_round_trips_with_hash_and_manifest_validation(
    tmp_path: Path,
) -> None:
    output = tmp_path
    manifest = _manifest()
    dataset = _dataset(manifest)
    save_teacher_dataset(dataset, output)

    loaded = load_teacher_dataset(output / "teacher-data.npz", manifest)
    np.testing.assert_array_equal(loaded.boards, dataset.boards)
    np.testing.assert_array_equal(loaded.actions, dataset.actions)
    np.testing.assert_array_equal(loaded.scores, dataset.scores)
    assert loaded.decisive_count == 4


def test_v24_teacher_save_recomputes_metadata_counters(tmp_path: Path) -> None:
    manifest = _manifest()
    dataset = _dataset(manifest)
    dataset.metadata.update(
        {"examples": 0, "decisive_examples": 0, "action_counts": [0] * ACTION_COUNT}
    )

    save_teacher_dataset(dataset, tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text())

    assert metadata["examples"] == dataset.count
    assert metadata["decisive_examples"] == dataset.decisive_count
    assert metadata["action_counts"] == [0, 1, 1, 1, 1, 0, 0, 0, 0]


def test_v23_teacher_dataset_rejects_holdout_seed(tmp_path: Path) -> None:
    output = tmp_path
    manifest = _manifest()
    seeds = np.asarray(
        [manifest.holdout_seeds[0], *manifest.training_seeds[1:4]], dtype=np.uint32
    )
    save_teacher_dataset(_dataset(manifest, seeds), output)

    with pytest.raises(ValueError, match="holdout seed"):
        load_teacher_dataset(output / "teacher-data.npz", manifest)


def test_fresh_teacher_collection_is_training_only_and_deterministic(
    tmp_path: Path,
) -> None:
    output = tmp_path
    manifest = _manifest()
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_json()))
    config = TeacherConfig(
        manifest_path=manifest_path,
        output_directory=output / "teacher",
        states_per_seed=2,
        lookahead_steps=2,
        native_lanes=2,
        max_collector_steps=500,
        collector_seed=99,
    )
    dataset = collect_teacher_dataset(config)
    assert dataset.count == len(manifest.training_seeds) * 2
    assert dataset.metadata["manifest_sha256"] == manifest.sha256
    assert dataset.metadata["legacy_inputs"] == "none"
    assert set(dataset.seeds.tolist()) <= set(manifest.training_seeds)
    assert not set(dataset.seeds.tolist()) & set(manifest.holdout_seeds)
    assert dataset.scores.shape == (dataset.count, ACTION_COUNT)
    assert np.isfinite(dataset.scores).all()
    assert np.isfinite(dataset.margins).all()
    assert (output / "teacher" / "teacher-data.npz").is_file()
    assert (output / "teacher" / "metadata.json").is_file()


def test_counterfactual_cache_reuses_exact_duplicate_snapshots() -> None:
    from dodge.native.batch import NativeBatchEnvironment

    with NativeBatchEnvironment(
        step_frames=4, full_state=True, board=True
    ) as environment:
        result = environment.reset_batch([30_200])
        snapshot = result.snapshot_bytes[0]
        assert snapshot is not None
        cache = CounterfactualCache()
        first = cache.score(environment, [snapshot, snapshot], lookahead_steps=2)
        second = cache.score(environment, [snapshot], lookahead_steps=2)
        np.testing.assert_array_equal(first[0], second[0])
        assert cache.to_json() == {"entries": 1, "hits": 1, "misses": 1}
