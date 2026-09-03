from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dodge.control import ControlRuntimeError
from dodge.dataset import ACTION_CHOICES
from dodge.ng.bc import BCConfig, load_bc_actor_state, run_behavior_cloning
from dodge.ng.manifest import SeedManifest, save_manifest
from dodge.ng.teacher import (
    BOARD_SHAPE,
    TeacherDataset,
    load_teacher_dataset,
    save_teacher_dataset,
)
from dodge.rl.ppo import DodgeActorCriticCNN, EvaluationResult


def _manifest() -> SeedManifest:
    return SeedManifest.fresh_default(seed_start=30_200, seed_count=20, split_seed=17)


def _dataset(manifest: SeedManifest) -> TeacherDataset:
    seeds = np.repeat(np.asarray(manifest.training_seeds, dtype=np.uint32), 2)
    count = len(seeds)
    scores = np.zeros((count, len(ACTION_CHOICES)), dtype=np.float32)
    actions = np.arange(count, dtype=np.int64) % len(ACTION_CHOICES)
    scores[np.arange(count), actions] = 4.0
    scores[np.arange(count), (actions + 1) % len(ACTION_CHOICES)] = 0.0
    return TeacherDataset(
        boards=np.random.default_rng(7).random(
            (count, *BOARD_SHAPE), dtype=np.float32
        ),
        actions=actions,
        scores=scores,
        margins=np.full(count, 4.0, dtype=np.float32),
        seeds=seeds,
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
            "actions": list(ACTION_CHOICES),
        },
    )


def test_bc_selects_training_side_and_writes_compatible_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    manifest_path = tmp_path / "manifest.json"
    teacher_directory = tmp_path / "teacher"
    output_directory = tmp_path / "bc"
    save_manifest(manifest_path, manifest)
    save_teacher_dataset(_dataset(manifest), teacher_directory)

    def fake_evaluate(model, config, seeds, **kwargs):
        del model, config, kwargs
        return EvaluationResult(
            tuple(seeds),
            tuple(200 for _ in seeds),
            tuple(True for _ in seeds),
        )

    monkeypatch.setattr("dodge.ng.bc.evaluate_policy", fake_evaluate)
    result = run_behavior_cloning(
        BCConfig(
            manifest_path=manifest_path,
            teacher_data_path=teacher_directory / "teacher-data.npz",
            output_directory=output_directory,
            epochs=1,
            batch_size=8,
            eval_every=1,
            device="cpu",
            native_lanes=4,
        )
    )

    assert result["fit_seeds"] == list(manifest.training_seeds[10:])
    assert result["inner_validation_seeds"] == list(manifest.training_seeds[:10])
    assert set(result["fit_seeds"]).isdisjoint(result["inner_validation_seeds"])
    assert set(result["holdout_seeds"]).isdisjoint(result["fit_seeds"])
    assert (output_directory / "checkpoint-best.pt").is_file()
    assert (output_directory / "checkpoint-latest.pt").is_file()
    assert (output_directory / "metrics.jsonl").is_file()
    assert (output_directory / "report.json").is_file()

    loaded_manifest = json.loads(manifest_path.read_text())
    assert loaded_manifest["manifest_sha256"] == manifest.sha256
    loaded_teacher = load_teacher_dataset(
        teacher_directory / "teacher-data.npz", manifest
    )
    actor_state = load_bc_actor_state(
        output_directory / "checkpoint-best.pt",
        manifest,
        teacher_data=loaded_teacher,
    )
    assert "features.convolution.0.weight" in actor_state
    assert "policy_head.weight" in actor_state


def test_bc_checkpoint_rejects_a_different_manifest(tmp_path: Path) -> None:
    manifest = _manifest()
    other_manifest = SeedManifest.fresh_default(
        seed_start=30_300, seed_count=20, split_seed=17
    )
    source = DodgeActorCriticCNN()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "version": 1,
            "model_type": "DodgeActorCriticCNN",
            "board_shape": list(BOARD_SHAPE),
            "actions": list(ACTION_CHOICES),
            "manifest_sha256": manifest.sha256,
            "teacher_data_sha256": "teacher-hash",
            "model_state_dict": source.state_dict(),
        },
        checkpoint,
    )

    with pytest.raises(ControlRuntimeError, match="incompatible metadata"):
        load_bc_actor_state(checkpoint, other_manifest)
