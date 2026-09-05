from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from dodge.ng.confirm import (
    ConfirmationConfig,
    _parse_csv_ints,
    run_confirmation,
)
from dodge.ng.dqn import DuelingWaypointDQN
from dodge.ng.manifest import SeedManifest, save_manifest

pytest.importorskip("dodge_native")


def _parameters(learning_rate: float) -> dict[str, object]:
    return {
        "learning_rate": learning_rate,
        "weight_decay": 0.0,
        "target_update_interval": 500,
        "batch_size": 64,
        "n_step": 3,
        "epsilon_decay_steps": 25_000,
        "epsilon_final": 0.1,
    }


def _evaluation(seeds: tuple[int, ...], mean: int) -> dict[str, object]:
    survival = [mean] * len(seeds)
    terminated = [True] * len(seeds)
    return {
        "seeds": list(seeds),
        "survival_frames": survival,
        "terminated": terminated,
        "summary": {
            "mean_survival_frames": mean,
            "median_survival_frames": mean,
            "p10_survival_frames": mean,
            "worst_survival_frames": mean,
        },
    }


def test_confirmation_uses_training_scores_before_one_holdout_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = SeedManifest.fresh_default(
        seed_start=30_200,
        seed_count=10,
        split_seed=17,
    )
    manifest_path = tmp_path / "manifest.json"
    hpo_path = tmp_path / "hpo.json"
    save_manifest(manifest_path, manifest)
    hpo_path.write_text(
        json.dumps(
            {
                "manifest_sha256": manifest.sha256,
                "study_name": "test-study",
                "budgets": [20],
                "trials": [
                    {
                        "number": 0,
                        "state": "COMPLETE",
                        "params": _parameters(1e-4),
                    },
                    {
                        "number": 1,
                        "state": "COMPLETE",
                        "params": _parameters(2e-4),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    train_calls: list[dict[str, object]] = []
    evaluation_calls: list[tuple[int, ...]] = []
    holdout_calls: list[tuple[int, ...]] = []

    def fake_train(config, directory, loaded_manifest, **kwargs):
        train_calls.append(
            {
                "config": config,
                "directory": directory,
                "manifest": loaded_manifest,
                **kwargs,
            }
        )
        directory.mkdir(parents=True, exist_ok=True)
        model = DuelingWaypointDQN(hidden_size=config.hidden_size)
        torch.save(
            {"best_model_state": model.state_dict()},
            directory / "checkpoint-best.pt",
        )
        torch.save(
            {"best_model_state": model.state_dict()},
            directory / "checkpoint-latest.pt",
        )
        return {}

    def fake_evaluate(model, seeds, config):
        del model
        seed_tuple = tuple(seeds)
        if seed_tuple == manifest.holdout_seeds:
            holdout_calls.append(seed_tuple)
        else:
            evaluation_calls.append(seed_tuple)
        mean = 1_000 if config.learning_rate > 1.5e-4 else 700
        return _evaluation(seed_tuple, mean)

    monkeypatch.setattr("dodge.ng.confirm.train_waypoint_dqn", fake_train)
    monkeypatch.setattr("dodge.ng.confirm.evaluate_waypoint_dqn", fake_evaluate)
    result = run_confirmation(
        ConfirmationConfig(
            manifest_path=manifest_path,
            hpo_path=hpo_path,
            run_directory=tmp_path / "confirmation",
            candidate_trials=(1, 0),
            learner_seeds=(7, 8),
            native_lanes=2,
        )
    )

    assert len(train_calls) == 4
    assert all(call["evaluate_holdout"] is False for call in train_calls)
    assert all(call["evaluate_training"] is False for call in train_calls)
    assert evaluation_calls == [
        manifest.training_seeds,
        manifest.training_seeds,
        manifest.training_seeds,
        manifest.training_seeds,
    ]
    assert holdout_calls == [manifest.holdout_seeds]
    assert result["selected_candidate"]["trial"] == 1
    assert result["selected_learner"]["learner_seed"] == 7
    assert result["selection"]["holdout_used_for_selection"] is False
    assert (tmp_path / "confirmation" / "REPORT.md").is_file()


def test_confirmation_parser_requires_comma_separated_integers() -> None:
    assert _parse_csv_ints("5, 4,0") == (5, 4, 0)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_csv_ints("5,nope")


def test_confirmation_config_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        ConfirmationConfig(learner_seeds=(1, 1)).validate()
