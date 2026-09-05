from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch

from dodge.ng.dqn import DuelingWaypointDQN
from dodge.ng.hpo import (
    HPOConfig,
    _parse_budgets,
    _score_run,
    _trial_config,
    run_hpo,
)
from dodge.ng.manifest import SeedManifest, save_manifest


def test_hpo_config_requires_sorted_unique_budgets(tmp_path: Path) -> None:
    config = HPOConfig(
        run_directory=tmp_path,
        budgets=(20, 10),
    )

    with pytest.raises(ValueError, match="sorted"):
        config.validate()


def test_hpo_budget_parser_and_score() -> None:
    assert _parse_budgets("20,60,120") == (20, 60, 120)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_budgets("20,nope")

    score, metrics = _score_run(
        {
            "final_validation": {
                "summary": {
                    "mean_survival_frames": 400,
                    "median_survival_frames": 390,
                    "p10_survival_frames": 200,
                    "worst_survival_frames": 150,
                }
            }
        }
    )

    assert score == pytest.approx(450)
    assert metrics["score"] == pytest.approx(450)


def test_trial_config_maps_search_parameters() -> None:
    config = _trial_config(
        {
            "learning_rate": 3e-5,
            "weight_decay": 1e-4,
            "target_update_interval": 500,
            "batch_size": 64,
            "n_step": 5,
            "epsilon_decay_steps": 100_000,
            "epsilon_final": 0.2,
        },
        budget=120,
        learner_seed=7,
        native_lanes=2,
        device="cpu",
    )

    assert config.total_steps == 120
    assert config.learning_rate == pytest.approx(3e-5)
    assert config.weight_decay == pytest.approx(1e-4)
    assert config.target_update_interval == 500
    assert config.batch_size == 64
    assert config.n_step == 5
    assert config.epsilon_decay_steps == 100_000
    assert config.epsilon_final == pytest.approx(0.2)


def test_hpo_keeps_holdout_out_of_trial_objective(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = SeedManifest.fresh_default(
        seed_start=30_200,
        seed_count=10,
        split_seed=17,
    )
    manifest_path = tmp_path / "manifest.json"
    run_directory = tmp_path / "hpo"
    save_manifest(manifest_path, manifest)
    train_calls: list[dict[str, object]] = []
    evaluation_seeds: list[tuple[int, ...]] = []

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
            {"best_model_state": model.state_dict()}, directory / "checkpoint-best.pt"
        )
        return {
            "final_validation": {
                "summary": {
                    "mean_survival_frames": 400,
                    "median_survival_frames": 390,
                    "p10_survival_frames": 200,
                    "worst_survival_frames": 150,
                }
            }
        }

    def fake_evaluate(model, seeds, config):
        del model, config
        evaluation_seeds.append(tuple(seeds))
        return {
            "seeds": list(seeds),
            "summary": {
                "mean_survival_frames": 500,
                "median_survival_frames": 500,
                "p10_survival_frames": 500,
                "worst_survival_frames": 500,
                "horizon_completion_fraction": 0.0,
            },
        }

    monkeypatch.setattr("dodge.ng.hpo.train_waypoint_dqn", fake_train)
    monkeypatch.setattr("dodge.ng.hpo.evaluate_waypoint_dqn", fake_evaluate)
    result = run_hpo(
        HPOConfig(
            manifest_path=manifest_path,
            run_directory=run_directory,
            study_name="test-hpo-holdout-boundary",
            trials=1,
            budgets=(20,),
            native_lanes=2,
        )
    )

    assert train_calls
    assert all(call["evaluate_holdout"] is False for call in train_calls)
    assert all(call["evaluate_training"] is False for call in train_calls)
    assert evaluation_seeds == [manifest.training_seeds, manifest.holdout_seeds]
    stored = json.loads((run_directory / "hpo.json").read_text())
    assert stored["holdout_evaluated_once_after_selection"] is True
    assert result["holdout_seeds"] == list(manifest.holdout_seeds)
