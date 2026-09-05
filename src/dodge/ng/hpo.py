"""Training-only hyperparameter search for the NG waypoint DQN."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import optuna
import torch
from optuna.trial import Trial, TrialState

from dodge.control import PROJECT_ROOT, ControlRuntimeError
from dodge.ng.dqn import (
    DQNConfig,
    DuelingWaypointDQN,
    evaluate_waypoint_dqn,
    train_waypoint_dqn,
)
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, load_manifest

DEFAULT_RUN_DIRECTORY = PROJECT_ROOT / "history" / "dodge" / "ng" / "waypoint-hpo"
DEFAULT_STUDY_NAME: Final[str] = "dodge-ng-waypoint-dqn"
DEFAULT_STUDY_SEED: Final[int] = 2_026_0904
DEFAULT_LEARNER_SEED: Final[int] = 2_026_0903
DEFAULT_BUDGETS: tuple[int, ...] = (20_000, 60_000, 120_000)
INNER_VALIDATION_COUNT: Final[int] = 10


@dataclass(frozen=True, slots=True)
class HPOConfig:
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    run_directory: Path = DEFAULT_RUN_DIRECTORY
    study_name: str = DEFAULT_STUDY_NAME
    trials: int = 8
    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    study_seed: int = DEFAULT_STUDY_SEED
    learner_seed: int = DEFAULT_LEARNER_SEED
    native_lanes: int = 32
    device: str = "cpu"

    def validate(self) -> None:
        if self.trials < 1:
            raise ValueError("HPO trial count must be positive")
        if not self.budgets or any(budget < 1 for budget in self.budgets):
            raise ValueError("HPO budgets must be positive")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("HPO budgets must be sorted and unique")
        if self.native_lanes < 1:
            raise ValueError("HPO lane count must be positive")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("HPO device must be cpu, cuda, or auto")


def _baseline_parameters() -> dict[str, object]:
    """Return a fixed current-style control for the first study trial."""
    return {
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "target_update_interval": 1_000,
        "batch_size": 256,
        "n_step": 3,
        "epsilon_decay_steps": 60_000,
        "epsilon_final": 0.05,
    }


def _suggest_parameters(trial: Trial) -> dict[str, object]:
    """Suggest only the first DDQN stability family."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 3e-4, log=True),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 1e-5, 1e-4, 1e-3]
        ),
        "target_update_interval": trial.suggest_categorical(
            "target_update_interval", [250, 500, 1_000, 2_000, 5_000]
        ),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_step": trial.suggest_categorical("n_step", [1, 3, 5]),
        "epsilon_decay_steps": trial.suggest_categorical(
            "epsilon_decay_steps", [25_000, 50_000, 60_000, 100_000, 200_000]
        ),
        "epsilon_final": trial.suggest_categorical("epsilon_final", [0.05, 0.1, 0.2]),
    }


def _trial_config(
    parameters: dict[str, object],
    budget: int,
    learner_seed: int,
    native_lanes: int,
    device: str,
) -> DQNConfig:
    return DQNConfig(
        total_steps=budget,
        batch_size=int(parameters["batch_size"]),
        replay_capacity=100_000,
        learning_rate=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
        gamma=0.99,
        n_step=int(parameters["n_step"]),
        warmup_steps=2_000,
        train_frequency=1,
        target_update_interval=int(parameters["target_update_interval"]),
        hidden_size=256,
        grid_spacing=32,
        hold_decisions=8,
        step_frames=4,
        max_episode_steps=2_000,
        native_lanes=native_lanes,
        native_execution="parallel",
        reset_mode="native-startup",
        training_lives=1,
        life_loss_penalty=0.0,
        epsilon_decay_steps=int(parameters["epsilon_decay_steps"]),
        epsilon_final=float(parameters["epsilon_final"]),
        checkpoint_every=10_000,
        eval_every=5_000,
        seed=learner_seed,
        device=device,
    )


def _score_run(run: dict[str, object]) -> tuple[float, dict[str, float]]:
    validation = run.get("final_validation")
    if not isinstance(validation, dict):
        raise ControlRuntimeError("HPO trial has no training-side validation")
    summary = validation.get("summary")
    if not isinstance(summary, dict):
        raise ControlRuntimeError("HPO trial validation summary is invalid")
    try:
        mean = float(summary["mean_survival_frames"])
        median = float(summary["median_survival_frames"])
        p10 = float(summary["p10_survival_frames"])
        worst = float(summary["worst_survival_frames"])
    except (KeyError, TypeError, ValueError) as error:
        raise ControlRuntimeError("HPO trial validation metrics are invalid") from error
    score = mean + 0.25 * p10
    return score, {
        "mean_survival_frames": mean,
        "median_survival_frames": median,
        "p10_survival_frames": p10,
        "worst_survival_frames": worst,
        "score": score,
    }


def _load_best_model(path: Path, hidden_size: int) -> DuelingWaypointDQN:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ControlRuntimeError(f"could not load HPO checkpoint: {error}") from error
    if not isinstance(payload, dict):
        raise ControlRuntimeError("HPO checkpoint must contain an object")
    state = payload.get("best_model_state") or payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ControlRuntimeError("HPO checkpoint has no model state")
    model = DuelingWaypointDQN(hidden_size=hidden_size)
    try:
        model.load_state_dict(state)
    except (TypeError, RuntimeError) as error:
        raise ControlRuntimeError(f"HPO model state is invalid: {error}") from error
    model.eval()
    return model


def _trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, object]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": dict(trial.params),
        "user_attrs": dict(trial.user_attrs),
        "intermediate_values": {
            str(step): value for step, value in trial.intermediate_values.items()
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_report(path: Path, result: dict[str, object]) -> None:
    selected = result["selected_trial"]
    assert isinstance(selected, dict)
    rows = [
        "| Split | Mean | Median | P10 | Worst | Complete |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, value in (
        ("Training", result["training_evaluation"]),
        ("Holdout", result["holdout_evaluation"]),
    ):
        assert isinstance(value, dict)
        summary = value["summary"]
        assert isinstance(summary, dict)
        rows.append(
            f"| {label} | {float(summary['mean_survival_frames']):.1f} | "
            f"{float(summary['median_survival_frames']):.1f} | "
            f"{float(summary['p10_survival_frames']):.1f} | "
            f"{float(summary['worst_survival_frames']):.1f} | "
            f"{float(summary['horizon_completion_fraction']):.1%} |"
        )
    content = [
        "# Dodge NG waypoint DQN HPO",
        "",
        f"Manifest SHA-256: `{result['manifest_sha256']}`  ",
        f"Study: `{result['study_name']}`  ",
        f"Trials: `{result['trial_count']}`; budgets: `{result['budgets']}`  ",
        "",
        "## Selected trial",
        "",
        f"Trial `{selected['number']}`; score `{float(selected['value']):.2f}`.",
        "",
        "```json",
        json.dumps(selected["params"], indent=2, sort_keys=True),
        "```",
        "",
        *rows,
        "",
        "Trial objectives used inner training seeds only. The locked holdout was "
        "evaluated once after study selection.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def run_hpo(config: HPOConfig) -> dict[str, object]:
    config.validate()
    manifest = load_manifest(config.manifest_path)
    manifest.validate()
    if len(manifest.training_seeds) < config.native_lanes:
        raise ValueError("HPO lane count exceeds training seed count")
    config.run_directory.mkdir(parents=True, exist_ok=True)
    database = (config.run_directory / "study.db").resolve()
    storage = f"sqlite:///{database}"
    sampler = optuna.samplers.TPESampler(seed=config.study_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=4,
        n_warmup_steps=1,
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=config.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(_baseline_parameters())

    def objective(trial: Trial) -> float:
        parameters = _suggest_parameters(trial)
        trial_directory = config.run_directory / f"trial-{trial.number:04d}"
        scores: list[dict[str, float]] = []
        for budget in config.budgets:
            trial_config = _trial_config(
                parameters,
                budget,
                config.learner_seed,
                config.native_lanes,
                config.device,
            )
            run = train_waypoint_dqn(
                trial_config,
                trial_directory,
                manifest,
                resume=(trial_directory / "checkpoint-latest.pt").is_file(),
                evaluate_holdout=False,
                evaluate_training=False,
            )
            score, metrics = _score_run(run)
            scores.append({"budget": float(budget), **metrics})
            trial.report(score, step=budget)
            if trial.should_prune():
                trial.set_user_attr("rung_metrics", scores)
                trial.set_user_attr("run_directory", str(trial_directory))
                raise optuna.TrialPruned()
        trial.set_user_attr("rung_metrics", scores)
        trial.set_user_attr("run_directory", str(trial_directory))
        trial.set_user_attr("manifest_sha256", manifest.sha256)
        return scores[-1]["score"]

    finished = sum(
        trial.state in {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
        for trial in study.trials
    )
    remaining = max(config.trials - finished, 0)
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        raise ControlRuntimeError("HPO study has no completed trial")
    best = study.best_trial
    trial_directory = Path(str(best.user_attrs["run_directory"]))
    checkpoint = trial_directory / "checkpoint-best.pt"
    model = _load_best_model(checkpoint, hidden_size=256)
    evaluation_config = _trial_config(
        dict(best.params),
        config.budgets[-1],
        config.learner_seed,
        config.native_lanes,
        config.device,
    )
    training_evaluation = evaluate_waypoint_dqn(
        model,
        manifest.training_seeds,
        evaluation_config,
    )
    holdout_evaluation = evaluate_waypoint_dqn(
        model,
        manifest.holdout_seeds,
        evaluation_config,
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "kind": "dodge_ng_waypoint_dqn_hpo",
        "host": platform.node(),
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.sha256,
        "study_name": config.study_name,
        "study_database": str(database),
        "trial_count": len(study.trials),
        "budgets": list(config.budgets),
        "training_seeds": list(manifest.training_seeds),
        "inner_validation_seeds": list(
            manifest.training_seeds[:INNER_VALIDATION_COUNT]
        ),
        "holdout_seeds": list(manifest.holdout_seeds),
        "holdout_evaluated_once_after_selection": True,
        "selected_trial": _trial_record(best),
        "selected_trial_directory": str(trial_directory),
        "selected_checkpoint": str(checkpoint),
        "training_evaluation": training_evaluation,
        "holdout_evaluation": holdout_evaluation,
        "target": {
            "relevance_gate_frames": 800,
            "training_mean_reached": float(
                training_evaluation["summary"]["mean_survival_frames"]
            )
            >= 800,
        },
        "trials": [_trial_record(trial) for trial in study.trials],
    }
    _write_json(config.run_directory / "hpo.json", result)
    _write_report(config.run_directory / "REPORT.md", result)
    return result


def _parse_budgets(value: str) -> tuple[int, ...]:
    try:
        budgets = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "budgets must be comma-separated integers"
        ) from error
    if not budgets:
        raise argparse.ArgumentTypeError("at least one budget is required")
    return budgets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-hpo")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIRECTORY)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument(
        "--budgets",
        type=_parse_budgets,
        default=DEFAULT_BUDGETS,
        help="comma-separated macro-step promotion budgets",
    )
    parser.add_argument("--study-seed", type=int, default=DEFAULT_STUDY_SEED)
    parser.add_argument("--learner-seed", type=int, default=DEFAULT_LEARNER_SEED)
    parser.add_argument("--native-lanes", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    arguments = parser.parse_args(argv)
    config = HPOConfig(
        manifest_path=arguments.manifest,
        run_directory=arguments.run_dir,
        study_name=arguments.study_name,
        trials=arguments.trials,
        budgets=arguments.budgets,
        study_seed=arguments.study_seed,
        learner_seed=arguments.learner_seed,
        native_lanes=arguments.native_lanes,
        device=arguments.device,
    )
    try:
        started = time.monotonic()
        result = run_hpo(config)
        result["wall_seconds"] = time.monotonic() - started
        _write_json(config.run_directory / "hpo.json", result)
    except (ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-ng-hpo: {error}")
        return 1
    print(
        json.dumps(
            {
                "run_directory": str(config.run_directory),
                "study_name": config.study_name,
                "trial_count": result["trial_count"],
                "selected_trial": result["selected_trial"]["number"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
