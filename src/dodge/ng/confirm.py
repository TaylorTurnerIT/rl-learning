"""Independent learner-seed confirmation for waypoint DQN HPO candidates."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch

from dodge.control import PROJECT_ROOT, ControlRuntimeError
from dodge.ng.dqn import DuelingWaypointDQN, evaluate_waypoint_dqn, train_waypoint_dqn
from dodge.ng.hpo import _trial_config
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest

DEFAULT_HPO_PATH = (
    PROJECT_ROOT / "history" / "dodge" / "ng" / "waypoint-hpo-20260904-v1" / "hpo.json"
)
DEFAULT_RUN_DIRECTORY = (
    PROJECT_ROOT
    / "history"
    / "dodge"
    / "ng"
    / "waypoint-hpo-confirmation-20260904-v1"
)
DEFAULT_LEARNER_SEEDS: tuple[int, ...] = (2_026_0903, 2_026_0905, 2_026_0907)
DEFAULT_CANDIDATE_TRIALS: tuple[int, ...] = (5, 4, 0)
CONFIRMATION_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True, slots=True)
class ConfirmationConfig:
    """Configuration for a training-side candidate confirmation campaign."""

    manifest_path: Path = DEFAULT_MANIFEST_PATH
    hpo_path: Path = DEFAULT_HPO_PATH
    run_directory: Path = DEFAULT_RUN_DIRECTORY
    candidate_trials: tuple[int, ...] = DEFAULT_CANDIDATE_TRIALS
    learner_seeds: tuple[int, ...] = DEFAULT_LEARNER_SEEDS
    native_lanes: int = 32
    device: str = "cpu"

    def validate(self) -> None:
        if not self.candidate_trials:
            raise ValueError("confirmation requires at least one candidate trial")
        if len(set(self.candidate_trials)) != len(self.candidate_trials):
            raise ValueError("confirmation candidate trials must be unique")
        if not self.learner_seeds:
            raise ValueError("confirmation requires at least one learner seed")
        if len(set(self.learner_seeds)) != len(self.learner_seeds):
            raise ValueError("confirmation learner seeds must be unique")
        if self.native_lanes < 1:
            raise ValueError("confirmation lane count must be positive")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("confirmation device must be cpu, cuda, or auto")


def _load_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlRuntimeError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ControlRuntimeError(f"{label} must contain a JSON object: {path}")
    return value


def _candidate_records(
    hpo: Mapping[str, object], candidate_trials: Sequence[int]
) -> list[dict[str, object]]:
    raw_trials = hpo.get("trials")
    if not isinstance(raw_trials, list):
        raise ControlRuntimeError("HPO result has no trial records")
    by_number: dict[int, dict[str, object]] = {}
    for raw_trial in raw_trials:
        if not isinstance(raw_trial, dict):
            raise ControlRuntimeError("HPO trial record is not an object")
        number = raw_trial.get("number")
        if isinstance(number, bool) or not isinstance(number, int):
            raise ControlRuntimeError("HPO trial record has an invalid number")
        by_number[number] = raw_trial
    records: list[dict[str, object]] = []
    for number in candidate_trials:
        record = by_number.get(number)
        if record is None:
            raise ControlRuntimeError(f"HPO candidate trial does not exist: {number}")
        if record.get("state") != "COMPLETE":
            raise ControlRuntimeError(f"HPO candidate trial is not complete: {number}")
        parameters = record.get("params")
        if not isinstance(parameters, dict):
            raise ControlRuntimeError(
                f"HPO candidate trial has no parameters: {number}"
            )
        records.append(
            {
                "number": number,
                "params": dict(parameters),
            }
        )
    return records


def _load_best_model(path: Path, hidden_size: int) -> DuelingWaypointDQN:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ControlRuntimeError(
            f"could not load confirmation checkpoint: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ControlRuntimeError("confirmation checkpoint must contain an object")
    state = payload.get("best_model_state") or payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ControlRuntimeError("confirmation checkpoint has no model state")
    model = DuelingWaypointDQN(hidden_size=hidden_size)
    try:
        model.load_state_dict(state)
    except (TypeError, RuntimeError) as error:
        raise ControlRuntimeError(
            f"confirmation model state is invalid: {error}"
        ) from error
    model.eval()
    return model


def _evaluation_score(evaluation: Mapping[str, object]) -> dict[str, float]:
    summary = evaluation.get("summary")
    if not isinstance(summary, Mapping):
        raise ControlRuntimeError("confirmation evaluation has no summary")
    try:
        mean = float(summary["mean_survival_frames"])
        median = float(summary["median_survival_frames"])
        p10 = float(summary["p10_survival_frames"])
        worst = float(summary["worst_survival_frames"])
    except (KeyError, TypeError, ValueError) as error:
        raise ControlRuntimeError(
            "confirmation evaluation summary is invalid"
        ) from error
    return {
        "mean_survival_frames": mean,
        "median_survival_frames": median,
        "p10_survival_frames": p10,
        "worst_survival_frames": worst,
        "score": mean + 0.25 * p10,
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


def _cached_run(
    path: Path,
    *,
    manifest: SeedManifest,
    trial: int,
    seed: int,
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if (
        value.get("manifest_sha256") != manifest.sha256
        or value.get("candidate_trial") != trial
        or value.get("learner_seed") != seed
    ):
        return None
    return value


def _cached_confirmation(
    path: Path,
    *,
    manifest: SeedManifest,
    hpo_path: Path,
    candidate_trials: Sequence[int],
    learner_seeds: Sequence[int],
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if (
        value.get("manifest_sha256") != manifest.sha256
        or value.get("hpo_result") != str(hpo_path)
        or value.get("candidate_trials") != list(candidate_trials)
        or value.get("learner_seeds") != list(learner_seeds)
        or not isinstance(value.get("holdout_evaluation"), dict)
    ):
        return None
    return value


def _hpo_reusable_seed(
    hpo: Mapping[str, object],
    *,
    manifest: SeedManifest,
    hpo_path: Path,
    candidate_trial: int,
    learner_seed: int,
    total_steps: int,
    parameters: Mapping[str, object],
) -> dict[str, object] | None:
    selected = hpo.get("selected_trial")
    if not isinstance(selected, Mapping):
        return None
    if selected.get("number") != candidate_trial:
        return None
    if selected.get("params") != dict(parameters):
        return None
    selected_checkpoint = hpo.get("selected_checkpoint")
    selected_directory = hpo.get("selected_trial_directory")
    training_evaluation = hpo.get("training_evaluation")
    if not all(
        isinstance(value, str)
        for value in (selected_checkpoint, selected_directory)
    ) or not isinstance(training_evaluation, dict):
        return None
    source_directory = Path(selected_directory)
    source_checkpoint = Path(selected_checkpoint)
    if not source_directory.is_absolute():
        source_directory = PROJECT_ROOT / source_directory
    if not source_checkpoint.is_absolute():
        source_checkpoint = PROJECT_ROOT / source_checkpoint
    source_run = source_directory / "run.json"
    try:
        source_record = json.loads(source_run.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(source_record, dict):
        return None
    source_config = source_record.get("config")
    if not isinstance(source_config, Mapping):
        return None
    if (
        source_record.get("manifest_sha256") != manifest.sha256
        or source_config.get("seed") != learner_seed
        or source_config.get("total_steps") != total_steps
        or not source_checkpoint.is_file()
    ):
        return None
    return {
        "source": str(hpo_path),
        "checkpoint": str(selected_checkpoint),
        "training_evaluation": training_evaluation,
    }


def run_confirmation(config: ConfirmationConfig) -> dict[str, object]:
    config.validate()
    manifest = load_manifest(config.manifest_path)
    manifest.validate()
    if config.native_lanes > len(manifest.training_seeds):
        raise ValueError("confirmation lane count exceeds training seed count")
    hpo = _load_object(config.hpo_path, "HPO result")
    if hpo.get("manifest_sha256") != manifest.sha256:
        raise ControlRuntimeError("HPO result does not match the confirmation manifest")
    raw_budgets = hpo.get("budgets")
    if not isinstance(raw_budgets, list) or not raw_budgets:
        raise ControlRuntimeError("HPO result has no training budgets")
    try:
        total_steps = int(raw_budgets[-1])
    except (TypeError, ValueError) as error:
        raise ControlRuntimeError("HPO final budget is invalid") from error
    if total_steps < 1:
        raise ControlRuntimeError("HPO final budget must be positive")
    candidates = _candidate_records(hpo, config.candidate_trials)
    config.run_directory.mkdir(parents=True, exist_ok=True)
    cached = _cached_confirmation(
        config.run_directory / "confirmation.json",
        manifest=manifest,
        hpo_path=config.hpo_path,
        candidate_trials=config.candidate_trials,
        learner_seeds=config.learner_seeds,
    )
    if cached is not None:
        return cached
    candidate_results: list[dict[str, object]] = []
    for candidate in candidates:
        trial = int(candidate["number"])
        parameters = candidate["params"]
        assert isinstance(parameters, dict)
        learner_results: list[dict[str, object]] = []
        for learner_seed in config.learner_seeds:
            seed_directory = (
                config.run_directory
                / f"trial-{trial:04d}"
                / f"learner-{learner_seed}"
            )
            result_path = seed_directory / "confirmation.json"
            cached = _cached_run(
                result_path,
                manifest=manifest,
                trial=trial,
                seed=learner_seed,
            )
            if cached is not None:
                learner_results.append(cached)
                continue
            train_config = _trial_config(
                parameters,
                total_steps,
                learner_seed,
                config.native_lanes,
                config.device,
            )
            reusable = _hpo_reusable_seed(
                hpo,
                manifest=manifest,
                hpo_path=config.hpo_path,
                candidate_trial=trial,
                learner_seed=learner_seed,
                total_steps=total_steps,
                parameters=parameters,
            )
            if reusable is not None:
                training_evaluation = _mapping(reusable["training_evaluation"])
                learner_result = {
                    "manifest_sha256": manifest.sha256,
                    "candidate_trial": trial,
                    "learner_seed": learner_seed,
                    "run_directory": str(seed_directory),
                    "checkpoint": reusable["checkpoint"],
                    "config": train_config.to_json(),
                    "training_evaluation": training_evaluation,
                    "training_metrics": _evaluation_score(training_evaluation),
                    "reused_from_hpo": reusable["source"],
                }
                _write_json(result_path, learner_result)
                learner_results.append(learner_result)
                continue
            train_waypoint_dqn(
                train_config,
                seed_directory,
                manifest,
                resume=(seed_directory / "checkpoint-latest.pt").is_file(),
                evaluate_holdout=False,
                evaluate_training=False,
            )
            checkpoint = seed_directory / "checkpoint-best.pt"
            model = _load_best_model(checkpoint, hidden_size=train_config.hidden_size)
            training_evaluation = evaluate_waypoint_dqn(
                model,
                manifest.training_seeds,
                train_config,
            )
            metrics = _evaluation_score(training_evaluation)
            learner_result: dict[str, object] = {
                "manifest_sha256": manifest.sha256,
                "candidate_trial": trial,
                "learner_seed": learner_seed,
                "run_directory": str(seed_directory),
                "checkpoint": str(checkpoint),
                "config": train_config.to_json(),
                "training_evaluation": training_evaluation,
                "training_metrics": metrics,
            }
            _write_json(result_path, learner_result)
            learner_results.append(learner_result)
        scores = [
            float(_mapping(result["training_metrics"])["score"])
            for result in learner_results
        ]
        means = [
            float(_mapping(result["training_metrics"])["mean_survival_frames"])
            for result in learner_results
        ]
        p10s = [
            float(_mapping(result["training_metrics"])["p10_survival_frames"])
            for result in learner_results
        ]
        candidate_results.append(
            {
                "trial": trial,
                "params": parameters,
                "learner_results": learner_results,
                "aggregate_training": {
                    "mean_score": statistics.fmean(scores),
                    "median_score": statistics.median(scores),
                    "worst_score": min(scores),
                    "mean_survival_frames": statistics.fmean(means),
                    "mean_p10_survival_frames": statistics.fmean(p10s),
                },
            }
        )
    selected_candidate = max(
        candidate_results,
        key=lambda value: (
            float(_mapping(value["aggregate_training"])["mean_score"]),
            float(_mapping(value["aggregate_training"])["worst_score"]),
        ),
    )
    selected_learner = max(
        _sequence(selected_candidate["learner_results"]),
        key=lambda value: float(
            _mapping(_mapping(value)["training_metrics"])["score"]
        ),
    )
    selected_learner = dict(selected_learner)
    selected_seed = int(selected_learner["learner_seed"])
    selected_parameters = selected_candidate["params"]
    assert isinstance(selected_parameters, dict)
    evaluation_config = _trial_config(
        selected_parameters,
        total_steps,
        selected_seed,
        config.native_lanes,
        config.device,
    )
    selected_checkpoint = Path(str(selected_learner["checkpoint"]))
    selected_model = _load_best_model(
        selected_checkpoint,
        hidden_size=evaluation_config.hidden_size,
    )
    holdout_evaluation = evaluate_waypoint_dqn(
        selected_model,
        manifest.holdout_seeds,
        evaluation_config,
    )
    selected_training = _mapping(selected_learner["training_evaluation"])
    selected_training_metrics = _mapping(selected_learner["training_metrics"])
    selected_aggregate = _mapping(selected_candidate["aggregate_training"])
    result: dict[str, object] = {
        "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "kind": "dodge_ng_waypoint_dqn_confirmation",
        "host": platform.node(),
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.sha256,
        "hpo_result": str(config.hpo_path),
        "hpo_study_name": hpo.get("study_name"),
        "budgets": list(raw_budgets),
        "total_steps": total_steps,
        "candidate_trials": list(config.candidate_trials),
        "learner_seeds": list(config.learner_seeds),
        "training_seeds": list(manifest.training_seeds),
        "holdout_seeds": list(manifest.holdout_seeds),
        "holdout_evaluated_once_after_training_selection": True,
        "candidate_results": candidate_results,
        "selected_candidate": selected_candidate,
        "selected_learner": selected_learner,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_training_evaluation": selected_training,
        "holdout_evaluation": holdout_evaluation,
        "target": {
            "relevance_gate_frames": 800,
            "training_mean_reached": float(
                selected_training_metrics["mean_survival_frames"]
            )
            >= 800,
        },
        "selection": {
            "metric": "mean training-side learner-seed score",
            "score": float(selected_aggregate["mean_score"]),
            "holdout_used_for_selection": False,
        },
    }
    _write_json(config.run_directory / "confirmation.json", result)
    _write_report(config.run_directory / "REPORT.md", result)
    return result


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ControlRuntimeError("confirmation result contains an invalid object")
    return value


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ControlRuntimeError("confirmation result contains an invalid list")
    return value


def _write_report(path: Path, result: Mapping[str, object]) -> None:
    selected_candidate = _mapping(result["selected_candidate"])
    selected_learner = _mapping(result["selected_learner"])
    selected_training = _mapping(result["selected_training_evaluation"])
    holdout = _mapping(result["holdout_evaluation"])
    training_metrics = _evaluation_score(selected_training)
    holdout_metrics = _evaluation_score(holdout)
    rows = [
        "| Candidate | Mean score | Worst learner score | Mean frames | Mean P10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in _sequence(result["candidate_results"]):
        value = _mapping(candidate)
        aggregate = _mapping(value["aggregate_training"])
        rows.append(
            f"| Trial {int(value['trial'])} | {float(aggregate['mean_score']):.2f} | "
            f"{float(aggregate['worst_score']):.2f} | "
            f"{float(aggregate['mean_survival_frames']):.1f} | "
            f"{float(aggregate['mean_p10_survival_frames']):.1f} |"
        )
    content = [
        "# Dodge NG waypoint DQN learner-seed confirmation",
        "",
        f"Manifest SHA-256: `{result['manifest_sha256']}`  ",
        f"HPO result: `{result['hpo_result']}`  ",
        f"Candidates: `{result['candidate_trials']}`; "
        f"learner seeds: `{result['learner_seeds']}`  ",
        "",
        *rows,
        "",
        f"Selected trial: `{selected_candidate['trial']}`; learner seed: "
        f"`{selected_learner['learner_seed']}`.",
        "",
        "| Split | Mean | Median | P10 | Worst | Score |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Training | {training_metrics['mean_survival_frames']:.1f} | "
        f"{training_metrics['median_survival_frames']:.1f} | "
        f"{training_metrics['p10_survival_frames']:.1f} | "
        f"{training_metrics['worst_survival_frames']:.1f} | "
        f"{training_metrics['score']:.2f} |",
        f"| Holdout | {holdout_metrics['mean_survival_frames']:.1f} | "
        f"{holdout_metrics['median_survival_frames']:.1f} | "
        f"{holdout_metrics['p10_survival_frames']:.1f} | "
        f"{holdout_metrics['worst_survival_frames']:.1f} | "
        f"{holdout_metrics['score']:.2f} |",
        "",
        "Candidate and learner-seed selection used training seeds only. The locked "
        "holdout was evaluated once after that selection.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be comma-separated integers"
        ) from error
    if not parsed:
        raise argparse.ArgumentTypeError("value must contain at least one integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-confirm")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--hpo-json", type=Path, default=DEFAULT_HPO_PATH)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIRECTORY)
    parser.add_argument(
        "--candidate-trials",
        type=_parse_csv_ints,
        default=DEFAULT_CANDIDATE_TRIALS,
        help="comma-separated complete HPO trial numbers",
    )
    parser.add_argument(
        "--learner-seeds",
        type=_parse_csv_ints,
        default=DEFAULT_LEARNER_SEEDS,
        help="comma-separated independent learner RNG seeds",
    )
    parser.add_argument("--native-lanes", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    arguments = parser.parse_args(argv)
    config = ConfirmationConfig(
        manifest_path=arguments.manifest,
        hpo_path=arguments.hpo_json,
        run_directory=arguments.run_dir,
        candidate_trials=arguments.candidate_trials,
        learner_seeds=arguments.learner_seeds,
        native_lanes=arguments.native_lanes,
        device=arguments.device,
    )
    try:
        started = time.monotonic()
        result = run_confirmation(config)
        result["wall_seconds"] = time.monotonic() - started
        _write_json(config.run_directory / "confirmation.json", result)
    except (ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-ng-confirm: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_directory": str(config.run_directory),
                "selected_trial": result["selected_candidate"]["trial"],
                "selected_learner_seed": result["selected_learner"]["learner_seed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
