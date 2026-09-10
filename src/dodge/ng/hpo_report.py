"""Post-hoc report and integrity audit for waypoint-DDQN HPO artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dodge.dataset import ACTION_CHOICES
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest
from dodge.ng.report import summarize_evaluation

HPO_REPORT_SCHEMA_VERSION = 1
PLOT_NAMES = (
    "hpo_trial_curves.png",
    "hpo_split_comparison.png",
    "hpo_per_seed_survival.png",
)
_SCALAR_METRICS = (
    "collection_seconds",
    "learning_seconds",
    "evaluation_seconds",
    "checkpoint_seconds",
    "epsilon",
    "loss",
    "td_error",
    "gradient_norm",
    "q_mean",
    "target_mean",
    "macro_reward_mean",
    "life_loss_count",
    "final_death_count",
    "corner_target_count",
    "replay_size",
)


def build_hpo_report(
    run_directory: Path,
    manifest: SeedManifest,
    *,
    job_directory: Path | None = None,
) -> dict[str, object]:
    """Build a full HPO report from one retrieved run directory."""
    root = Path(run_directory).resolve()
    hpo = _load_object(root / "hpo.json", "HPO result")
    manifest.validate()
    routing = _validate_seed_routing(hpo, manifest)
    trial_records = _trial_records(hpo)
    trials = [_trial_report(root, trial_record) for trial_record in trial_records]
    selected = _object(hpo.get("selected_trial"), "selected trial")
    selected_number = _integer(selected, "number")
    selected_trial = next(
        (trial for trial in trials if trial["number"] == selected_number), None
    )
    if selected_trial is None:
        raise ValueError(f"selected trial is not present: {selected_number}")
    splits = {
        "training": _evaluation_summary(hpo.get("training_evaluation")),
        "holdout": _evaluation_summary(hpo.get("holdout_evaluation")),
        "inner_validation": _evaluation_summary(selected_trial.get("final_validation")),
    }
    comparison = _split_comparison(splits)
    performance = _performance_summary(root, hpo, trials)
    artifact_audit = _artifact_audit(job_directory, root, manifest)
    failure_modes = _failure_modes(
        hpo,
        selected_trial,
        splits,
        comparison,
        trials,
    )
    report: dict[str, object] = {
        "schema_version": HPO_REPORT_SCHEMA_VERSION,
        "kind": "dodge_ng_waypoint_dqn_hpo_performance_report",
        "run_directory": str(root),
        "manifest": {
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": manifest.sha256,
            "training_count": len(manifest.training_seeds),
            "holdout_count": len(manifest.holdout_seeds),
            "training_seeds": list(manifest.training_seeds),
            "holdout_seeds": list(manifest.holdout_seeds),
        },
        "routing": routing,
        "selection": {
            "selected_trial": selected,
            "selected_trial_number": selected_number,
            "selected_trial_directory": str(root / f"trial-{selected_number:04d}"),
        },
        "splits": splits,
        "comparison": comparison,
        "performance": performance,
        "trials": trials,
        "failure_modes": failure_modes,
        "artifact_audit": artifact_audit,
        "plots": list(PLOT_NAMES),
    }
    _write_report_files(root, report)
    return report


def _trial_records(hpo: Mapping[str, object]) -> list[dict[str, object]]:
    raw = hpo.get("trials")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("HPO result has no trial records")
    records: list[dict[str, object]] = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise ValueError("HPO trial record must be an object")
        records.append(dict(value))
    return sorted(records, key=lambda record: _integer(record, "number"))


def _trial_report(
    run_directory: Path,
    trial_record: Mapping[str, object],
) -> dict[str, object]:
    number = _integer(trial_record, "number")
    trial_directory = run_directory / f"trial-{number:04d}"
    run_path = trial_directory / "run.json"
    run = _load_object(run_path, f"trial {number} run") if run_path.is_file() else {}
    config = run.get("config")
    config_object = dict(config) if isinstance(config, Mapping) else {}
    final_validation = run.get("final_validation")
    validation = (
        _evaluation_summary(final_validation)
        if isinstance(final_validation, Mapping)
        else None
    )
    metrics_path = trial_directory / "metrics.jsonl"
    metrics = _metric_report(metrics_path) if metrics_path.is_file() else {}
    user_attrs = trial_record.get("user_attrs")
    attrs = dict(user_attrs) if isinstance(user_attrs, Mapping) else {}
    return {
        "number": number,
        "state": trial_record.get("state"),
        "value": trial_record.get("value"),
        "params": dict(trial_record.get("params", {}))
        if isinstance(trial_record.get("params"), Mapping)
        else {},
        "intermediate_values": dict(trial_record.get("intermediate_values", {}))
        if isinstance(trial_record.get("intermediate_values"), Mapping)
        else {},
        "rung_metrics": attrs.get("rung_metrics", []),
        "directory": str(trial_directory),
        "config": config_object,
        "observation_size": run.get("observation_size"),
        "updates_completed": run.get("updates_completed"),
        "best_inner": run.get("best_inner"),
        "selected_model": run.get("selected_model"),
        "holdout_evaluated": run.get("holdout_evaluated"),
        "training_evaluated": run.get("training_evaluated"),
        "final_validation": validation,
        "metrics": metrics,
    }


def _metric_report(path: Path) -> dict[str, object]:
    accumulators: dict[str, dict[str, float | int | None]] = {}
    evaluation_points: list[dict[str, float]] = []
    action_counts = [0] * len(ACTION_CHOICES)
    action_records = 0
    record_count = 0
    first_step: int | None = None
    last_step: int | None = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid metrics line {line_number} in {path}: {error}"
                ) from error
            if not isinstance(value, Mapping):
                raise ValueError(f"metrics line {line_number} must be an object")
            record_count += 1
            step = value.get("step")
            if isinstance(step, int) and not isinstance(step, bool):
                first_step = step if first_step is None else first_step
                last_step = step
            for name in _SCALAR_METRICS:
                raw = value.get(name)
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    continue
                if not math.isfinite(float(raw)):
                    continue
                _add_stat(accumulators, name, float(raw))
            nested = value.get("inner_validation")
            if isinstance(nested, Mapping) and isinstance(step, int):
                point: dict[str, float] = {"step": float(step)}
                for name in (
                    "mean_survival_frames",
                    "median_survival_frames",
                    "p10_survival_frames",
                    "worst_survival_frames",
                    "horizon_completion_fraction",
                ):
                    raw = nested.get(name)
                    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                        point[name] = float(raw)
                if "mean_survival_frames" in point:
                    evaluation_points.append(point)
            raw_counts = value.get("waypoint_action_counts")
            if _valid_action_counts(raw_counts):
                action_records += 1
                for index, raw in enumerate(raw_counts):
                    action_counts[index] += int(raw)
    return {
        "path": str(path),
        "record_count": record_count,
        "first_step": first_step,
        "last_step": last_step,
        "evaluation_points": evaluation_points,
        "scalars": {name: _finish_stat(stat) for name, stat in accumulators.items()},
        "action_counts": action_counts if action_records else None,
        "action_records": action_records,
        "phase_timing": _phase_timing(accumulators),
    }


def _add_stat(
    accumulators: dict[str, dict[str, float | int | None]],
    name: str,
    value: float,
) -> None:
    stat = accumulators.setdefault(
        name,
        {"count": 0, "sum": 0.0, "min": None, "max": None, "first": None, "last": None},
    )
    stat["count"] = int(stat["count"] or 0) + 1
    stat["sum"] = float(stat["sum"] or 0.0) + value
    stat["min"] = value if stat["min"] is None else min(float(stat["min"]), value)
    stat["max"] = value if stat["max"] is None else max(float(stat["max"]), value)
    if stat["first"] is None:
        stat["first"] = value
    stat["last"] = value


def _finish_stat(
    stat: Mapping[str, float | int | None],
) -> dict[str, float | int | None]:
    count = int(stat.get("count") or 0)
    total = float(stat.get("sum") or 0.0)
    return {
        **dict(stat),
        "mean": total / count if count else None,
    }


def _phase_timing(
    accumulators: Mapping[str, Mapping[str, float | int | None]],
) -> dict[str, object]:
    names = (
        "collection_seconds",
        "learning_seconds",
        "evaluation_seconds",
        "checkpoint_seconds",
    )
    totals = {
        name: float(accumulators[name].get("sum") or 0.0)
        for name in names
        if name in accumulators
    }
    total = sum(totals.values())
    return {
        "available": bool(totals),
        "seconds": totals,
        "total_seconds": total,
        "fractions": {
            name: value / total if total else 0.0 for name, value in totals.items()
        },
    }


def _performance_summary(
    run_directory: Path,
    hpo: Mapping[str, object],
    trials: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    complete = [trial for trial in trials if trial.get("state") == "COMPLETE"]
    total_updates = sum(
        int(trial["updates_completed"])
        for trial in complete
        if isinstance(trial.get("updates_completed"), int)
    )
    estimated_training_frames = 0
    estimated_validation_frames = 0
    observed_validation_points = 0
    phase_seconds: dict[str, float] = {}
    phase_available = False
    inner_seed_count = len(_integer_list(hpo, "inner_validation_seeds"))
    for trial in complete:
        config = trial.get("config")
        if not isinstance(config, Mapping):
            continue
        updates = trial.get("updates_completed")
        if isinstance(updates, int):
            estimated_training_frames += (
                updates
                * int(config.get("native_lanes", 0))
                * int(config.get("hold_decisions", 0))
                * int(config.get("step_frames", 0))
            )
        metrics = trial.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        points = metrics.get("evaluation_points")
        if isinstance(points, Sequence):
            observed_validation_points += len(points)
            estimated_validation_frames += (
                len(points)
                * inner_seed_count
                * int(config.get("max_episode_steps", 0))
                * int(config.get("hold_decisions", 0))
                * int(config.get("step_frames", 0))
            )
        timing = metrics.get("phase_timing")
        if isinstance(timing, Mapping) and timing.get("available"):
            phase_available = True
            seconds = timing.get("seconds")
            if isinstance(seconds, Mapping):
                for name, value in seconds.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        phase_seconds[name] = phase_seconds.get(name, 0.0) + float(
                            value
                        )
    return {
        "wall_seconds": hpo.get("wall_seconds"),
        "complete_trial_count": len(complete),
        "total_updates": total_updates,
        "estimated_training_game_frames": estimated_training_frames,
        "observed_inner_validation_points": observed_validation_points,
        "estimated_validation_game_frames": estimated_validation_frames,
        "phase_timing_available": phase_available,
        "phase_seconds": phase_seconds,
        "artifact_run_directory": str(run_directory),
    }


def _split_comparison(splits: Mapping[str, object]) -> dict[str, float]:
    training = _object(splits.get("training"), "training")
    holdout = _object(splits.get("holdout"), "holdout")
    keys = (
        "mean_survival_frames",
        "median_survival_frames",
        "p10_survival_frames",
        "worst_survival_frames",
        "horizon_completion_fraction",
    )
    return {
        f"train_minus_holdout_{key}": _number(training, key) - _number(holdout, key)
        for key in keys
    }


def _failure_modes(
    hpo: Mapping[str, object],
    selected_trial: Mapping[str, object],
    splits: Mapping[str, object],
    comparison: Mapping[str, float],
    trials: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    training = _object(splits.get("training"), "training")
    holdout = _object(splits.get("holdout"), "holdout")
    inner = _object(splits.get("inner_validation"), "inner validation")
    gate = 800.0
    best_inner = selected_trial.get("best_inner")
    best_step = (
        best_inner.get("step")
        if isinstance(best_inner, Mapping)
        and isinstance(best_inner.get("step"), int)
        and not isinstance(best_inner.get("step"), bool)
        else None
    )
    updates_completed = selected_trial.get("updates_completed")
    checkpoint_timing = "timing unavailable"
    if isinstance(best_step, int) and isinstance(updates_completed, int):
        checkpoint_timing = (
            "best inner checkpoint was final step"
            if best_step == updates_completed
            else "best inner checkpoint was not final step"
        )
    training_gate = _number(training, "mean_survival_frames") >= gate
    holdout_gate = _number(holdout, "mean_survival_frames") >= gate
    if training_gate and holdout_gate:
        gate_finding = "training and holdout gates passed"
    elif training_gate:
        gate_finding = "training-side gate passed; holdout below gate"
    else:
        gate_finding = "training-side gate failed"
    findings: list[dict[str, object]] = [
        {
            "name": "relevance_gate",
            "finding": gate_finding,
            "evidence": {
                "gate_frames": gate,
                "training_mean": _number(training, "mean_survival_frames"),
                "holdout_mean": _number(holdout, "mean_survival_frames"),
            },
        },
        {
            "name": "generalization_gap",
            "finding": "train/holdout split differs",
            "evidence": dict(comparison),
            "confidence": "measured",
        },
        {
            "name": "tail_risk",
            "finding": "tail is below mean"
            if _number(holdout, "p10_survival_frames")
            < _number(holdout, "mean_survival_frames")
            else "not observed",
            "evidence": {
                "holdout_mean": _number(holdout, "mean_survival_frames"),
                "holdout_p10": _number(holdout, "p10_survival_frames"),
                "holdout_worst": _number(holdout, "worst_survival_frames"),
            },
            "confidence": "measured",
        },
        {
            "name": "best_checkpoint_timing",
            "finding": checkpoint_timing,
            "evidence": {
                "best_inner": best_inner,
                "updates_completed": updates_completed,
            },
            "confidence": "measured",
        },
    ]
    metrics = selected_trial.get("metrics")
    if isinstance(metrics, Mapping):
        scalars = metrics.get("scalars")
        action_counts = metrics.get("action_counts")
        if isinstance(action_counts, Sequence) and action_counts:
            total = sum(int(value) for value in action_counts)
            peak = max(int(value) for value in action_counts)
            findings.append(
                {
                    "name": "action_concentration",
                    "finding": "action concentration measured",
                    "evidence": {
                        "counts": list(action_counts),
                        "labels": list(ACTION_CHOICES),
                        "max_fraction": peak / total if total else 0.0,
                    },
                    "confidence": "chosen actions include epsilon exploration",
                }
            )
        else:
            findings.append(
                {
                    "name": "action_concentration",
                    "finding": "not measured in this artifact",
                    "evidence": "No waypoint_action_counts telemetry was recorded.",
                    "confidence": "unavailable",
                }
            )
        if isinstance(scalars, Mapping):
            findings.append(
                {
                    "name": "learner_dynamics",
                    "finding": "raw learner dynamics retained for diagnosis",
                    "evidence": {
                        name: scalars[name]
                        for name in (
                            "loss",
                            "td_error",
                            "gradient_norm",
                            "q_mean",
                            "target_mean",
                        )
                        if name in scalars
                    },
                    "confidence": "measured; causal interpretation deferred",
                }
            )
    complete_scores = [
        float(trial["value"])
        for trial in trials
        if trial.get("state") == "COMPLETE"
        and isinstance(trial.get("value"), (int, float))
        and not isinstance(trial.get("value"), bool)
    ]
    if complete_scores:
        findings.append(
            {
                "name": "trial_variance",
                "finding": "hyperparameters materially change inner score",
                "evidence": {
                    "best_score": max(complete_scores),
                    "worst_score": min(complete_scores),
                    "spread": max(complete_scores) - min(complete_scores),
                    "complete_trials": len(complete_scores),
                },
                "confidence": "measured on training-side selection scores",
            }
        )
    config = selected_trial.get("config")
    if isinstance(config, Mapping):
        observation_mode = config.get("observation_mode", "legacy")
        findings.append(
            {
                "name": "representation",
                "finding": f"selected observation_mode={observation_mode}",
                "evidence": {
                    "observation_mode": config.get("observation_mode", "legacy"),
                    "observation_size": selected_trial.get("observation_size"),
                    "hold_decisions": config.get("hold_decisions"),
                    "step_frames": config.get("step_frames"),
                },
                "confidence": "configuration fact; not a causal claim",
            }
        )
    del hpo, inner
    return findings


def _validate_seed_routing(
    hpo: Mapping[str, object], manifest: SeedManifest
) -> dict[str, object]:
    training = _integer_list(hpo, "training_seeds")
    inner = _integer_list(hpo, "inner_validation_seeds")
    holdout = _integer_list(hpo, "holdout_seeds")
    if training != manifest.training_seeds:
        raise ValueError("HPO training seeds do not match manifest")
    if holdout != manifest.holdout_seeds:
        raise ValueError("HPO holdout seeds do not match manifest")
    if not set(inner) <= set(training):
        raise ValueError("HPO inner validation contains a holdout seed")
    if hpo.get("holdout_evaluated_once_after_selection") is not True:
        raise ValueError("HPO holdout-selection boundary is not asserted")
    holdout_eval = _object(hpo.get("holdout_evaluation"), "holdout evaluation")
    if _integer_list(holdout_eval, "seeds") != manifest.holdout_seeds:
        raise ValueError("HPO holdout evaluation seeds do not match manifest")
    training_eval = _object(hpo.get("training_evaluation"), "training evaluation")
    if _integer_list(training_eval, "seeds") != manifest.training_seeds:
        raise ValueError("HPO training evaluation seeds do not match manifest")
    return {
        "training_matches_manifest": True,
        "inner_subset_of_training": True,
        "holdout_matches_manifest": True,
        "holdout_evaluated_once_after_selection": True,
        "trial_objectives_holdout_free": True,
    }


def _artifact_audit(
    job_directory: Path | None,
    run_directory: Path,
    manifest: SeedManifest,
) -> dict[str, object]:
    if job_directory is None:
        return {"available": False, "reason": "job directory not supplied"}
    root = Path(job_directory).resolve()
    job = _load_object(root / "job.json", "job record")
    status = _load_object(root / "status.json", "job status")
    inventory = _load_object(root / "artifact-manifest.json", "artifact manifest")
    payload = _object(job.get("payload"), "payload")
    job_manifest = _object(job.get("manifest"), "manifest")
    expected = {
        "submission_id": job.get("submission_id"),
        "job_identity_sha256": job.get("job_identity_sha256"),
        "manifest_sha256": manifest.sha256,
        "source_bundle_sha256": payload.get("source_bundle_sha256"),
        "native_wheel_sha256": payload.get("native_wheel_sha256"),
        "payload_sha256": payload.get("payload_sha256"),
    }
    identity = {key: inventory.get(key) == value for key, value in expected.items()}
    files = inventory.get("files")
    file_checks: list[dict[str, object]] = []
    if not isinstance(files, list):
        raise ValueError("artifact manifest file inventory is invalid")
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("artifact manifest file entry is invalid")
        relative = item.get("path")
        expected_hash = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ValueError("artifact manifest file entry lacks path/hash")
        path = run_directory / relative
        actual_hash = _sha256_file(path) if path.is_file() else None
        file_checks.append(
            {
                "path": relative,
                "exists": path.is_file(),
                "hash_matches": actual_hash == expected_hash,
                "bytes_matches": path.stat().st_size == item.get("bytes")
                if path.is_file() and isinstance(item.get("bytes"), int)
                else False,
            }
        )
    receipt = run_directory / ".colab-artifact-receipt.json"
    receipt_object = (
        _load_object(receipt, "artifact receipt") if receipt.is_file() else {}
    )
    terminal = status.get("state") in {"succeeded", "failed", "cancelled"}
    all_files = all(
        bool(item["exists"] and item["hash_matches"] and item["bytes_matches"])
        for item in file_checks
    )
    return {
        "available": True,
        "job_id": job.get("job_id"),
        "status_state": status.get("state"),
        "terminal": terminal,
        "trainer_exit_code": inventory.get("trainer_exit_code"),
        "complete": inventory.get("complete") is True,
        "identity": identity,
        "job_manifest_sha256": job_manifest.get("sha256"),
        "file_count": len(file_checks),
        "files_all_verified": all_files,
        "files": file_checks,
        "receipt_present": receipt.is_file(),
        "receipt_archive_sha256": receipt_object.get("archive_sha256"),
        "fully_verified": (
            terminal
            and inventory.get("complete") is True
            and inventory.get("trainer_exit_code") == 0
            and all(identity.values())
            and all_files
            and receipt.is_file()
        ),
    }


def _write_report_files(run_directory: Path, report: Mapping[str, object]) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "hpo_performance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_trial_curves(run_directory, report)
    _plot_split_comparison(run_directory, report)
    _plot_per_seed(run_directory, report)
    (run_directory / "HPO_PERFORMANCE_REPORT.md").write_text(
        _markdown(report), encoding="utf-8"
    )


def _plot_trial_curves(run_directory: Path, report: Mapping[str, object]) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    trials = report.get("trials", [])
    selected = _integer(
        _object(report.get("selection"), "selection"), "selected_trial_number"
    )
    if isinstance(trials, Sequence):
        for trial in trials:
            if not isinstance(trial, Mapping):
                continue
            metrics = trial.get("metrics")
            points = (
                metrics.get("evaluation_points") if isinstance(metrics, Mapping) else []
            )
            if not isinstance(points, Sequence) or not points:
                continue
            curve_points = [
                point
                for point in points
                if isinstance(point, Mapping)
                and isinstance(point.get("step"), (int, float))
                and isinstance(point.get("mean_survival_frames"), (int, float))
            ]
            if not curve_points:
                continue
            x = [float(point["step"]) for point in curve_points]
            y = [float(point["mean_survival_frames"]) for point in curve_points]
            axis.plot(
                x,
                y,
                label=f"trial {trial['number']}",
                linewidth=2.0 if trial.get("number") == selected else 1.0,
                alpha=1.0 if trial.get("number") == selected else 0.55,
            )
    axis.axhline(800, color="black", linestyle=":", label="800-frame gate")
    axis.set_title("Inner-validation survival by HPO trial")
    axis.set_xlabel("macro update")
    axis.set_ylabel("mean survival frames")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(run_directory / PLOT_NAMES[0], dpi=140)
    plt.close(figure)


def _plot_split_comparison(run_directory: Path, report: Mapping[str, object]) -> None:
    splits = _object(report.get("splits"), "splits")
    names = ("inner_validation", "training", "holdout")
    labels = ("inner", "training", "holdout")
    keys = (
        ("mean_survival_frames", "mean"),
        ("median_survival_frames", "median"),
        ("p10_survival_frames", "p10"),
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    width = 0.24
    positions = list(range(len(names)))
    for offset, (key, label) in enumerate(keys):
        values = [_number(_object(splits.get(name), name), key) for name in names]
        axis.bar(
            [position + (offset - 1) * width for position in positions],
            values,
            width=width,
            label=label,
        )
    axis.axhline(800, color="black", linestyle=":")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("survival frames")
    axis.set_title("Selected-policy split performance")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_directory / PLOT_NAMES[1], dpi=140)
    plt.close(figure)


def _plot_per_seed(run_directory: Path, report: Mapping[str, object]) -> None:
    splits = _object(report.get("splits"), "splits")
    training = _object(splits.get("training"), "training")
    holdout = _object(splits.get("holdout"), "holdout")
    train_values = _integer_list(training, "survival_frames")
    holdout_values = _integer_list(holdout, "survival_frames")
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(range(len(train_values)), train_values, label="training")
    start = len(train_values) + 2
    axis.bar(
        range(start, start + len(holdout_values)),
        holdout_values,
        label="locked holdout",
        color="C3",
    )
    axis.axhline(800, color="black", linestyle=":")
    axis.set_xlabel("seed order within split")
    axis.set_ylabel("survival frames")
    axis.set_title("Selected-policy survival for every evaluated seed")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_directory / PLOT_NAMES[2], dpi=140)
    plt.close(figure)


def _markdown(report: Mapping[str, object]) -> str:
    manifest = _object(report.get("manifest"), "manifest")
    selection = _object(report.get("selection"), "selection")
    splits = _object(report.get("splits"), "splits")
    comparison = _object(report.get("comparison"), "comparison")
    performance = _object(report.get("performance"), "performance")
    audit = _object(report.get("artifact_audit"), "artifact audit")
    selected = _object(selection.get("selected_trial"), "selected trial")
    selected_number = _integer(selection, "selected_trial_number")
    train_mean_gap = _number(comparison, "train_minus_holdout_mean_survival_frames")
    train_p10_gap = _number(comparison, "train_minus_holdout_p10_survival_frames")
    lines = [
        "# Dodge NG waypoint DDQN HPO performance report",
        "",
        f"Manifest: `{manifest['manifest_id']}`  ",
        f"Manifest SHA-256: `{manifest['manifest_sha256']}`  ",
        f"Seeds: {manifest['training_count']} training / "
        f"{manifest['holdout_count']} locked holdout  ",
        f"Selected trial: `{selected_number}`; "
        f"score `{_format(selected.get('value'))}`  ",
        "",
        "## Result",
        "",
        "| Split | Mean | Median | P10 | Worst | Best | Complete |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, name in (
        ("Inner validation", "inner_validation"),
        ("Training", "training"),
        ("Locked holdout", "holdout"),
    ):
        value = _object(splits.get(name), name)
        lines.append(
            f"| {label} | {_number(value, 'mean_survival_frames'):.1f} | "
            f"{_number(value, 'median_survival_frames'):.1f} | "
            f"{_number(value, 'p10_survival_frames'):.1f} | "
            f"{_number(value, 'worst_survival_frames'):.1f} | "
            f"{_number(value, 'best_survival_frames'):.1f} | "
            f"{_number(value, 'horizon_completion_fraction'):.1%} |"
        )
    lines.extend(
        [
            "",
            f"Train minus holdout mean: **{train_mean_gap:+.1f} frames**.",
            f"Train minus holdout P10: **{train_p10_gap:+.1f} frames**.",
            "",
            "## Holdout boundary",
            "",
            "HPO trial objectives use inner training seeds only. "
            "The locked holdout is evaluated after selection and is not used "
            "to choose parameters.",
            f"Routing checks: `{_format(report.get('routing'))}`  ",
            f"Artifact audit fully verified: "
            f"`{audit.get('fully_verified', 'not supplied')}`",
            "",
            "## Selected configuration",
            "",
            "```json",
            json.dumps(selected.get("params", {}), indent=2, sort_keys=True),
            "```",
            "",
            "## Trial comparison",
            "",
            "| Trial | State | Score | Inner mean | P10 | Worst | Best step | "
            "LR | Batch | N-step | Target update |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    trials = report.get("trials", [])
    if isinstance(trials, Sequence):
        for trial in trials:
            if not isinstance(trial, Mapping):
                continue
            params = trial.get("params")
            params = params if isinstance(params, Mapping) else {}
            validation = trial.get("final_validation")
            summary = validation if isinstance(validation, Mapping) else {}
            best_inner = trial.get("best_inner")
            best_step = (
                best_inner.get("step") if isinstance(best_inner, Mapping) else None
            )
            lines.append(
                f"| {trial.get('number')} | {trial.get('state')} | "
                f"{_format(trial.get('value'))} | "
                f"{_format(summary.get('mean_survival_frames'))} | "
                f"{_format(summary.get('p10_survival_frames'))} | "
                f"{_format(summary.get('worst_survival_frames'))} | "
                f"{best_step or '—'} | "
                f"{_format(params.get('learning_rate'))} | "
                f"{params.get('batch_size', '—')} | "
                f"{params.get('n_step', '—')} | "
                f"{params.get('target_update_interval', '—')} |"
            )
    lines.extend(
        [
            "",
            "## Runtime and overhead",
            "",
            f"Complete trials: `{performance.get('complete_trial_count')}`; "
            f"total learner updates: `{performance.get('total_updates')}`.",
            f"Configured training simulation estimate: "
            f"`{performance.get('estimated_training_game_frames')}` "
            "game-frame advances.",
            f"Observed inner-validation points: "
            f"`{performance.get('observed_inner_validation_points')}`.",
            f"Measured phase timing available: "
            f"`{performance.get('phase_timing_available')}`.",
            "",
            "## Failure-mode evidence",
            "",
        ]
    )
    findings = report.get("failure_modes", [])
    if isinstance(findings, Sequence):
        for finding in findings:
            if isinstance(finding, Mapping):
                finding_line = (
                    f"- **{finding.get('name')}**: "
                    f"{finding.get('finding')}; "
                    f"evidence `{_format(finding.get('evidence'))}`."
                )
                lines.append(finding_line)
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            *[f"- [{name}]({name})" for name in PLOT_NAMES],
            "- [hpo_performance_report.json](hpo_performance_report.json)",
            "- [hpo.json](hpo.json)",
            "- [REPORT.md](REPORT.md) (trainer-generated summary)",
            "",
        ]
    )
    return "\n".join(lines)


def _evaluation_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("evaluation must be an object")
    if all(key in value for key in ("seeds", "survival_frames", "terminated")):
        return summarize_evaluation(value)
    summary = value.get("summary")
    if isinstance(summary, Mapping):
        return dict(summary)
    raise ValueError("evaluation lacks summary or per-seed fields")


def _load_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object: {path}")
    return dict(value)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _number(value: Mapping[str, object], key: str) -> float:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{key} must be numeric")
    result = float(raw)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _integer(value: Mapping[str, object], key: str) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{key} must be an integer")
    return raw


def _integer_list(value: Mapping[str, object], key: str) -> tuple[int, ...]:
    raw = value.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{key} must be an integer list")
    result = tuple(raw)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise ValueError(f"{key} must be an integer list")
    return result


def _valid_action_counts(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == len(ACTION_CHOICES)
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in value
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-hpo-report")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--job-dir", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = build_hpo_report(
            arguments.run_directory,
            load_manifest(arguments.manifest),
            job_directory=arguments.job_dir,
        )
    except (OSError, ValueError) as error:
        print(f"dodge-ng-hpo-report: {error}", file=sys.stderr)
        return 1
    splits = _object(report.get("splits"), "splits")
    print(
        json.dumps(
            {
                "run_directory": str(arguments.run_directory),
                "selected_trial": _object(report.get("selection"), "selection").get(
                    "selected_trial_number"
                ),
                "holdout_mean_survival_frames": _number(
                    _object(splits.get("holdout"), "holdout"),
                    "mean_survival_frames",
                ),
                "report": str(
                    Path(arguments.run_directory) / "HPO_PERFORMANCE_REPORT.md"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
