from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest

REPORT_SCHEMA_VERSION = 1
PLOT_NAMES = (
    "survival_curves.png",
    "split_comparison.png",
    "per_seed_survival.png",
    "training_diagnostics.png",
)


def summarize_evaluation(value: Mapping[str, object]) -> dict[str, object]:
    """Normalize a PPO evaluation into comparable split statistics."""
    seeds = _integer_tuple(value, "seeds")
    survival = _integer_tuple(value, "survival_frames")
    terminated = _boolean_tuple(value, "terminated")
    if not seeds:
        raise ValueError("evaluation must contain at least one seed")
    if len({len(seeds), len(survival), len(terminated)}) != 1:
        raise ValueError("evaluation fields must have matching lengths")
    ordered = sorted(survival)
    p10_index = max(0, math.ceil(len(ordered) * 0.10) - 1)
    return {
        "count": len(seeds),
        "mean_survival_frames": statistics.fmean(survival),
        "median_survival_frames": statistics.median(survival),
        "p10_survival_frames": ordered[p10_index],
        "worst_survival_frames": ordered[0],
        "best_survival_frames": ordered[-1],
        "horizon_completion_fraction": sum(
            not terminated_value for terminated_value in terminated
        )
        / len(terminated),
        "seeds": list(seeds),
        "survival_frames": list(survival),
        "terminated": list(terminated),
        "per_seed": [
            {
                "seed": seed,
                "survival_frames": frames,
                "terminated": episode_terminated,
            }
            for seed, frames, episode_terminated in zip(
                seeds, survival, terminated, strict=True
            )
        ],
    }


def build_report(run_directory: Path, manifest: SeedManifest) -> dict[str, object]:
    """Build JSON, Markdown, and plots for one completed NG baseline run."""
    run_record = _load_object(run_directory / "run.json")
    metrics = _load_metrics(run_directory / "metrics.jsonl")
    config = _object(run_record, "config")
    _assert_seed_routing(config, run_record, manifest)

    final_training = summarize_evaluation(
        _object(run_record, "final_training_evaluation")
    )
    final_validation = summarize_evaluation(_object(run_record, "final_validation"))
    final_holdout = summarize_evaluation(_object(run_record, "final_evaluation"))
    curves = _curves(metrics)
    trend = _trend_summary(curves)
    comparison = {
        "mean_train_minus_holdout": float(final_training["mean_survival_frames"])
        - float(final_holdout["mean_survival_frames"]),
        "median_train_minus_holdout": float(final_training["median_survival_frames"])
        - float(final_holdout["median_survival_frames"]),
        "p10_train_minus_holdout": float(final_training["p10_survival_frames"])
        - float(final_holdout["p10_survival_frames"]),
        "horizon_completion_train_minus_holdout": float(
            final_training["horizon_completion_fraction"]
        )
        - float(final_holdout["horizon_completion_fraction"]),
    }
    ng_record = _optional_object(run_directory / "ng-run.json")
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "dodge_ng_baseline_report",
        "run_directory": str(run_directory),
        "manifest": {
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": manifest.sha256,
            "sample_count": manifest.sample_count,
            "training_count": len(manifest.training_seeds),
            "holdout_count": len(manifest.holdout_seeds),
            "training_seeds": list(manifest.training_seeds),
            "holdout_seeds": list(manifest.holdout_seeds),
        },
        "provenance": {
            "ppo_run": run_record,
            "ng_run": ng_record,
        },
        "splits": {
            "training": final_training,
            "inner_validation": final_validation,
            "holdout": final_holdout,
        },
        "comparison": comparison,
        "curves": curves,
        "trend": trend,
        "plots": list(PLOT_NAMES),
    }
    _write_report_files(run_directory, report)
    return report


def _assert_seed_routing(
    config: Mapping[str, object],
    run_record: Mapping[str, object],
    manifest: SeedManifest,
) -> None:
    configured_training = _integer_tuple(config, "training_seeds")
    if configured_training != manifest.training_seeds:
        raise ValueError("run training seeds do not exactly match the NG manifest")
    if config.get("training_seed_manifest") != manifest.sha256:
        raise ValueError("run training seed manifest hash does not match NG manifest")
    training_evaluation = _object(run_record, "final_training_evaluation")
    holdout_evaluation = _object(run_record, "final_evaluation")
    if _integer_tuple(training_evaluation, "seeds") != manifest.training_seeds:
        raise ValueError("final training evaluation is not the NG training split")
    if _integer_tuple(holdout_evaluation, "seeds") != manifest.holdout_seeds:
        raise ValueError("final evaluation is not the locked NG holdout split")
    validation = _object(run_record, "final_validation")
    validation_seeds = set(_integer_tuple(validation, "seeds"))
    if not validation_seeds <= set(manifest.training_seeds):
        raise ValueError("inner validation contains a holdout seed")


def _curves(
    metrics: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, float]]]:
    curves: dict[str, list[dict[str, float]]] = {
        "training_mean_survival": [],
        "inner_validation_mean_survival": [],
        "rollout_reward": [],
        "entropy": [],
        "policy_loss": [],
        "value_loss": [],
        "approx_kl": [],
        "clip_fraction": [],
        "explained_variance": [],
    }
    for metric in metrics:
        update = _number(metric, "update")
        _append_nested_curve(
            curves["training_mean_survival"], metric, "training_evaluation", update
        )
        _append_nested_curve(
            curves["inner_validation_mean_survival"],
            metric,
            "validation",
            update,
        )
        for name in (
            "rollout_reward",
            "entropy",
            "policy_loss",
            "value_loss",
            "approx_kl",
            "clip_fraction",
            "explained_variance",
        ):
            if name in metric:
                curves[name].append({"update": update, "value": _number(metric, name)})
    return curves


def _append_nested_curve(
    target: list[dict[str, float]],
    metric: Mapping[str, object],
    key: str,
    update: float,
) -> None:
    nested = metric.get(key)
    if isinstance(nested, Mapping) and "mean_survival_frames" in nested:
        target.append(
            {"update": update, "value": _number(nested, "mean_survival_frames")}
        )


def _trend_summary(
    curves: Mapping[str, Sequence[Mapping[str, float]]],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for name in ("training_mean_survival", "inner_validation_mean_survival"):
        points = curves[name]
        if not points:
            continue
        first = points[0]
        last = points[-1]
        best = max(points, key=lambda point: point["value"])
        update_delta = last["update"] - first["update"]
        summary[name] = {
            "first_value": first["value"],
            "last_value": last["value"],
            "gain": last["value"] - first["value"],
            "slope_per_update": (
                (last["value"] - first["value"]) / update_delta if update_delta else 0.0
            ),
            "best_value": best["value"],
            "best_update": best["update"],
        }
    return summary


def _write_report_files(run_directory: Path, report: Mapping[str, object]) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_survival_curves(run_directory, report)
    _plot_split_comparison(run_directory, report)
    _plot_per_seed(run_directory, report)
    _plot_diagnostics(run_directory, report)
    (run_directory / "REPORT.md").write_text(_markdown_report(report), encoding="utf-8")


def _plot_survival_curves(run_directory: Path, report: Mapping[str, object]) -> None:
    curves = _object(report, "curves")
    figure, axis = plt.subplots(figsize=(9, 5))
    _plot_curve(axis, curves, "training_mean_survival", "training mean", "C0")
    _plot_curve(
        axis,
        curves,
        "inner_validation_mean_survival",
        "inner validation mean",
        "C1",
    )
    holdout = _object(_object(report, "splits"), "holdout")
    axis.axhline(
        _number(holdout, "mean_survival_frames"),
        color="C3",
        linestyle="--",
        label="locked holdout final mean",
    )
    axis.set_title("Dodge NG survival during PPO training")
    axis.set_xlabel("update")
    axis.set_ylabel("survival frames")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_directory / "survival_curves.png", dpi=140)
    plt.close(figure)


def _plot_curve(
    axis: Any,
    curves: Mapping[str, object],
    key: str,
    label: str,
    color: str,
) -> None:
    points = curves.get(key, [])
    if isinstance(points, Sequence) and points:
        axis.plot(
            [
                _number(point, "update")
                for point in points
                if isinstance(point, Mapping)
            ],
            [_number(point, "value") for point in points if isinstance(point, Mapping)],
            marker="o",
            color=color,
            label=label,
        )


def _plot_split_comparison(run_directory: Path, report: Mapping[str, object]) -> None:
    splits = _object(report, "splits")
    names = ("training", "inner_validation", "holdout")
    labels = ("train", "inner val", "holdout")
    metrics = (
        ("mean_survival_frames", "mean"),
        ("median_survival_frames", "median"),
        ("p10_survival_frames", "p10"),
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    width = 0.24
    positions = list(range(len(names)))
    for offset, (key, label) in enumerate(metrics):
        values = [_number(_object(splits, name), key) for name in names]
        axis.bar(
            [position + (offset - 1) * width for position in positions],
            values,
            width=width,
            label=label,
        )
    axis.set_xticks(positions, labels)
    axis.set_title("Final survival by data split")
    axis.set_ylabel("survival frames")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_directory / "split_comparison.png", dpi=140)
    plt.close(figure)


def _plot_per_seed(run_directory: Path, report: Mapping[str, object]) -> None:
    splits = _object(report, "splits")
    training = _object(splits, "training")
    holdout = _object(splits, "holdout")
    training_values = _integer_tuple(training, "survival_frames")
    holdout_values = _integer_tuple(holdout, "survival_frames")
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(range(len(training_values)), training_values, color="C0", label="train")
    holdout_start = len(training_values) + 2
    axis.bar(
        range(holdout_start, holdout_start + len(holdout_values)),
        holdout_values,
        color="C3",
        label="locked holdout",
    )
    axis.set_title("Final survival for every NG seed")
    axis.set_xlabel("seed order within split")
    axis.set_ylabel("survival frames")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_directory / "per_seed_survival.png", dpi=140)
    plt.close(figure)


def _plot_diagnostics(run_directory: Path, report: Mapping[str, object]) -> None:
    curves = _object(report, "curves")
    names = (
        ("rollout_reward", "rollout reward"),
        ("entropy", "policy entropy"),
        ("value_loss", "value loss"),
        ("approx_kl", "approx KL"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    for axis, (key, label) in zip(axes.flat, names, strict=True):
        points = curves.get(key, [])
        if isinstance(points, Sequence) and points:
            axis.plot(
                [
                    _number(point, "update")
                    for point in points
                    if isinstance(point, Mapping)
                ],
                [
                    _number(point, "value")
                    for point in points
                    if isinstance(point, Mapping)
                ],
                color="C2",
            )
        axis.set_title(label)
        axis.set_xlabel("update")
        axis.grid(alpha=0.25)
    figure.suptitle("PPO training diagnostics")
    figure.tight_layout()
    figure.savefig(run_directory / "training_diagnostics.png", dpi=140)
    plt.close(figure)


def _markdown_report(report: Mapping[str, object]) -> str:
    manifest = _object(report, "manifest")
    splits = _object(report, "splits")
    comparison = _object(report, "comparison")
    trend = _object(report, "trend")
    provenance = _object(report, "provenance")
    train = _object(splits, "training")
    holdout = _object(splits, "holdout")
    ppo_run = provenance.get("ppo_run")
    if not isinstance(ppo_run, Mapping):
        ppo_run = {}
    lines = [
        "# Dodge NG baseline report",
        "",
        f"Manifest: `{manifest['manifest_id']}`  ",
        f"Manifest SHA-256: `{manifest['manifest_sha256']}`  ",
        f"Sample space: {manifest['sample_count']} seeds "
        f"({manifest['training_count']} train / {manifest['holdout_count']} holdout)",
        "",
        "## Final performance",
        "",
        "| Split | Mean | Median | P10 | Worst | Best | Horizon completion |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, split in (
        ("Training", train),
        ("Inner validation", _object(splits, "inner_validation")),
        ("Locked holdout", holdout),
    ):
        lines.append(
            f"| {label} | {_number(split, 'mean_survival_frames'):.1f} | "
            f"{_number(split, 'median_survival_frames'):.1f} | "
            f"{_number(split, 'p10_survival_frames'):.1f} | "
            f"{_number(split, 'worst_survival_frames'):.1f} | "
            f"{_number(split, 'best_survival_frames'):.1f} | "
            f"{_number(split, 'horizon_completion_fraction'):.1%} |"
        )
    lines.extend(
        [
            "",
            "Train minus holdout mean: "
            f"**{_number(comparison, 'mean_train_minus_holdout'):.1f} frames**.",
            "Train minus holdout p10: "
            f"**{_number(comparison, 'p10_train_minus_holdout'):.1f} frames**.",
            "",
            "## Learning trend",
            "",
        ]
    )
    _append_trend_markdown(
        lines, trend, "training_mean_survival", "Training evaluation"
    )
    _append_trend_markdown(
        lines, trend, "inner_validation_mean_survival", "Inner validation"
    )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            *[f"- [{name}]({name})" for name in PLOT_NAMES],
            "- [report.json](report.json)",
            "- [run.json](run.json)",
            "- [metrics.jsonl](metrics.jsonl)",
            "",
            f"Updates: {ppo_run.get('updates_completed', 'unknown')}  ",
            f"Global steps: {ppo_run.get('global_step', 'unknown')}",
            "",
        ]
    )
    return "\n".join(lines)


def _append_trend_markdown(
    lines: list[str], trend: Mapping[str, object], key: str, label: str
) -> None:
    value = trend.get(key)
    if not isinstance(value, Mapping):
        lines.append(f"{label}: no checkpoint curve recorded.")
        return
    lines.append(
        f"{label}: {_number(value, 'first_value'):.1f} → "
        f"{_number(value, 'last_value'):.1f} frames "
        f"({_number(value, 'gain'):+.1f}); best {_number(value, 'best_value'):.1f} "
        f"at update {int(_number(value, 'best_update'))}."
    )


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON object {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON value must be an object: {path}")
    return dict(value)


def _optional_object(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return _load_object(path)


def _load_metrics(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read metrics {path}: {error}") from error
    metrics: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid metrics line {line_number}: {error}") from error
        if not isinstance(value, Mapping):
            raise ValueError(f"metrics line {line_number} must be an object")
        metrics.append(dict(value))
    return metrics


def _object(value: Mapping[str, object], key: str) -> dict[str, object]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ValueError(f"missing object field: {key}")
    return dict(nested)


def _number(value: Mapping[str, object], key: str) -> float:
    field = value.get(key)
    if isinstance(field, bool) or not isinstance(field, (int, float)):
        raise ValueError(f"{key} must be numeric")
    result = float(field)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _integer_tuple(value: Mapping[str, object], key: str) -> tuple[int, ...]:
    field = value.get(key)
    if isinstance(field, (str, bytes)) or not isinstance(field, Sequence):
        raise ValueError(f"{key} must be an integer list")
    result = tuple(field)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise ValueError(f"{key} must be an integer list")
    return result


def _boolean_tuple(value: Mapping[str, object], key: str) -> tuple[bool, ...]:
    field = value.get(key)
    if isinstance(field, (str, bytes)) or not isinstance(field, Sequence):
        raise ValueError(f"{key} must be a boolean list")
    result = tuple(field)
    if any(not isinstance(item, bool) for item in result):
        raise ValueError(f"{key} must be a boolean list")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-report")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    arguments = parser.parse_args(argv)
    try:
        report = build_report(
            arguments.run_directory, load_manifest(arguments.manifest)
        )
    except (OSError, ValueError) as error:
        print(f"dodge-ng-report: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_directory": str(arguments.run_directory),
                "holdout_mean_survival_frames": _number(
                    _object(_object(report, "splits"), "holdout"),
                    "mean_survival_frames",
                ),
                "report": str(arguments.run_directory / "REPORT.md"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
