from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dodge.control import ControlInputError, ControlRuntimeError
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest
from dodge.ng.report import summarize_evaluation
from dodge.rl.ppo import DodgeActorCriticCNN, PPOConfig, evaluate_policy

DEFAULT_SCRATCH_DIRECTORY = Path("history/dodge/ng/p2-ppo-scratch-20260912")
DEFAULT_WARM_DIRECTORY = Path("history/dodge/ng/p2-ppo-bc-warm-20260912")
DEFAULT_OUTPUT_DIRECTORY = Path("history/dodge/ng/p2-ppo-comparison-20260912")
INNER_VALIDATION_COUNT = 10
COMPARISON_SCHEMA_VERSION = 1


def compare_ppo_runs(
    scratch_directory: Path,
    warm_directory: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    runs = {
        "scratch": _load_selected_run(scratch_directory, manifest),
        "bc_warm_start": _load_selected_run(warm_directory, manifest),
    }
    _assert_matching_configs(runs)
    output_directory.mkdir(parents=True, exist_ok=True)
    comparison: dict[str, object] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "kind": "dodge_ng_ppo_comparison",
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.sha256,
        "selection_rule": "best inner-validation mean survival",
        "holdout_rule": "evaluate only after checkpoint selection",
        "runs": runs,
        "delta": _delta(runs),
        "sample_efficiency": _sample_efficiency(runs),
        "plots": ["ppo_comparison.png"],
    }
    _write_json(output_directory / "comparison.json", comparison)
    _plot_comparison(output_directory, runs)
    (output_directory / "COMPARISON.md").write_text(
        _markdown_comparison(comparison), encoding="utf-8"
    )
    return comparison


def _load_selected_run(
    run_directory: Path, manifest: SeedManifest
) -> dict[str, object]:
    run = _load_object(run_directory / "run.json")
    config_value = _object(run, "config")
    config = _ppo_config(config_value)
    if config.training_seed_manifest != manifest.sha256:
        raise ControlInputError(
            f"run is bound to a different NG manifest: {run_directory}"
        )
    if tuple(config.training_seeds) != manifest.training_seeds:
        raise ControlInputError(
            f"run training seeds do not match manifest: {run_directory}"
        )
    checkpoint_name = run.get("best_checkpoint")
    if not isinstance(checkpoint_name, str):
        raise ControlInputError(
            f"run has no inner-selected checkpoint: {run_directory}"
        )
    checkpoint = run_directory / checkpoint_name
    model, checkpoint_payload = _load_checkpoint(checkpoint, config)
    selected: dict[str, object] = {}
    for name, seeds in (
        ("inner_validation", manifest.training_seeds[:INNER_VALIDATION_COUNT]),
        ("training", manifest.training_seeds),
        ("holdout", manifest.holdout_seeds),
    ):
        result = evaluate_policy(
            model,
            config,
            seeds,
            temporary_root=run_directory / ".selection-eval",
        )
        selected[name] = summarize_evaluation(result.to_json())
    metrics = _load_metrics(run_directory / "metrics.jsonl")
    return {
        "run_directory": str(run_directory),
        "run_kind": run.get("kind"),
        "initialization": run.get("initialization"),
        "checkpoint": checkpoint_name,
        "selected_update": checkpoint_payload["updates_completed"],
        "best_inner_validation": checkpoint_payload["best_validation"],
        "selected": selected,
        "latest": {
            "training": _optional_evaluation(run, "final_training_evaluation"),
            "holdout": _optional_evaluation(run, "final_evaluation"),
        },
        "curve": _inner_curve(metrics),
        "global_step": run.get("global_step"),
        "config": config.to_json(),
    }


def _load_checkpoint(
    checkpoint: Path, config: PPOConfig
) -> tuple[DodgeActorCriticCNN, Mapping[str, object]]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError) as error:
        raise ControlRuntimeError(
            f"could not load PPO checkpoint {checkpoint}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ControlRuntimeError(f"PPO checkpoint is not an object: {checkpoint}")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ControlRuntimeError(f"PPO checkpoint has no model state: {checkpoint}")
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ControlRuntimeError(f"PPO checkpoint has no config: {checkpoint}")
    if dict(checkpoint_config) != config.to_json():
        raise ControlInputError(
            f"checkpoint config does not match run record: {checkpoint}"
        )
    model = DodgeActorCriticCNN()
    try:
        model.load_state_dict(state)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ControlRuntimeError(
            f"PPO checkpoint model is incompatible: {error}"
        ) from error
    model.eval()
    return model, payload


def _ppo_config(value: Mapping[str, object]) -> PPOConfig:
    values = dict(value)
    training_seeds = values.get("training_seeds")
    if not isinstance(training_seeds, Sequence) or isinstance(
        training_seeds, (str, bytes)
    ):
        raise ControlInputError("PPO run config has invalid training seeds")
    values["training_seeds"] = tuple(int(seed) for seed in training_seeds)
    try:
        config = PPOConfig(**values)  # type: ignore[arg-type]
        config.validate()
    except (TypeError, ValueError) as error:
        raise ControlInputError(f"PPO run config is invalid: {error}") from error
    return config


def _assert_matching_configs(runs: Mapping[str, object]) -> None:
    configs = [
        _object(_object(runs, label), "config")
        for label in ("scratch", "bc_warm_start")
    ]
    if configs[0] != configs[1]:
        raise ControlInputError("matched PPO comparison configs differ")


def _delta(runs: Mapping[str, object]) -> dict[str, float]:
    scratch = _selected_splits(_object(runs, "scratch"))
    warm = _selected_splits(_object(runs, "bc_warm_start"))
    return {
        "inner_mean_survival_frames": _mean(warm, "inner_validation")
        - _mean(scratch, "inner_validation"),
        "training_mean_survival_frames": _mean(warm, "training")
        - _mean(scratch, "training"),
        "holdout_mean_survival_frames": _mean(warm, "holdout")
        - _mean(scratch, "holdout"),
        "holdout_p10_survival_frames": _number(
            _split(warm, "holdout"), "p10_survival_frames"
        )
        - _number(_split(scratch, "holdout"), "p10_survival_frames"),
    }


def _sample_efficiency(runs: Mapping[str, object]) -> dict[str, object]:
    thresholds = (200.0, 250.0, 300.0, 350.0)
    result: dict[str, object] = {}
    for label in ("scratch", "bc_warm_start"):
        curve = _object(runs, label).get("curve", [])
        if not isinstance(curve, Sequence):
            raise ControlRuntimeError(f"run curve is invalid: {label}")
        result[label] = {
            f"first_inner_at_least_{int(threshold)}": _first_at_least(curve, threshold)
            for threshold in thresholds
        }
    return result


def _first_at_least(curve: Sequence[object], threshold: float) -> int | None:
    for point in curve:
        if isinstance(point, Mapping) and _number(point, "value") >= threshold:
            return int(_number(point, "update"))
    return None


def _plot_comparison(output_directory: Path, runs: Mapping[str, object]) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, color in (("scratch", "C0"), ("bc_warm_start", "C1")):
        curve = _object(runs, label).get("curve", [])
        if isinstance(curve, Sequence):
            points = [point for point in curve if isinstance(point, Mapping)]
            axis.plot(
                [_number(point, "update") for point in points],
                [_number(point, "value") for point in points],
                marker="o",
                color=color,
                label=label,
            )
    axis.set_title("PPO inner-validation survival: scratch vs BC warm start")
    axis.set_xlabel("update")
    axis.set_ylabel("survival frames")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "ppo_comparison.png", dpi=140)
    plt.close(figure)


def _markdown_comparison(comparison: Mapping[str, object]) -> str:
    runs = _object(comparison, "runs")
    delta = _object(comparison, "delta")
    lines = [
        "# Dodge NG PPO comparison",
        "",
        f"Manifest SHA-256: `{comparison['manifest_sha256']}`",
        "",
        "Checkpoint selection used only the first 10 training seeds. The locked "
        "holdout was evaluated after selection.",
        "",
        "| Run | Selected update | Inner mean | Train mean | Holdout mean | "
        "Holdout p10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in ("scratch", "bc_warm_start"):
        run = _object(runs, label)
        selected = _object(run, "selected")
        inner = _split(selected, "inner_validation")
        training = _split(selected, "training")
        holdout = _split(selected, "holdout")
        lines.append(
            f"| {label} | {run['selected_update']} | "
            f"{_number(inner, 'mean_survival_frames'):.1f} | "
            f"{_number(training, 'mean_survival_frames'):.1f} | "
            f"{_number(holdout, 'mean_survival_frames'):.1f} | "
            f"{_number(holdout, 'p10_survival_frames'):.1f} |"
        )
    lines.extend(
        [
            "",
            f"Selected holdout mean delta (warm - scratch): "
            f"**{delta['holdout_mean_survival_frames']:+.1f} frames**.",
            f"Selected inner mean delta (warm - scratch): "
            f"**{delta['inner_mean_survival_frames']:+.1f} frames**.",
            "",
            "## Artifacts",
            "",
            "- [ppo_comparison.png](ppo_comparison.png)",
            "- [comparison.json](comparison.json)",
            "",
        ]
    )
    return "\n".join(lines)


def _inner_curve(metrics: Sequence[Mapping[str, object]]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for metric in metrics:
        validation = metric.get("validation")
        if isinstance(validation, Mapping) and "mean_survival_frames" in validation:
            points.append(
                {
                    "update": _number(metric, "update"),
                    "value": _number(validation, "mean_survival_frames"),
                }
            )
    return points


def _load_metrics(path: Path) -> list[Mapping[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ControlRuntimeError(f"could not read metrics {path}: {error}") from error
    if not all(isinstance(value, Mapping) for value in values):
        raise ControlRuntimeError(f"metrics must contain objects: {path}")
    return [value for value in values if isinstance(value, Mapping)]


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlRuntimeError(f"could not read JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ControlRuntimeError(f"JSON value must be an object: {path}")
    return value


def _optional_evaluation(
    run: Mapping[str, object], key: str
) -> dict[str, object] | None:
    value = run.get(key)
    if not isinstance(value, Mapping):
        return None
    return summarize_evaluation(value)


def _selected_splits(run: Mapping[str, object]) -> Mapping[str, object]:
    return _object(run, "selected")


def _split(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    return _object(value, name)


def _mean(value: Mapping[str, object], name: str) -> float:
    return _number(_split(value, name), "mean_survival_frames")


def _object(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ControlRuntimeError(f"comparison field is not an object: {key}")
    return nested


def _number(value: Mapping[str, object], key: str) -> float:
    field = value.get(key)
    if isinstance(field, bool) or not isinstance(field, (int, float)):
        raise ControlRuntimeError(f"comparison field is not numeric: {key}")
    if not math.isfinite(float(field)):
        raise ControlRuntimeError(f"comparison field is not finite: {key}")
    return float(field)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dodge-ng-compare")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH_DIRECTORY)
    parser.add_argument("--warm-start", type=Path, default=DEFAULT_WARM_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = compare_ppo_runs(
            arguments.scratch,
            arguments.warm_start,
            arguments.manifest,
            arguments.output_dir,
        )
    except (ControlInputError, ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-ng-compare: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output_directory": str(arguments.output_dir),
                "holdout_delta": result["delta"]["holdout_mean_survival_frames"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
