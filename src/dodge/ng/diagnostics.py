from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from dodge.control import ControlRuntimeError
from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import NativeBatchEnvironment
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest
from dodge.ng.report import summarize_evaluation

DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[3]
    / "history"
    / "dodge"
    / "ng"
    / "p1-action-controls"
)
ActionExecution = Literal["serial", "parallel"]


def evaluate_fixed_action(
    action_index: int,
    seeds: Sequence[int],
    *,
    step_frames: int = 4,
    max_episode_steps: int = 2_000,
    execution: ActionExecution = "parallel",
) -> dict[str, object]:
    """Evaluate one constant action with the same native reward path as PPO."""
    if not 0 <= action_index < len(ACTION_CHOICES):
        raise ValueError(f"unknown action index: {action_index}")
    if not seeds:
        raise ValueError("fixed-action evaluation requires at least one seed")
    if max_episode_steps < 1:
        raise ValueError("maximum episode steps must be positive")
    environment = NativeBatchEnvironment(
        step_frames=step_frames,
        execution=execution,
        full_state=False,
        pixels=False,
        board=False,
    )
    active = [True] * len(seeds)
    survival_frames = [0.0] * len(seeds)
    terminated = [False] * len(seeds)
    try:
        environment.reset_batch(np.asarray(seeds, dtype=np.uint32))
        for _ in range(max_episode_steps):
            if not any(active):
                break
            actions = np.full(len(seeds), action_index, dtype=np.uint8)
            result = environment.step_batch(actions)
            completed: list[int] = []
            for lane, is_active in enumerate(active):
                if not is_active:
                    continue
                survival_frames[lane] += float(result.rewards[lane])
                if bool(result.done[lane]):
                    active[lane] = False
                    terminated[lane] = True
                    completed.append(lane)
            if completed:
                environment.reset_lanes(
                    np.asarray(completed, dtype=np.uint32),
                    np.zeros(len(completed), dtype=np.uint32),
                )
        evaluation = {
            "seeds": list(seeds),
            "survival_frames": [int(round(value)) for value in survival_frames],
            "terminated": terminated,
        }
    finally:
        environment.close()
    return evaluation


def build_action_diagnostic(
    output_directory: Path,
    manifest: SeedManifest,
    *,
    step_frames: int = 4,
    max_episode_steps: int = 2_000,
    execution: ActionExecution = "parallel",
) -> dict[str, object]:
    """Write constant-action train/holdout controls for all nine actions."""
    manifest.validate()
    actions: list[dict[str, object]] = []
    for action_index, action in enumerate(ACTION_CHOICES):
        training = evaluate_fixed_action(
            action_index,
            manifest.training_seeds,
            step_frames=step_frames,
            max_episode_steps=max_episode_steps,
            execution=execution,
        )
        holdout = evaluate_fixed_action(
            action_index,
            manifest.holdout_seeds,
            step_frames=step_frames,
            max_episode_steps=max_episode_steps,
            execution=execution,
        )
        actions.append(
            {
                "action_index": action_index,
                "action": action,
                "training": summarize_evaluation(training),
                "holdout": summarize_evaluation(holdout),
            }
        )
    diagnostic: dict[str, object] = {
        "schema_version": 1,
        "kind": "dodge_ng_fixed_action_diagnostic",
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.sha256,
        "sample_count": manifest.sample_count,
        "training_count": len(manifest.training_seeds),
        "holdout_count": len(manifest.holdout_seeds),
        "step_frames": step_frames,
        "max_episode_steps": max_episode_steps,
        "execution": execution,
        "actions": actions,
        "selection_policy": "diagnostic_only; no action selected from holdout",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "action-controls.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "ACTION_CONTROLS.md").write_text(
        _markdown(diagnostic), encoding="utf-8"
    )
    return diagnostic


def _markdown(diagnostic: dict[str, object]) -> str:
    actions = diagnostic["actions"]
    if not isinstance(actions, Sequence):
        raise ValueError("diagnostic actions must be a sequence")
    lines = [
        "# Dodge NG fixed-action controls",
        "",
        f"Manifest: `{diagnostic['manifest_id']}`  ",
        f"Manifest SHA-256: `{diagnostic['manifest_sha256']}`  ",
        f"Seeds: {diagnostic['training_count']} train / "
        f"{diagnostic['holdout_count']} locked holdout  ",
        f"Step frames: {diagnostic['step_frames']}  ",
        f"Maximum episode steps: {diagnostic['max_episode_steps']}",
        "",
        "| Action | Train mean | Train p10 | Holdout mean | Holdout p10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in actions:
        if not isinstance(item, dict):
            raise ValueError("diagnostic action entry must be an object")
        training = item["training"]
        holdout = item["holdout"]
        if not isinstance(training, dict) or not isinstance(holdout, dict):
            raise ValueError("diagnostic split entry must be an object")
        lines.append(
            f"| `{item['action']}` | "
            f"{float(training['mean_survival_frames']):.1f} | "
            f"{float(training['p10_survival_frames']):.1f} | "
            f"{float(holdout['mean_survival_frames']):.1f} | "
            f"{float(holdout['p10_survival_frames']):.1f} |"
        )
    lines.extend(
        [
            "",
            "This is an action-authority diagnostic. It does not select a policy",
            "and does not use holdout results for training or tuning.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-diagnose-actions")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=2_000)
    parser.add_argument(
        "--execution", choices=("serial", "parallel"), default="parallel"
    )
    arguments = parser.parse_args(argv)
    try:
        diagnostic = build_action_diagnostic(
            arguments.output_dir,
            load_manifest(arguments.manifest),
            step_frames=arguments.step_frames,
            max_episode_steps=arguments.max_episode_steps,
            execution=arguments.execution,
        )
    except (ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-ng-diagnose-actions: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output_directory": str(arguments.output_dir),
                "manifest_sha256": diagnostic["manifest_sha256"],
                "action_count": len(diagnostic["actions"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
