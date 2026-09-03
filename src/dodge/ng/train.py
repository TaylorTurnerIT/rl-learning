from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from dodge.control import PROJECT_ROOT, ControlInputError, ControlRuntimeError
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest
from dodge.ng.report import build_report
from dodge.rl.ppo import NativeExecution, PPOConfig, train_ppo

DEFAULT_RUN_DIRECTORY = PROJECT_ROOT / "history" / "dodge" / "ng" / "baseline-p1"
DEFAULT_BASELINE_SEED = 2_026_0903
INNER_VALIDATION_COUNT = 10


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    run_directory: Path = DEFAULT_RUN_DIRECTORY
    resume: bool = False
    updates: int = 200
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 2.5e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    neutral_bonus: float = 0.02
    stability_bonus_cap: float = 1.0
    step_frames: int = 4
    max_episode_steps: int = 2_000
    environment_restarts: int = 3
    checkpoint_every: int = 25
    eval_every: int = 25
    seed: int = DEFAULT_BASELINE_SEED
    device: str = "auto"
    native_lanes: int = 32
    native_execution: NativeExecution = "parallel"

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        value["manifest_path"] = str(self.manifest_path)
        value["run_directory"] = str(self.run_directory)
        return value

    def ppo_config(self, manifest: SeedManifest) -> PPOConfig:
        return PPOConfig(
            updates=self.updates,
            rollout_steps=self.rollout_steps,
            update_epochs=self.update_epochs,
            minibatch_size=self.minibatch_size,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_coef=self.clip_coef,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
            max_grad_norm=self.max_grad_norm,
            neutral_bonus=self.neutral_bonus,
            stability_bonus_cap=self.stability_bonus_cap,
            step_frames=self.step_frames,
            max_episode_steps=self.max_episode_steps,
            environment_restarts_per_rollout=self.environment_restarts,
            checkpoint_every=self.checkpoint_every,
            eval_every=self.eval_every,
            seed=self.seed,
            device=self.device,
            backend="native",
            native_lanes=self.native_lanes,
            native_execution=self.native_execution,
            observation_mode="board",
            training_seeds=manifest.training_seeds,
            training_seed_manifest=manifest.sha256,
        )


def run_baseline(config: BaselineConfig) -> dict[str, object]:
    manifest = load_manifest(config.manifest_path)
    ppo_config = config.ppo_config(manifest)
    ppo_config.validate()
    inner_validation_seeds = manifest.training_seeds[:INNER_VALIDATION_COUNT]
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    record = train_ppo(
        ppo_config,
        config.run_directory,
        resume=config.resume,
        validation_seeds=inner_validation_seeds,
        training_evaluation_seeds=manifest.training_seeds,
        evaluation_seeds=manifest.holdout_seeds,
    )
    wall_seconds = time.monotonic() - started
    ng_record = {
        "schema_version": 1,
        "kind": "dodge_ng_baseline_run",
        "started_at_utc": started_at,
        "training_wall_seconds": wall_seconds,
        "host": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.sha256,
        "sample_count": manifest.sample_count,
        "training_seeds": list(manifest.training_seeds),
        "inner_validation_seeds": list(inner_validation_seeds),
        "holdout_seeds": list(manifest.holdout_seeds),
        "legacy_inputs": "none",
        "baseline_config": config.to_json(),
    }
    _write_json(config.run_directory / "ng-run.json", ng_record)
    report = build_report(config.run_directory, manifest)
    return {**record, "ng_run": ng_record, "report": report}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dodge-ng-train")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIRECTORY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--updates", type=_positive_int, default=200)
    parser.add_argument("--rollout-steps", type=_positive_int, default=256)
    parser.add_argument("--update-epochs", type=_positive_int, default=4)
    parser.add_argument("--minibatch-size", type=_positive_int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--neutral-bonus", type=float, default=0.02)
    parser.add_argument("--stability-bonus-cap", type=float, default=1.0)
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=_positive_int, default=2_000)
    parser.add_argument("--environment-restarts", type=_nonnegative_int, default=3)
    parser.add_argument("--checkpoint-every", type=_positive_int, default=25)
    parser.add_argument("--eval-every", type=_nonnegative_int, default=25)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASELINE_SEED)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--native-lanes", type=_positive_int, default=32)
    parser.add_argument(
        "--native-execution", choices=("serial", "parallel"), default="parallel"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = BaselineConfig(
        manifest_path=arguments.manifest,
        run_directory=arguments.run_dir,
        resume=arguments.resume,
        updates=arguments.updates,
        rollout_steps=arguments.rollout_steps,
        update_epochs=arguments.update_epochs,
        minibatch_size=arguments.minibatch_size,
        learning_rate=arguments.learning_rate,
        gamma=arguments.gamma,
        gae_lambda=arguments.gae_lambda,
        clip_coef=arguments.clip_coef,
        entropy_coef=arguments.entropy_coef,
        value_coef=arguments.value_coef,
        max_grad_norm=arguments.max_grad_norm,
        neutral_bonus=arguments.neutral_bonus,
        stability_bonus_cap=arguments.stability_bonus_cap,
        step_frames=arguments.step_frames,
        max_episode_steps=arguments.max_episode_steps,
        environment_restarts=arguments.environment_restarts,
        checkpoint_every=arguments.checkpoint_every,
        eval_every=arguments.eval_every,
        seed=arguments.seed,
        device=arguments.device,
        native_lanes=arguments.native_lanes,
        native_execution=arguments.native_execution,
    )
    try:
        result = run_baseline(config)
    except (ControlInputError, ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-ng-train: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_directory": str(config.run_directory),
                "manifest_sha256": result["ng_run"]["manifest_sha256"],
                "training_wall_seconds": result["ng_run"]["training_wall_seconds"],
                "report": str(config.run_directory / "REPORT.md"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
