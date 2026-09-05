"""Reproducible long-run waypoint DQN training preset."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dodge.control import PROJECT_ROOT, ControlRuntimeError
from dodge.ng.dqn import DQNConfig, train_waypoint_dqn
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest

HPO_SOURCE: Final[str] = (
    "history/dodge/ng/waypoint-hpo-20260904-v1/hpo.json"
)
HPO_TRIAL: Final[int] = 5
HPO_PARAMETERS: Final[dict[str, object]] = {
    "learning_rate": 0.00024074661619742944,
    "weight_decay": 0.0001,
    "target_update_interval": 500,
    "batch_size": 128,
    "n_step": 5,
    "epsilon_decay_steps": 50_000,
    "epsilon_final": 0.05,
}
DEFAULT_RUN_DIRECTORY = (
    PROJECT_ROOT
    / "history"
    / "dodge"
    / "ng"
    / "waypoint-dqn-overnight-lives-20260904"
)
DEFAULT_TOTAL_STEPS: Final[int] = 1_000_000
DEFAULT_BATCH_SIZE: Final[int] = 1_024
DEFAULT_TRAINING_LIVES: Final[int] = 3
DEFAULT_LIFE_LOSS_PENALTY: Final[float] = -64.0
DEFAULT_LEARNER_SEED: Final[int] = 2_026_0903
DEFAULT_NATIVE_LANES: Final[int] = 32
DEFAULT_CHECKPOINT_EVERY: Final[int] = 10_000
DEFAULT_EVAL_EVERY: Final[int] = 10_000


@dataclass(frozen=True, slots=True)
class OvernightConfig:
    """Controls for one resumable, training-only overnight campaign."""

    manifest_path: Path = DEFAULT_MANIFEST_PATH
    run_directory: Path = DEFAULT_RUN_DIRECTORY
    total_steps: int = DEFAULT_TOTAL_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    training_lives: int = DEFAULT_TRAINING_LIVES
    life_loss_penalty: float = DEFAULT_LIFE_LOSS_PENALTY
    native_lanes: int = DEFAULT_NATIVE_LANES
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    eval_every: int = DEFAULT_EVAL_EVERY
    seed: int = DEFAULT_LEARNER_SEED
    device: str = "cpu"
    resume: bool = False

    def validate(self) -> None:
        if self.total_steps < 1:
            raise ValueError("overnight total steps must be positive")
        if self.batch_size < 1:
            raise ValueError("overnight batch size must be positive")
        if self.native_lanes < 1:
            raise ValueError("overnight lane count must be positive")
        if self.checkpoint_every < 1 or self.eval_every < 1:
            raise ValueError("overnight intervals must be positive")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("overnight device must be cpu, cuda, or auto")

    def dqn_config(self) -> DQNConfig:
        return DQNConfig(
            total_steps=self.total_steps,
            batch_size=self.batch_size,
            replay_capacity=100_000,
            learning_rate=float(HPO_PARAMETERS["learning_rate"]),
            weight_decay=float(HPO_PARAMETERS["weight_decay"]),
            gamma=0.99,
            n_step=int(HPO_PARAMETERS["n_step"]),
            warmup_steps=2_000,
            train_frequency=1,
            target_update_interval=int(HPO_PARAMETERS["target_update_interval"]),
            hidden_size=256,
            grid_spacing=32,
            hold_decisions=8,
            step_frames=4,
            max_episode_steps=2_000,
            native_lanes=self.native_lanes,
            native_execution="parallel",
            reset_mode="native-startup",
            training_lives=self.training_lives,
            life_loss_penalty=self.life_loss_penalty,
            epsilon_decay_steps=int(HPO_PARAMETERS["epsilon_decay_steps"]),
            epsilon_final=float(HPO_PARAMETERS["epsilon_final"]),
            checkpoint_every=self.checkpoint_every,
            eval_every=self.eval_every,
            seed=self.seed,
            device=self.device,
        )


def _metadata(config: OvernightConfig, manifest: SeedManifest) -> dict[str, object]:
    dqn = config.dqn_config()
    return {
        "schema_version": 1,
        "kind": "dodge_ng_waypoint_dqn_overnight",
        "host": platform.node(),
        "hpo_source": HPO_SOURCE,
        "hpo_trial": HPO_TRIAL,
        "inherited_hpo_parameters": dict(HPO_PARAMETERS),
        "declared_overrides": {
            "total_steps": config.total_steps,
            "batch_size": config.batch_size,
            "training_lives": config.training_lives,
            "life_loss_penalty": config.life_loss_penalty,
            "checkpoint_every": config.checkpoint_every,
            "eval_every": config.eval_every,
        },
        "manifest_sha256": manifest.sha256,
        "training_seeds": list(manifest.training_seeds),
        "evaluation_mode": {
            "training_split": False,
            "holdout_split": False,
            "inner_training_seeds": True,
        },
        "config": dqn.to_json(),
    }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_metadata(path: Path, expected: dict[str, object]) -> None:
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ControlRuntimeError(
                f"overnight metadata is unreadable: {path}"
            ) from error
        if not isinstance(existing, dict):
            raise ControlRuntimeError("overnight metadata must be a JSON object")
        for field in (
            "kind",
            "manifest_sha256",
            "hpo_trial",
            "inherited_hpo_parameters",
        ):
            if existing.get(field) != expected.get(field):
                raise ValueError(
                    f"overnight metadata field {field} does not match requested run"
                )
        old_config = existing.get("config")
        new_config = expected.get("config")
        if not isinstance(old_config, dict) or not isinstance(new_config, dict):
            raise ControlRuntimeError("overnight metadata config is invalid")
        for field, old_value in old_config.items():
            new_value = new_config.get(field)
            if field == "total_steps":
                if not isinstance(old_value, int) or not isinstance(new_value, int):
                    raise ControlRuntimeError(
                        "overnight total-step metadata is invalid"
                    )
                if old_value > new_value:
                    raise ValueError(
                        "overnight resume cannot reduce the configured total steps"
                    )
            elif old_value != new_value:
                raise ValueError(
                    f"overnight metadata config field {field} does not match"
                )
        return
    _write_json(path, expected)


def run_overnight(config: OvernightConfig) -> dict[str, object]:
    config.validate()
    manifest = load_manifest(config.manifest_path)
    manifest.validate()
    dqn_config = config.dqn_config()
    dqn_config.validate()
    if len(manifest.training_seeds) < config.native_lanes:
        raise ValueError("overnight lane count exceeds training seed count")
    config.run_directory.mkdir(parents=True, exist_ok=True)
    _prepare_metadata(
        config.run_directory / "overnight.json",
        _metadata(config, manifest),
    )
    return train_waypoint_dqn(
        dqn_config,
        config.run_directory,
        manifest,
        resume=config.resume,
        evaluate_holdout=False,
        evaluate_training=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-overnight")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIRECTORY)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--training-lives",
        type=int,
        default=DEFAULT_TRAINING_LIVES,
    )
    parser.add_argument(
        "--life-loss-penalty",
        type=float,
        default=DEFAULT_LIFE_LOSS_PENALTY,
    )
    parser.add_argument("--native-lanes", type=int, default=DEFAULT_NATIVE_LANES)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
    )
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--seed", type=int, default=DEFAULT_LEARNER_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    arguments = parser.parse_args(argv)
    config = OvernightConfig(
        manifest_path=arguments.manifest,
        run_directory=arguments.run_dir,
        total_steps=arguments.total_steps,
        batch_size=arguments.batch_size,
        training_lives=arguments.training_lives,
        life_loss_penalty=arguments.life_loss_penalty,
        native_lanes=arguments.native_lanes,
        checkpoint_every=arguments.checkpoint_every,
        eval_every=arguments.eval_every,
        seed=arguments.seed,
        device=arguments.device,
        resume=arguments.resume,
    )
    try:
        started = time.monotonic()
        run = run_overnight(config)
    except (ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-ng-overnight: {error}")
        return 1
    print(
        json.dumps(
            {
                "run_directory": str(config.run_directory),
                "updates_completed": run["updates_completed"],
                "stopped_early": run["stopped_early"],
                "wall_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
