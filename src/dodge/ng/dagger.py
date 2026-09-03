from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from dodge.control import PROJECT_ROOT, ControlInputError, ControlRuntimeError
from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import NativeBatchEnvironment
from dodge.ng.bc import BCConfig, run_behavior_cloning
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest
from dodge.ng.teacher import (
    ACTION_COUNT,
    BOARD_SHAPE,
    CounterfactualCache,
    TeacherDataset,
    _merge_results,
    load_teacher_dataset,
    save_teacher_dataset,
)
from dodge.rl.ppo import DodgeActorCriticCNN

DAGGER_SCHEMA_VERSION = 1
DEFAULT_BASE_TEACHER_DATA = (
    PROJECT_ROOT
    / "history"
    / "dodge"
    / "ng"
    / "p2-teacher-planner-v3"
    / "teacher-data.npz"
)
DEFAULT_LEARNER_CHECKPOINT = (
    PROJECT_ROOT
    / "history"
    / "dodge"
    / "ng"
    / "p2-ppo-bc-warm-20260912"
    / "checkpoint-best.pt"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "history" / "dodge" / "ng" / "dagger-p2-r1"
DEFAULT_LEARNER_SEED = 2_026_0913
DEFAULT_LOOKAHEAD_STEPS = 64
DEFAULT_STATES_PER_SEED = 32
DEFAULT_NATIVE_LANES = 32
DEFAULT_MAX_COLLECTOR_STEPS = 8_000


@dataclass(frozen=True, slots=True)
class DaggerConfig:
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    base_teacher_data_path: Path = DEFAULT_BASE_TEACHER_DATA
    learner_checkpoint: Path = DEFAULT_LEARNER_CHECKPOINT
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
    round_index: int = 1
    states_per_seed: int = DEFAULT_STATES_PER_SEED
    lookahead_steps: int = DEFAULT_LOOKAHEAD_STEPS
    step_frames: int = 4
    native_lanes: int = DEFAULT_NATIVE_LANES
    native_execution: str = "parallel"
    learner_seed: int = DEFAULT_LEARNER_SEED
    learner_deterministic: bool = True
    max_collector_steps: int = DEFAULT_MAX_COLLECTOR_STEPS
    bc_epochs: int = 40
    bc_batch_size: int = 256
    bc_learning_rate: float = 1e-3
    bc_weight_decay: float = 1e-4
    bc_label_smoothing: float = 0.02
    bc_min_margin: float = 1.0
    bc_class_weight_power: float = 0.5
    bc_eval_every: int = 5

    def validate(self, manifest: SeedManifest) -> None:
        if self.round_index < 1:
            raise ValueError("DAgger round must be positive")
        if self.states_per_seed < 1 or self.lookahead_steps < 1:
            raise ValueError("DAgger states and lookahead must be positive")
        if not 3 <= self.step_frames <= 5:
            raise ValueError("step frames must be between 3 and 5")
        if not 1 <= self.native_lanes <= len(manifest.training_seeds):
            raise ValueError("native lanes must fit the training seed count")
        if self.native_execution not in {"serial", "parallel"}:
            raise ValueError("native execution must be serial or parallel")
        if self.max_collector_steps < 1:
            raise ValueError("maximum collector steps must be positive")
        if self.bc_epochs < 1 or self.bc_batch_size < 1:
            raise ValueError("BC epochs and batch size must be positive")
        if self.bc_learning_rate <= 0 or self.bc_weight_decay < 0:
            raise ValueError("BC learning rate must be positive and decay nonnegative")
        if not 0 <= self.bc_label_smoothing < 1:
            raise ValueError("BC label smoothing must be between 0 and 1")
        if self.bc_min_margin < 0 or self.bc_class_weight_power < 0:
            raise ValueError("BC margin and class-weight power must be nonnegative")
        if self.bc_eval_every < 1:
            raise ValueError("BC evaluation interval must be positive")

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        for name in (
            "manifest_path",
            "base_teacher_data_path",
            "learner_checkpoint",
            "output_directory",
        ):
            value[name] = str(getattr(self, name))
        return value


def run_dagger(config: DaggerConfig) -> dict[str, object]:
    manifest = load_manifest(config.manifest_path)
    config.validate(manifest)
    base = load_teacher_dataset(config.base_teacher_data_path, manifest)
    _assert_teacher_contract(base, config)
    learner, learner_metadata = _load_learner(config.learner_checkpoint, manifest)
    round_dataset = collect_learner_dataset(config, manifest, learner)

    round_directory = config.output_directory / "round-data"
    save_teacher_dataset(round_dataset, round_directory)
    round_path = round_directory / "teacher-data.npz"
    round_dataset = load_teacher_dataset(round_path, manifest)
    aggregate = _aggregate(base, round_dataset, config, learner_metadata)
    aggregate_directory = config.output_directory / "aggregate"
    save_teacher_dataset(aggregate, aggregate_directory)
    aggregate_path = aggregate_directory / "teacher-data.npz"
    aggregate = load_teacher_dataset(aggregate_path, manifest)

    bc_result = run_behavior_cloning(_bc_config(config, aggregate_path, manifest))
    record: dict[str, object] = {
        "schema_version": DAGGER_SCHEMA_VERSION,
        "kind": "dodge_ng_dagger_run",
        "round": config.round_index,
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.sha256,
        "base_teacher_data": str(config.base_teacher_data_path),
        "base_teacher_data_sha256": base.metadata["data_sha256"],
        "round_teacher_data": str(round_path),
        "round_teacher_data_sha256": round_dataset.metadata["data_sha256"],
        "aggregate_teacher_data": str(aggregate_path),
        "aggregate_teacher_data_sha256": aggregate.metadata["data_sha256"],
        "learner_checkpoint": str(config.learner_checkpoint),
        "learner_checkpoint_sha256": learner_metadata["checkpoint_sha256"],
        "learner_checkpoint_kind": learner_metadata["kind"],
        "learner_deterministic": config.learner_deterministic,
        "states_per_seed": config.states_per_seed,
        "round_examples": round_dataset.count,
        "round_decisive_examples": round_dataset.decisive_count,
        "aggregate_examples": aggregate.count,
        "aggregate_decisive_examples": aggregate.decisive_count,
        "score_cache": round_dataset.metadata["score_cache"],
        "legacy_inputs": "none",
        "config": config.to_json(),
        "bc_run_directory": str(config.output_directory / "bc"),
        "bc_selected_epoch": bc_result["selected_epoch"],
        "bc_best_inner_survival_frames": bc_result["best_inner_survival_frames"],
        "bc_final_training_evaluation": bc_result["final_training_evaluation"],
        "bc_final_holdout_evaluation": bc_result["final_evaluation"],
    }
    _write_json(config.output_directory / "run.json", record)
    report = _write_report(
        config.output_directory, record, base, round_dataset, aggregate
    )
    return {**record, "report": report}


def collect_learner_dataset(
    config: DaggerConfig,
    manifest: SeedManifest,
    learner: DodgeActorCriticCNN,
) -> TeacherDataset:
    counts = {seed: 0 for seed in manifest.training_seeds}
    current_seeds = list(manifest.training_seeds[: config.native_lanes])
    next_seed_index = config.native_lanes
    boards: list[np.ndarray] = []
    snapshots: list[bytes] = []
    seeds: list[int] = []
    frames: list[int] = []
    state_hashes: list[int] = []
    pixel_hashes: list[int] = []
    score_cache = CounterfactualCache()
    torch.manual_seed(config.learner_seed)
    learner.eval()

    with NativeBatchEnvironment(
        step_frames=config.step_frames,
        execution=config.native_execution,  # type: ignore[arg-type]
        full_state=True,
        pixels=False,
        board=True,
        difficulty=2,
        patterns_enabled=True,
        powerups_enabled=True,
    ) as environment:
        result = environment.reset_batch(np.asarray(current_seeds, dtype=np.uint32))
        for _collector_step in range(config.max_collector_steps):
            current_boards = result.board
            if current_boards is None:
                raise ControlRuntimeError("DAgger collection requires board buffers")
            for lane, seed in enumerate(current_seeds):
                snapshot = result.snapshot_bytes[lane]
                if (
                    result.modes[lane] == 2
                    and snapshot is not None
                    and counts[seed] < config.states_per_seed
                ):
                    boards.append(np.array(current_boards[lane], copy=True))
                    snapshots.append(snapshot)
                    seeds.append(seed)
                    frames.append(int(result.frames[lane]))
                    state_hashes.append(int(result.state_hashes[lane]))
                    pixel_hashes.append(int(result.pixel_hashes[lane]))
                    counts[seed] += 1

            if all(count == config.states_per_seed for count in counts.values()):
                break

            board_tensor = torch.from_numpy(
                np.asarray(current_boards, dtype=np.float32).copy()
            )
            with torch.inference_mode():
                logits, _ = learner(board_tensor)
                if config.learner_deterministic:
                    actions = logits.argmax(dim=1)
                else:
                    actions = Categorical(logits=logits).sample()
            done_result = environment.step_batch(
                actions.cpu().numpy().astype(np.uint8, copy=False)
            )
            reset_lanes: list[int] = []
            reset_seeds: list[int] = []
            for lane, seed in enumerate(current_seeds):
                needs_reset = bool(done_result.done[lane]) or (
                    counts[seed] >= config.states_per_seed
                )
                if not needs_reset:
                    continue
                reset_lanes.append(lane)
                if counts[seed] < config.states_per_seed:
                    reset_seeds.append(seed)
                elif next_seed_index < len(manifest.training_seeds):
                    replacement = manifest.training_seeds[next_seed_index]
                    next_seed_index += 1
                    current_seeds[lane] = replacement
                    reset_seeds.append(replacement)
                else:
                    reset_seeds.append(seed)
            if reset_lanes:
                reset_result = environment.reset_lanes(
                    np.asarray(reset_lanes, dtype=np.uint32),
                    np.asarray(reset_seeds, dtype=np.uint32),
                )
                result = _merge_results(done_result, reset_result)
            else:
                result = done_result
        else:
            incomplete = [
                seed for seed, count in counts.items() if count < config.states_per_seed
            ]
            raise ControlRuntimeError(
                "DAgger collection reached its step limit; incomplete seeds: "
                f"{incomplete[:8]}"
            )

        scores = score_cache.score(environment, snapshots, config.lookahead_steps)

    scores = np.asarray(scores, dtype=np.float32)
    actions = np.argmax(scores, axis=1).astype(np.int64)
    margins = _score_margins(scores)
    return TeacherDataset(
        boards=np.asarray(boards, dtype=np.float32),
        actions=actions,
        scores=scores,
        margins=margins,
        seeds=np.asarray(seeds, dtype=np.uint32),
        frames=np.asarray(frames, dtype=np.uint32),
        state_hashes=np.asarray(state_hashes, dtype=np.uint64),
        pixel_hashes=np.asarray(pixel_hashes, dtype=np.uint64),
        metadata={
            "schema_version": 1,
            "data_version": 1,
            "kind": "dodge_ng_teacher_dataset",
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": manifest.sha256,
            "seed_scope": "training_only",
            "training_seeds": list(manifest.training_seeds),
            "holdout_examples": 0,
            "legacy_inputs": "none",
            "board_shape": list(BOARD_SHAPE),
            "actions": list(ACTION_CHOICES),
            "examples": len(actions),
            "decisive_examples": int(np.count_nonzero(margins > 0)),
            "action_counts": np.bincount(actions, minlength=ACTION_COUNT).tolist(),
            "lookahead_steps": config.lookahead_steps,
            "step_frames": config.step_frames,
            "collection_policy": "dagger",
            "learner_deterministic": config.learner_deterministic,
            "learner_seed": config.learner_seed,
            "score_cache": score_cache.to_json(),
            "native_config": {
                "difficulty": 2,
                "patterns_enabled": True,
                "powerups_enabled": True,
            },
            "collector_config": config.to_json(),
        },
    )


def _aggregate(
    base: TeacherDataset,
    round_dataset: TeacherDataset,
    config: DaggerConfig,
    learner_metadata: Mapping[str, object],
) -> TeacherDataset:
    metadata = dict(base.metadata)
    metadata.update(
        {
            "collection_policy": "dagger_aggregate",
            "dagger_round": config.round_index,
            "dagger_round_examples": round_dataset.count,
            "dagger_round_decisive_examples": round_dataset.decisive_count,
            "dagger_base_data_sha256": base.metadata["data_sha256"],
            "dagger_round_data_sha256": round_dataset.metadata["data_sha256"],
            "dagger_learner_checkpoint": str(config.learner_checkpoint),
            "dagger_learner_checkpoint_sha256": learner_metadata["checkpoint_sha256"],
        }
    )
    return TeacherDataset(
        boards=np.concatenate((base.boards, round_dataset.boards)),
        actions=np.concatenate((base.actions, round_dataset.actions)),
        scores=np.concatenate((base.scores, round_dataset.scores)),
        margins=np.concatenate((base.margins, round_dataset.margins)),
        seeds=np.concatenate((base.seeds, round_dataset.seeds)),
        frames=np.concatenate((base.frames, round_dataset.frames)),
        state_hashes=np.concatenate((base.state_hashes, round_dataset.state_hashes)),
        pixel_hashes=np.concatenate((base.pixel_hashes, round_dataset.pixel_hashes)),
        metadata=metadata,
    )


def _bc_config(
    config: DaggerConfig, aggregate_path: Path, manifest: SeedManifest
) -> BCConfig:
    return BCConfig(
        manifest_path=config.manifest_path,
        teacher_data_path=aggregate_path,
        output_directory=config.output_directory / "bc",
        epochs=config.bc_epochs,
        batch_size=config.bc_batch_size,
        learning_rate=config.bc_learning_rate,
        weight_decay=config.bc_weight_decay,
        label_smoothing=config.bc_label_smoothing,
        min_margin=config.bc_min_margin,
        class_weight_power=config.bc_class_weight_power,
        eval_every=config.bc_eval_every,
        seed=config.learner_seed,
        device="auto",
        step_frames=config.step_frames,
        native_lanes=config.native_lanes,
        native_execution=config.native_execution,  # type: ignore[arg-type]
    )


def _load_learner(
    checkpoint: Path, manifest: SeedManifest
) -> tuple[DodgeActorCriticCNN, dict[str, object]]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError) as error:
        raise ControlRuntimeError(
            f"could not load DAgger learner checkpoint {checkpoint}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ControlInputError("DAgger learner checkpoint must be an object")
    if payload.get("model_type") != "DodgeActorCriticCNN":
        raise ControlInputError("DAgger learner checkpoint has incompatible model")
    if tuple(payload.get("board_shape", ())) != BOARD_SHAPE:
        raise ControlInputError(
            "DAgger learner checkpoint has incompatible board shape"
        )
    if tuple(payload.get("actions", ())) != ACTION_CHOICES:
        raise ControlInputError("DAgger learner checkpoint has incompatible actions")
    manifest_hash = payload.get("manifest_sha256")
    checkpoint_config = payload.get("config")
    if isinstance(checkpoint_config, Mapping):
        manifest_hash = checkpoint_config.get("training_seed_manifest", manifest_hash)
    if manifest_hash != manifest.sha256:
        raise ControlInputError(
            "DAgger learner checkpoint is bound to a different NG manifest"
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ControlInputError("DAgger learner checkpoint has no model state")
    model = DodgeActorCriticCNN()
    try:
        model.load_state_dict(state)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ControlInputError(
            f"DAgger learner checkpoint model is incompatible: {error}"
        ) from error
    model.eval()
    kind = "ppo" if isinstance(checkpoint_config, Mapping) else "bc"
    return model, {
        "kind": kind,
        "checkpoint_sha256": _sha256(checkpoint),
        "manifest_sha256": manifest.sha256,
    }


def _assert_teacher_contract(base: TeacherDataset, config: DaggerConfig) -> None:
    if base.metadata.get("lookahead_steps") != config.lookahead_steps:
        raise ControlInputError(
            "DAgger lookahead must match base teacher data for aggregation"
        )
    if base.metadata.get("step_frames") != config.step_frames:
        raise ControlInputError(
            "DAgger step frames must match base teacher data for aggregation"
        )


def _score_margins(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(scores, axis=1)
    return (ordered[:, -1] - ordered[:, -2]).astype(np.float32)


def _write_report(
    output_directory: Path,
    record: Mapping[str, object],
    base: TeacherDataset,
    round_dataset: TeacherDataset,
    aggregate: TeacherDataset,
) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": DAGGER_SCHEMA_VERSION,
        "kind": "dodge_ng_dagger_report",
        "run": dict(record),
        "datasets": {
            "base": {
                "examples": base.count,
                "decisive_examples": base.decisive_count,
                "data_sha256": base.metadata["data_sha256"],
            },
            "round": {
                "examples": round_dataset.count,
                "decisive_examples": round_dataset.decisive_count,
                "data_sha256": round_dataset.metadata["data_sha256"],
            },
            "aggregate": {
                "examples": aggregate.count,
                "decisive_examples": aggregate.decisive_count,
                "data_sha256": aggregate.metadata["data_sha256"],
            },
        },
        "bc": {
            "directory": record["bc_run_directory"],
            "selected_epoch": record["bc_selected_epoch"],
            "best_inner_survival_frames": record["bc_best_inner_survival_frames"],
            "final_training_evaluation": record["bc_final_training_evaluation"],
            "final_holdout_evaluation": record["bc_final_holdout_evaluation"],
        },
        "cache": record["score_cache"],
    }
    _write_json(output_directory / "report.json", report)
    (output_directory / "REPORT.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    return report


def _markdown_report(report: Mapping[str, object]) -> str:
    run = _object(report, "run")
    datasets = _object(report, "datasets")
    bc = _object(report, "bc")
    holdout = _object(bc, "final_holdout_evaluation")
    training = _object(bc, "final_training_evaluation")
    lines = [
        "# Dodge NG DAgger round report",
        "",
        f"Round: {run['round']}",
        f"Manifest SHA-256: `{run['manifest_sha256']}`",
        f"Learner checkpoint: `{run['learner_checkpoint']}`",
        "",
        "| Dataset | Examples | Decisive |",
        "|---|---:|---:|",
    ]
    for name in ("base", "round", "aggregate"):
        dataset = _object(datasets, name)
        lines.append(
            f"| {name} | {dataset['examples']} | {dataset['decisive_examples']} |"
        )
    lines.extend(
        [
            "",
            f"Selected BC epoch: {bc['selected_epoch']}",
            f"Inner selection survival: {bc['best_inner_survival_frames']:.1f}",
            f"Final training mean: {training['mean_survival_frames']:.1f}",
            f"Final holdout mean: {holdout['mean_survival_frames']:.1f}",
            "",
            "The holdout is reported after training-side checkpoint selection and "
            "does not select the DAgger checkpoint.",
            "",
            f"Score cache: `{run['score_cache']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _object(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise ControlRuntimeError(f"DAgger report field is not an object: {key}")
    return nested


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ControlInputError(f"could not read DAgger checkpoint: {error}") from error
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-dagger")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--base-teacher-data", type=Path, default=DEFAULT_BASE_TEACHER_DATA
    )
    parser.add_argument(
        "--learner-checkpoint", type=Path, default=DEFAULT_LEARNER_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--round", type=_positive_int, default=1)
    parser.add_argument("--states-per-seed", type=_positive_int, default=32)
    parser.add_argument("--lookahead-steps", type=_positive_int, default=64)
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--native-lanes", type=_positive_int, default=32)
    parser.add_argument(
        "--native-execution", choices=("serial", "parallel"), default="parallel"
    )
    parser.add_argument("--learner-seed", type=int, default=DEFAULT_LEARNER_SEED)
    parser.add_argument("--stochastic-learner", action="store_true")
    parser.add_argument(
        "--max-collector-steps", type=_positive_int, default=DEFAULT_MAX_COLLECTOR_STEPS
    )
    parser.add_argument("--bc-epochs", type=_positive_int, default=40)
    parser.add_argument("--bc-batch-size", type=_positive_int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-weight-decay", type=float, default=1e-4)
    parser.add_argument("--bc-label-smoothing", type=float, default=0.02)
    parser.add_argument("--bc-min-margin", type=float, default=1.0)
    parser.add_argument("--bc-class-weight-power", type=float, default=0.5)
    parser.add_argument("--bc-eval-every", type=_positive_int, default=5)
    arguments = parser.parse_args(argv)
    config = DaggerConfig(
        manifest_path=arguments.manifest,
        base_teacher_data_path=arguments.base_teacher_data,
        learner_checkpoint=arguments.learner_checkpoint,
        output_directory=arguments.output_dir,
        round_index=arguments.round,
        states_per_seed=arguments.states_per_seed,
        lookahead_steps=arguments.lookahead_steps,
        step_frames=arguments.step_frames,
        native_lanes=arguments.native_lanes,
        native_execution=arguments.native_execution,
        learner_seed=arguments.learner_seed,
        learner_deterministic=not arguments.stochastic_learner,
        max_collector_steps=arguments.max_collector_steps,
        bc_epochs=arguments.bc_epochs,
        bc_batch_size=arguments.bc_batch_size,
        bc_learning_rate=arguments.bc_learning_rate,
        bc_weight_decay=arguments.bc_weight_decay,
        bc_label_smoothing=arguments.bc_label_smoothing,
        bc_min_margin=arguments.bc_min_margin,
        bc_class_weight_power=arguments.bc_class_weight_power,
        bc_eval_every=arguments.bc_eval_every,
    )
    try:
        result = run_dagger(config)
    except (
        ControlInputError,
        ControlRuntimeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"dodge-ng-dagger: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output_directory": str(config.output_directory),
                "round_examples": result["round_examples"],
                "bc_final_holdout_mean": result["bc_final_holdout_evaluation"][
                    "mean_survival_frames"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
