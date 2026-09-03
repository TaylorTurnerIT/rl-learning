from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from dodge.control import PROJECT_ROOT, ControlInputError, ControlRuntimeError
from dodge.dataset import ACTION_CHOICES
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest
from dodge.ng.teacher import (
    BOARD_SHAPE,
    TeacherDataset,
    load_teacher_dataset,
)
from dodge.rl.ppo import DodgeActorCriticCNN, PPOConfig, evaluate_policy

BC_CHECKPOINT_VERSION = 1
BC_RUN_VERSION = 1
BC_MODEL_TYPE = "DodgeActorCriticCNN"
INNER_VALIDATION_COUNT = 10
DEFAULT_TEACHER_DATA = (
    PROJECT_ROOT
    / "history"
    / "dodge"
    / "ng"
    / "p2-teacher-planner-v3"
    / "teacher-data.npz"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "history" / "dodge" / "ng" / "bc-p2-v1"
DEFAULT_SEED = 2_026_0911


@dataclass(frozen=True, slots=True)
class BCConfig:
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    teacher_data_path: Path = DEFAULT_TEACHER_DATA
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.02
    min_margin: float = 1.0
    class_weight_power: float = 0.5
    eval_every: int = 5
    seed: int = DEFAULT_SEED
    device: str = "auto"
    step_frames: int = 4
    max_episode_steps: int = 2_000
    native_lanes: int = 32
    native_execution: str = "parallel"

    def validate(self, manifest: SeedManifest) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning rate must be positive and weight decay nonnegative"
            )
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label smoothing must be between 0 and 1")
        if self.min_margin < 0 or self.class_weight_power < 0:
            raise ValueError("margin and class weight power must be nonnegative")
        if self.eval_every < 1:
            raise ValueError("BC evaluation interval must be positive")
        if not 3 <= self.step_frames <= 5:
            raise ValueError("step frames must be between 3 and 5")
        if self.max_episode_steps < 1:
            raise ValueError("maximum episode steps must be positive")
        if self.native_lanes < 1 or self.native_lanes > len(manifest.training_seeds):
            raise ValueError("native lanes must fit the training seed count")
        if self.native_execution not in {"serial", "parallel"}:
            raise ValueError("native execution must be serial or parallel")

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        value["manifest_path"] = str(self.manifest_path)
        value["teacher_data_path"] = str(self.teacher_data_path)
        value["output_directory"] = str(self.output_directory)
        return value


@dataclass(frozen=True, slots=True)
class BCEpoch:
    epoch: int
    training_loss: float
    inner_loss: float
    training_accuracy: float
    inner_accuracy: float
    inner_survival_frames: float | None

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def run_behavior_cloning(config: BCConfig) -> dict[str, object]:
    manifest = load_manifest(config.manifest_path)
    config.validate(manifest)
    dataset = load_teacher_dataset(config.teacher_data_path, manifest)
    fit_seeds = manifest.training_seeds[INNER_VALIDATION_COUNT:]
    inner_seeds = manifest.training_seeds[:INNER_VALIDATION_COUNT]
    training_data = _decisive_subset(
        dataset.training_subset(fit_seeds), config.min_margin
    )
    inner_data = _decisive_subset(
        dataset.training_subset(inner_seeds), config.min_margin
    )
    if training_data.count < 1 or inner_data.count < 1:
        raise ControlInputError("teacher data has no decisive examples in a BC split")

    device = _resolve_device(config.device)
    torch.manual_seed(config.seed)
    model = DodgeActorCriticCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-5,
    )
    class_weights = _class_weights(training_data.actions, config.class_weight_power)
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=config.label_smoothing
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(training_data.boards.copy()),
            torch.from_numpy(training_data.actions.copy()),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    ppo_config = _evaluation_config(config, manifest)
    history: list[BCEpoch] = []
    best_state: dict[str, Tensor] | None = None
    best_inner = -math.inf
    best_epoch = 0
    latest_state: dict[str, Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for boards, actions in loader:
            logits, _ = model(boards.to(device))
            loss = loss_function(logits, actions.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(actions)
            total_examples += len(actions)

        training_loss = total_loss / total_examples
        _, training_accuracy = _classification_metrics(
            model, training_data, loss_function, device
        )
        inner_loss, inner_accuracy = _classification_metrics(
            model, inner_data, loss_function, device
        )
        inner_survival: float | None = None
        if epoch % config.eval_every == 0 or epoch == config.epochs:
            inner_result = evaluate_policy(model, ppo_config, inner_seeds)
            inner_survival = inner_result.mean_survival_frames
            if inner_survival > best_inner:
                best_inner = inner_survival
                best_epoch = epoch
                best_state = _cpu_state(model.state_dict())
        latest_state = _cpu_state(model.state_dict())
        entry = BCEpoch(
            epoch=epoch,
            training_loss=training_loss,
            inner_loss=inner_loss,
            training_accuracy=training_accuracy,
            inner_accuracy=inner_accuracy,
            inner_survival_frames=inner_survival,
        )
        history.append(entry)
        _append_jsonl(config.output_directory / "metrics.jsonl", entry.to_json())
        print(
            f"epoch={epoch}/{config.epochs} train_loss={training_loss:.5f} "
            f"inner_loss={inner_loss:.5f} train_acc={training_accuracy:.3f} "
            f"inner_acc={inner_accuracy:.3f} "
            f"inner_survival={inner_survival if inner_survival is not None else '-'}",
            flush=True,
        )

    if best_state is None or latest_state is None:
        raise ControlRuntimeError("BC did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_training = evaluate_policy(model, ppo_config, manifest.training_seeds)
    final_holdout = evaluate_policy(model, ppo_config, manifest.holdout_seeds)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    common_payload = {
        "version": BC_CHECKPOINT_VERSION,
        "model_type": BC_MODEL_TYPE,
        "board_shape": list(BOARD_SHAPE),
        "actions": list(ACTION_CHOICES),
        "manifest_sha256": manifest.sha256,
        "teacher_data_sha256": dataset.metadata["data_sha256"],
        "normalization": "none",
        "config": config.to_json(),
        "fit_seeds": list(fit_seeds),
        "inner_validation_seeds": list(inner_seeds),
        "selected_epoch": best_epoch,
        "best_inner_survival_frames": best_inner,
        "model_state_dict": best_state,
    }
    _atomic_torch_save(common_payload, config.output_directory / "checkpoint-best.pt")
    latest_payload = {**common_payload, "model_state_dict": latest_state}
    _atomic_torch_save(latest_payload, config.output_directory / "checkpoint-latest.pt")
    started_at = datetime.now(UTC).isoformat()
    record: dict[str, object] = {
        "version": BC_RUN_VERSION,
        "kind": "dodge_ng_bc_run",
        "started_at_utc": started_at,
        "model_type": BC_MODEL_TYPE,
        "board_shape": list(BOARD_SHAPE),
        "actions": list(ACTION_CHOICES),
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.sha256,
        "teacher_data": str(config.teacher_data_path),
        "teacher_data_sha256": dataset.metadata["data_sha256"],
        "legacy_inputs": "none",
        "fit_seeds": list(fit_seeds),
        "inner_validation_seeds": list(inner_seeds),
        "holdout_seeds": list(manifest.holdout_seeds),
        "training_examples": training_data.count,
        "inner_examples": inner_data.count,
        "dataset_examples": dataset.count,
        "dataset_decisive_examples": dataset.decisive_count,
        "config": config.to_json(),
        "selected_checkpoint": "checkpoint-best.pt",
        "latest_checkpoint": "checkpoint-latest.pt",
        "selected_epoch": best_epoch,
        "best_inner_survival_frames": best_inner,
        "final_training_evaluation": final_training.to_json(),
        "final_evaluation": final_holdout.to_json(),
    }
    _write_json(config.output_directory / "run.json", record)
    report = _write_report(config.output_directory, record, history, dataset)
    return {**record, "report": report}


def load_bc_actor_state(
    checkpoint: Path,
    manifest: SeedManifest,
    *,
    teacher_data: TeacherDataset | None = None,
) -> dict[str, Tensor]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as error:
        raise ControlRuntimeError(f"could not load BC checkpoint: {error}") from error
    if not isinstance(payload, Mapping):
        raise ControlRuntimeError("BC checkpoint must be an object")
    if (
        payload.get("version") != BC_CHECKPOINT_VERSION
        or payload.get("model_type") != BC_MODEL_TYPE
        or tuple(payload.get("board_shape", ())) != BOARD_SHAPE
        or tuple(payload.get("actions", ())) != ACTION_CHOICES
        or payload.get("manifest_sha256") != manifest.sha256
    ):
        raise ControlRuntimeError("BC checkpoint has incompatible metadata")
    if teacher_data is not None and payload.get(
        "teacher_data_sha256"
    ) != teacher_data.metadata.get("data_sha256"):
        raise ControlRuntimeError("BC checkpoint teacher dataset does not match")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ControlRuntimeError("BC checkpoint is missing model weights")
    return {name: value for name, value in state.items() if isinstance(value, Tensor)}


def _decisive_subset(dataset: TeacherDataset, minimum_margin: float) -> TeacherDataset:
    return (
        dataset
        if minimum_margin <= 0
        else _subset(dataset, dataset.margins >= minimum_margin)
    )


def _subset(dataset: TeacherDataset, mask: np.ndarray) -> TeacherDataset:
    return TeacherDataset(
        boards=dataset.boards[mask],
        actions=dataset.actions[mask],
        scores=dataset.scores[mask],
        margins=dataset.margins[mask],
        seeds=dataset.seeds[mask],
        frames=dataset.frames[mask],
        state_hashes=dataset.state_hashes[mask],
        pixel_hashes=dataset.pixel_hashes[mask],
        metadata=dict(dataset.metadata),
    )


def _class_weights(actions: np.ndarray, power: float) -> Tensor:
    counts = np.bincount(actions, minlength=len(ACTION_CHOICES)).astype(np.float32)
    weights = np.ones_like(counts)
    present = counts > 0
    if power > 0 and np.any(present):
        weights[present] = (counts[present] / counts[present].mean()) ** (-power)
        weights[present] /= weights[present].mean()
    return torch.from_numpy(weights)


def _classification_metrics(
    model: DodgeActorCriticCNN,
    dataset: TeacherDataset,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    boards = torch.from_numpy(dataset.boards.copy()).to(device)
    actions = torch.from_numpy(dataset.actions.copy()).to(device)
    with torch.inference_mode():
        logits, _ = model(boards)
        loss = loss_function(logits, actions)
        accuracy = (logits.argmax(dim=1) == actions).float().mean()
    model.train()
    return float(loss.cpu()), float(accuracy.cpu())


def _evaluation_config(config: BCConfig, manifest: SeedManifest) -> PPOConfig:
    return PPOConfig(
        updates=1,
        rollout_steps=config.native_lanes,
        update_epochs=1,
        minibatch_size=1,
        checkpoint_every=1,
        eval_every=0,
        learning_rate=config.learning_rate,
        step_frames=config.step_frames,
        max_episode_steps=config.max_episode_steps,
        seed=config.seed,
        device=config.device,
        backend="native",
        native_lanes=config.native_lanes,
        native_execution=config.native_execution,  # type: ignore[arg-type]
        observation_mode="board",
        training_seeds=manifest.training_seeds,
        training_seed_manifest=manifest.sha256,
    )


def _write_report(
    output_directory: Path,
    record: Mapping[str, object],
    history: list[BCEpoch],
    dataset: TeacherDataset,
) -> dict[str, object]:
    training = record["final_training_evaluation"]
    holdout = record["final_evaluation"]
    if not isinstance(training, Mapping) or not isinstance(holdout, Mapping):
        raise ControlRuntimeError("BC report is missing final evaluations")
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "dodge_ng_bc_report",
        "run": dict(record),
        "dataset": {
            "examples": dataset.count,
            "decisive_examples": dataset.decisive_count,
            "manifest_sha256": dataset.metadata["manifest_sha256"],
            "teacher_data_sha256": dataset.metadata["data_sha256"],
        },
        "history": [entry.to_json() for entry in history],
        "comparison": {
            "mean_train_minus_holdout": float(training["mean_survival_frames"])
            - float(holdout["mean_survival_frames"]),
        },
        "plots": ["bc_curves.png"],
    }
    _write_json(output_directory / "report.json", report)
    _plot_history(output_directory, history)
    (output_directory / "REPORT.md").write_text(_markdown_report(report))
    return report


def _plot_history(output_directory: Path, history: list[BCEpoch]) -> None:
    epochs = [entry.epoch for entry in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [entry.training_loss for entry in history], label="fit loss")
    axes[0].plot(epochs, [entry.inner_loss for entry in history], label="inner loss")
    axes[0].plot(
        epochs,
        [entry.training_accuracy for entry in history],
        label="fit accuracy",
        linestyle="--",
    )
    axes[0].plot(
        epochs,
        [entry.inner_accuracy for entry in history],
        label="inner accuracy",
        linestyle="--",
    )
    axes[0].set_title("Planner-label classification")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    evaluated = [entry for entry in history if entry.inner_survival_frames is not None]
    axes[1].plot(
        [entry.epoch for entry in evaluated],
        [entry.inner_survival_frames for entry in evaluated],
        marker="o",
        color="C1",
    )
    axes[1].set_title("Closed-loop inner survival")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("survival frames")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_directory / "bc_curves.png", dpi=140)
    plt.close(figure)


def _markdown_report(report: Mapping[str, object]) -> str:
    run = report["run"]
    comparison = report["comparison"]
    if not isinstance(run, Mapping) or not isinstance(comparison, Mapping):
        return "# Dodge NG BC report\n"
    training = run["final_training_evaluation"]
    holdout = run["final_evaluation"]
    return "\n".join(
        [
            "# Dodge NG behavior cloning report",
            "",
            f"- Manifest: `{run['manifest_sha256']}`",
            f"- Teacher examples: {run['dataset_examples']} "
            f"({run['dataset_decisive_examples']} decisive)",
            f"- Fit/inner examples: {run['training_examples']}/{run['inner_examples']}",
            f"- Selected epoch: {run['selected_epoch']}",
            f"- Inner selection survival: {run['best_inner_survival_frames']:.1f}",
            f"- Final training mean: {training['mean_survival_frames']:.1f}",
            f"- Locked holdout mean: {holdout['mean_survival_frames']:.1f}",
            f"- Train-minus-holdout: {comparison['mean_train_minus_holdout']:.1f}",
            "",
            "The holdout is reported after training-side checkpoint selection and "
            "does not select the BC checkpoint.",
            "",
        ]
    )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise ControlInputError(
                "CUDA is unavailable; use --device cpu or --device auto"
            )
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device: {value}")


def _cpu_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")
        stream.flush()


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
    parser = argparse.ArgumentParser(prog="dodge-ng-bc")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--teacher-data", type=Path, default=DEFAULT_TEACHER_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--epochs", type=_positive_int, default=40)
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--min-margin", type=float, default=1.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--eval-every", type=_positive_int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=_positive_int, default=2_000)
    parser.add_argument("--native-lanes", type=_positive_int, default=32)
    parser.add_argument(
        "--native-execution", choices=("serial", "parallel"), default="parallel"
    )
    arguments = parser.parse_args(argv)
    config = BCConfig(
        manifest_path=arguments.manifest,
        teacher_data_path=arguments.teacher_data,
        output_directory=arguments.output_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        label_smoothing=arguments.label_smoothing,
        min_margin=arguments.min_margin,
        class_weight_power=arguments.class_weight_power,
        eval_every=arguments.eval_every,
        seed=arguments.seed,
        device=arguments.device,
        step_frames=arguments.step_frames,
        max_episode_steps=arguments.max_episode_steps,
        native_lanes=arguments.native_lanes,
        native_execution=arguments.native_execution,
    )
    try:
        result = run_behavior_cloning(config)
    except (
        ControlInputError,
        ControlRuntimeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"dodge-ng-bc: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output_directory": str(config.output_directory),
                "selected_epoch": result["selected_epoch"],
                "training_examples": result["training_examples"],
                "final_training_mean": result["final_training_evaluation"][
                    "mean_survival_frames"
                ],
                "final_holdout_mean": result["final_evaluation"][
                    "mean_survival_frames"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
