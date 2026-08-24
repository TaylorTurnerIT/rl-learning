from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from dodge.control import ControlInputError, ControlRuntimeError
from dodge.dataset import ACTION_CHOICES, DEFAULT_DATABASE
from dodge.imitation.data import (
    Demonstrations,
    load_demonstrations,
    split_demonstrations,
)
from dodge.imitation.model import BehaviorCloningMLP

DEFAULT_MODEL = Path("history/dodge/models/behavior-cloning.pt")
DEFAULT_HISTORY = DEFAULT_MODEL.with_suffix(".metrics.json")


@dataclass(frozen=True, slots=True)
class TrainingEpoch:
    epoch: int
    training_loss: float
    validation_loss: float | None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: BehaviorCloningMLP
    mean: Tensor
    standard_deviation: Tensor
    examples: int
    final_loss: float
    final_validation_loss: float | None
    device: str
    history: tuple[TrainingEpoch, ...]


def train_behavior_cloning(
    demonstrations: Demonstrations,
    *,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "auto",
    validation_demonstrations: Demonstrations | None = None,
) -> TrainingResult:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if demonstrations.count < 1:
        raise ValueError("at least one demonstration is required")
    if validation_demonstrations is not None and validation_demonstrations.count < 1:
        raise ValueError("at least one validation demonstration is required")
    execution_device = _resolve_device(device)
    torch.manual_seed(seed)
    observations = torch.from_numpy(demonstrations.observations.copy())
    actions = torch.from_numpy(demonstrations.actions.copy())
    mean = observations.mean(dim=0)
    standard_deviation = observations.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (observations - mean) / standard_deviation
    normalized_validation: Tensor | None = None
    validation_actions: Tensor | None = None
    if validation_demonstrations is not None:
        validation_observations = torch.from_numpy(
            validation_demonstrations.observations.copy()
        )
        normalized_validation = (validation_observations - mean) / standard_deviation
        validation_actions = torch.from_numpy(validation_demonstrations.actions.copy())
    loader = DataLoader(
        TensorDataset(normalized, actions),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = BehaviorCloningMLP().to(execution_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    final_loss = 0.0
    final_validation_loss: float | None = None
    history: list[TrainingEpoch] = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_examples = 0
        for batch_observations, batch_actions in loader:
            optimizer.zero_grad()
            loss = loss_function(
                model(batch_observations.to(execution_device)),
                batch_actions.to(execution_device),
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_actions)
            total_examples += len(batch_actions)
        final_loss = total_loss / total_examples
        if normalized_validation is not None and validation_actions is not None:
            model.eval()
            with torch.inference_mode():
                validation_loss = loss_function(
                    model(normalized_validation.to(execution_device)),
                    validation_actions.to(execution_device),
                )
            final_validation_loss = float(validation_loss.detach().cpu())
            model.train()
        history.append(TrainingEpoch(epoch, final_loss, final_validation_loss))
        log = f"epoch={epoch}/{epochs} train_loss={final_loss:.6f}"
        if final_validation_loss is not None:
            log += f" validation_loss={final_validation_loss:.6f}"
        print(log, flush=True)
    return TrainingResult(
        model.to("cpu"),
        mean,
        standard_deviation,
        demonstrations.count,
        final_loss,
        final_validation_loss,
        execution_device.type,
        tuple(history),
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ControlInputError(
                "CUDA is unavailable; use --device cpu or --device auto"
            )
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device: {device}")


def save_training_result(result: TrainingResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "actions": list(ACTION_CHOICES),
            "state_dict": result.model.state_dict(),
            "mean": result.mean,
            "standard_deviation": result.standard_deviation,
        },
        output,
    )


def save_training_history(
    result: TrainingResult, output: Path, validation_seeds: tuple[int, ...]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "version": 1,
                "examples": result.examples,
                "device": result.device,
                "validation_seeds": list(validation_seeds),
                "epochs": [
                    {
                        "epoch": entry.epoch,
                        "training_loss": entry.training_loss,
                        "validation_loss": entry.validation_loss,
                    }
                    for entry in result.history
                ],
            },
            indent=2,
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-bc-train")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    parser.add_argument("--validation-seed-count", type=int, default=10)
    arguments = parser.parse_args(argv)
    try:
        split = split_demonstrations(
            load_demonstrations(arguments.database), arguments.validation_seed_count
        )
        result = train_behavior_cloning(
            split.training,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            seed=arguments.seed,
            device=arguments.device,
            validation_demonstrations=split.validation,
        )
        save_training_result(result, arguments.output)
        save_training_history(result, arguments.history, split.validation_seeds)
    except (ControlInputError, ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-bc-train: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "examples": result.examples,
                "final_loss": result.final_loss,
                "final_validation_loss": result.final_validation_loss,
                "model": str(arguments.output),
                "history": str(arguments.history),
                "device": result.device,
            },
            separators=(",", ":"),
        )
    )
    return 0
