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
from dodge.imitation.data import Demonstrations, load_demonstrations
from dodge.imitation.model import BehaviorCloningMLP

DEFAULT_MODEL = Path("history/dodge/models/behavior-cloning.pt")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: BehaviorCloningMLP
    mean: Tensor
    standard_deviation: Tensor
    examples: int
    final_loss: float
    device: str


def train_behavior_cloning(
    demonstrations: Demonstrations,
    *,
    epochs: int = 50,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "auto",
) -> TrainingResult:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if demonstrations.count < 1:
        raise ValueError("at least one demonstration is required")
    execution_device = _resolve_device(device)
    torch.manual_seed(seed)
    observations = torch.from_numpy(demonstrations.observations.copy())
    actions = torch.from_numpy(demonstrations.actions.copy())
    mean = observations.mean(dim=0)
    standard_deviation = observations.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (observations - mean) / standard_deviation
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
        print(f"epoch={epoch}/{epochs} loss={final_loss:.6f}", flush=True)
    return TrainingResult(
        model.to("cpu"),
        mean,
        standard_deviation,
        demonstrations.count,
        final_loss,
        execution_device.type,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-bc-train")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="auto")
    arguments = parser.parse_args(argv)
    try:
        result = train_behavior_cloning(
            load_demonstrations(arguments.database),
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate,
            seed=arguments.seed,
            device=arguments.device,
        )
        save_training_result(result, arguments.output)
    except (ControlInputError, ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-bc-train: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "examples": result.examples,
                "final_loss": result.final_loss,
                "model": str(arguments.output),
                "device": result.device,
            },
            separators=(",", ":"),
        )
    )
    return 0
