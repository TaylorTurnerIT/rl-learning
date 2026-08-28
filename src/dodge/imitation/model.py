from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from dodge.dataset import ACTION_CHOICES
from dodge.imitation.board import BOARD_CHANNELS, BOARD_SHAPE, encode_board
from dodge.neat.bridge import Direction
from dodge.neat.state import OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION, RawState


class BehaviorCloningMLP(nn.Module):
    """Map one Dodge observation to a score for each configured direction."""

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.network = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(ACTION_CHOICES)),
        )

    def forward(self, observations: Tensor) -> Tensor:
        if (
            observations.ndim != 2
            or observations.shape[1] != OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION
        ):
            raise ValueError(
                "observations must have shape "
                f"(N, {OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION})"
            )
        return self.network(observations)


class BehaviorCloningCNN(nn.Module):
    """Map a spatial Dodge board tensor to a score for each direction."""

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.network = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, len(ACTION_CHOICES)),
        )

    def forward(self, observations: Tensor) -> Tensor:
        if observations.ndim != 4 or tuple(observations.shape[1:]) != BOARD_SHAPE:
            raise ValueError(
                "observations must have shape "
                f"(N, {BOARD_SHAPE[0]}, {BOARD_SHAPE[1]}, {BOARD_SHAPE[2]})"
            )
        return self.network(observations)


def predict_action(
    model: BehaviorCloningCNN,
    state: RawState,
    mean: Tensor,
    standard_deviation: Tensor,
) -> Direction:
    """Choose one next action from a raw board using training normalization."""
    device = next(model.parameters()).device
    board = torch.from_numpy(encode_board(state)).unsqueeze(0).to(device)
    normalized = (board - mean.to(device)) / standard_deviation.to(device)
    with torch.inference_mode():
        action_index = int(model(normalized).argmax(dim=1).item())
    return cast(Direction, ACTION_CHOICES[action_index])
