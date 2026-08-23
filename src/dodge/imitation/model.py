from __future__ import annotations

from torch import Tensor, nn

from dodge.dataset import ACTION_CHOICES
from dodge.neat.state import OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION


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
