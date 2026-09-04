"""Discrete waypoint geometry and native-action steering for Dodge NG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from dodge.dataset import ACTION_CHOICES
from dodge.neat.bridge import Direction
from dodge.neat.state import RawState

PLAYER_CENTER_MIN: Final[float] = 2.0
PLAYER_CENTER_MAX: Final[float] = 125.0
WAYPOINT_RESOLUTIONS: Final[tuple[int, ...]] = (8, 16, 32)

_ACTION_DELTAS: Final[dict[Direction, tuple[int, int]]] = {
    "neutral": (0, 0),
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
    "up_left": (-1, -1),
    "up_right": (1, -1),
    "down_left": (-1, 1),
    "down_right": (1, 1),
}
_DELTA_TO_ACTION: Final[dict[tuple[int, int], Direction]] = {
    delta: action for action, delta in _ACTION_DELTAS.items()
}


@dataclass(frozen=True, slots=True)
class WaypointGrid:
    """Axis-aligned waypoint grid bounded by native player-center limits."""

    spacing: int
    min_center: float = PLAYER_CENTER_MIN
    max_center: float = PLAYER_CENTER_MAX

    def __post_init__(self) -> None:
        if (
            isinstance(self.spacing, bool)
            or not isinstance(self.spacing, int)
            or self.spacing < 1
        ):
            raise ValueError("waypoint spacing must be a positive integer")
        if (
            self.min_center < 0
            or self.max_center > 128
            or self.max_center <= self.min_center
        ):
            raise ValueError("waypoint center bounds are invalid")

    @property
    def axis_points(self) -> tuple[float, ...]:
        points: list[float] = []
        point = self.min_center
        while point < self.max_center:
            points.append(point)
            point += self.spacing
        if not points or points[-1] != self.max_center:
            points.append(self.max_center)
        return tuple(points)

    @property
    def shape(self) -> tuple[int, int]:
        count = len(self.axis_points)
        return count, count

    @property
    def point_count(self) -> int:
        count = len(self.axis_points)
        return count * count

    def point(self, cell: tuple[int, int]) -> tuple[float, float]:
        column, row = cell
        points = self.axis_points
        if not 0 <= column < len(points) or not 0 <= row < len(points):
            raise ValueError("waypoint cell is outside grid")
        return points[column], points[row]

    def nearest_cell(self, x: float, y: float) -> tuple[int, int]:
        return self._nearest_axis(x), self._nearest_axis(y)

    def neighbor_cell(
        self, cell: tuple[int, int], waypoint_action_index: int
    ) -> tuple[int, int]:
        if not 0 <= waypoint_action_index < len(ACTION_CHOICES):
            raise ValueError("waypoint action index is outside the nine-action space")
        column, row = cell
        delta = _ACTION_DELTAS[ACTION_CHOICES[waypoint_action_index]]
        last = len(self.axis_points) - 1
        return (
            min(last, max(0, column + delta[0])),
            min(last, max(0, row + delta[1])),
        )

    def target_for_action(
        self, x: float, y: float, waypoint_action_index: int
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[float, float]]:
        current_cell = self.nearest_cell(x, y)
        target_cell = self.neighbor_cell(current_cell, waypoint_action_index)
        return current_cell, target_cell, self.point(target_cell)

    def _nearest_axis(self, value: float) -> int:
        points = self.axis_points
        return min(
            range(len(points)),
            key=lambda index: (abs(points[index] - value), index),
        )


@dataclass(frozen=True, slots=True)
class WaypointDecision:
    """One waypoint decision and its native steering action."""

    waypoint_action_index: int
    current_cell: tuple[int, int]
    target_cell: tuple[int, int]
    target: tuple[float, float]
    target_reached: bool
    native_action: Direction

    @property
    def native_action_index(self) -> int:
        return ACTION_CHOICES.index(self.native_action)


@dataclass(frozen=True, slots=True)
class WaypointController:
    """Translate relative waypoint choices into bounded native controls."""

    grid: WaypointGrid
    tolerance: float = 2.0

    def __post_init__(self) -> None:
        if self.tolerance < 0:
            raise ValueError("waypoint steering tolerance must not be negative")
        if self.tolerance >= self.grid.spacing / 2:
            raise ValueError("waypoint steering tolerance must be below half spacing")

    def decide(self, state: RawState, waypoint_action_index: int) -> WaypointDecision:
        current_cell, target_cell, target = self.grid.target_for_action(
            state.player.x,
            state.player.y,
            waypoint_action_index,
        )
        horizontal = _sign(target[0] - state.player.x, self.tolerance)
        vertical = _sign(target[1] - state.player.y, self.tolerance)
        native_action = _DELTA_TO_ACTION[(horizontal, vertical)]
        return WaypointDecision(
            waypoint_action_index=waypoint_action_index,
            current_cell=current_cell,
            target_cell=target_cell,
            target=target,
            target_reached=horizontal == 0 and vertical == 0,
            native_action=native_action,
        )


def _sign(delta: float, tolerance: float) -> int:
    if delta < -tolerance:
        return -1
    if delta > tolerance:
        return 1
    return 0
