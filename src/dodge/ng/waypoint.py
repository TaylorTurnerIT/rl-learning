"""Discrete waypoint geometry and native-action steering for Dodge NG."""

from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import numpy as np

from dodge.control import ControlRuntimeError
from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import (
    NativeBatchEnvironment,
    _decode_snapshot,
    _raw_state_from_snapshot,
)
from dodge.neat.bridge import Direction
from dodge.neat.state import RawState
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest
from dodge.ng.report import summarize_evaluation

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
_ACTION_INDEX_BY_ACTION: Final[dict[Direction, int]] = {
    action: index for index, action in enumerate(ACTION_CHOICES)
}
_DELTA_TO_ACTION_INDEX: Final[dict[tuple[int, int], int]] = {
    delta: _ACTION_INDEX_BY_ACTION[action] for delta, action in _DELTA_TO_ACTION.items()
}
_ACTION_INDEX_GRID: Final[np.ndarray] = np.asarray(
    (
        (5, 1, 7),
        (3, 0, 4),
        (6, 2, 8),
    ),
    dtype=np.uint8,
)

# These are the native Dodge player constants from dodge-core's update_player
# recurrence.  Keep the prediction in the same fixed-point domain as the game
# so a controller decision does not quietly introduce a second movement model.
_NATIVE_FIXED_SHIFT: Final[int] = 16
_NATIVE_FIXED_ONE: Final[int] = 1 << _NATIVE_FIXED_SHIFT
_NATIVE_PLAYER_SPEED_RAW: Final[int] = 32_768
_NATIVE_PLAYER_FRICTION_RAW: Final[int] = 52_428
_NATIVE_PLAYER_MIN_RAW: Final[int] = int(
    PLAYER_CENTER_MIN * _NATIVE_FIXED_ONE
)
_NATIVE_PLAYER_MAX_RAW: Final[int] = int(
    PLAYER_CENTER_MAX * _NATIVE_FIXED_ONE
)
_NATIVE_FIXED_MAX_ABS: Final[float] = (2**31 - 1) / _NATIVE_FIXED_ONE
_NATIVE_VELOCITY_COST_WEIGHT: Final[int] = 2
_NATIVE_ACTION_X: Final[np.ndarray] = np.asarray(
    [_ACTION_DELTAS[action][0] for action in ACTION_CHOICES], dtype=np.int64
)
_NATIVE_ACTION_Y: Final[np.ndarray] = np.asarray(
    [_ACTION_DELTAS[action][1] for action in ACTION_CHOICES], dtype=np.int64
)


@dataclass(frozen=True, slots=True)
class WaypointGrid:
    """Axis-aligned waypoint grid bounded by native player-center limits.

    ``WaypointGrid(spacing)`` preserves the original endpoint-based waypoint
    geometry. ``WaypointGrid.centered(resolution)`` is the fixed-N geometry
    used by the hazard observation: each point is the center of one equal
    cell, including the outer cells.
    """

    spacing: int | float
    min_center: float = PLAYER_CENTER_MIN
    max_center: float = PLAYER_CENTER_MAX
    ban_corner_nodes: bool = False
    resolution: int | None = None
    _axis_points: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.resolution is None:
            if (
                isinstance(self.spacing, bool)
                or not isinstance(self.spacing, int)
                or self.spacing < 1
            ):
                raise ValueError("waypoint spacing must be a positive integer")
        elif (
            isinstance(self.resolution, bool)
            or not isinstance(self.resolution, int)
            or self.resolution < 1
        ):
            raise ValueError("centered waypoint resolution must be a positive integer")
        if (
            self.min_center < 0
            or self.max_center > 128
            or self.max_center <= self.min_center
        ):
            raise ValueError("waypoint center bounds are invalid")
        if not isinstance(self.ban_corner_nodes, bool):
            raise ValueError("corner-node policy must be a boolean")

        if self.resolution is not None:
            cell_width = (self.max_center - self.min_center) / self.resolution
            points = [
                self.min_center + (column + 0.5) * cell_width
                for column in range(self.resolution)
            ]
            object.__setattr__(self, "spacing", cell_width)
        else:
            points = []
            point = self.min_center
            while point < self.max_center:
                points.append(point)
                point += self.spacing
            if not points or points[-1] != self.max_center:
                points.append(self.max_center)
        object.__setattr__(self, "_axis_points", tuple(points))

    @classmethod
    def centered(
        cls,
        resolution: int,
        *,
        min_center: float = PLAYER_CENTER_MIN,
        max_center: float = PLAYER_CENTER_MAX,
        ban_corner_nodes: bool = False,
    ) -> WaypointGrid:
        """Build ``resolution × resolution`` waypoints at cell centers."""
        return cls(
            1,
            min_center=min_center,
            max_center=max_center,
            ban_corner_nodes=ban_corner_nodes,
            resolution=resolution,
        )

    @property
    def axis_points(self) -> tuple[float, ...]:
        return self._axis_points

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

    def is_corner(self, cell: tuple[int, int]) -> bool:
        """Return whether a cell is one of the four grid corners."""
        last = len(self.axis_points) - 1
        return cell[0] in {0, last} and cell[1] in {0, last}

    def _apply_corner_policy(
        self,
        current_cell: tuple[int, int],
        target_cell: tuple[int, int],
    ) -> tuple[int, int]:
        if self.ban_corner_nodes and self.is_corner(target_cell):
            return current_cell
        return target_cell

    def target_for_action(
        self, x: float, y: float, waypoint_action_index: int
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[float, float]]:
        current_cell = self.nearest_cell(x, y)
        target_cell = self._apply_corner_policy(
            current_cell,
            self.neighbor_cell(current_cell, waypoint_action_index),
        )
        return current_cell, target_cell, self.point(target_cell)

    def target_cell_for_action(
        self, x: float, y: float, waypoint_action_index: int
    ) -> tuple[int, int]:
        """Return only the neighboring cell needed by the DQN hot path."""
        return self.target_cell_from_current(
            self.nearest_cell(x, y), waypoint_action_index
        )

    def target_cell_from_current(
        self, current_cell: tuple[int, int], waypoint_action_index: int
    ) -> tuple[int, int]:
        """Return an action target when the current cell is already known."""
        return self._apply_corner_policy(
            current_cell,
            self.neighbor_cell(current_cell, waypoint_action_index),
        )

    def _nearest_axis(self, value: float) -> int:
        points = self.axis_points
        right = bisect_left(points, value)
        if right == 0:
            return 0
        if right == len(points):
            return len(points) - 1
        left = right - 1
        if value - points[left] <= points[right] - value:
            return left
        return right


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
    arrival_latching: bool = False

    def __post_init__(self) -> None:
        if self.tolerance < 0:
            raise ValueError("waypoint steering tolerance must not be negative")
        if self.tolerance >= self.grid.spacing / 2:
            raise ValueError("waypoint steering tolerance must be below half spacing")
        if not isinstance(self.arrival_latching, bool):
            raise ValueError("arrival latching must be a boolean")

    def decide(self, state: RawState, waypoint_action_index: int) -> WaypointDecision:
        current_cell, target_cell, target = self.grid.target_for_action(
            state.player.x,
            state.player.y,
            waypoint_action_index,
        )
        return self._decision(
            state,
            waypoint_action_index,
            current_cell,
            target_cell,
            target,
        )

    def steer_to_cell(
        self,
        state: RawState,
        target_cell: tuple[int, int],
        waypoint_action_index: int = 0,
        *,
        arrived: bool = False,
    ) -> WaypointDecision:
        """Steer toward an already selected target until it is reached."""
        current_cell = self.grid.nearest_cell(state.player.x, state.player.y)
        target_cell = self.grid._apply_corner_policy(current_cell, target_cell)
        target = self.grid.point(target_cell)
        return self._decision(
            state,
            waypoint_action_index,
            current_cell,
            target_cell,
            target,
            arrived=arrived,
        )

    def steer_position(
        self,
        x: float,
        y: float,
        target_cell: tuple[int, int],
        waypoint_action_index: int = 0,
        *,
        arrived: bool = False,
    ) -> WaypointDecision:
        """Steer from numeric player coordinates without decoding hazards."""
        current_cell = self.grid.nearest_cell(x, y)
        target_cell = self.grid._apply_corner_policy(current_cell, target_cell)
        target = self.grid.point(target_cell)
        horizontal, vertical = self._steering_delta(x, y, target, arrived=arrived)
        native_action = _DELTA_TO_ACTION[(horizontal, vertical)]
        return WaypointDecision(
            waypoint_action_index=waypoint_action_index,
            current_cell=current_cell,
            target_cell=target_cell,
            target=target,
            target_reached=horizontal == 0 and vertical == 0,
            native_action=native_action,
        )

    def native_action_index_for_position(
        self,
        x: float,
        y: float,
        target_cell: tuple[int, int],
        *,
        arrived: bool = False,
    ) -> int:
        """Return a native action without allocating a decision object."""
        target = self.grid.point(target_cell)
        horizontal, vertical = self._steering_delta(x, y, target, arrived=arrived)
        return _DELTA_TO_ACTION_INDEX[(horizontal, vertical)]

    def native_action_indices_for_positions(
        self,
        positions: np.ndarray,
        target_positions: np.ndarray,
        *,
        arrived: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return scalar-equivalent native actions for one lane batch."""
        positions = np.asarray(positions, dtype=np.float64)
        target_positions = np.asarray(target_positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("waypoint positions must have shape (N, 2)")
        if target_positions.shape != positions.shape:
            raise ValueError("waypoint target positions must match positions")
        if not np.isfinite(positions).all() or not np.isfinite(target_positions).all():
            raise ValueError("waypoint positions must be finite")
        if arrived is not None:
            arrived = np.asarray(arrived, dtype=bool)
            if arrived.shape != (len(positions),):
                raise ValueError("waypoint arrival flags must match positions")

        delta_x = target_positions[:, 0] - positions[:, 0]
        delta_y = target_positions[:, 1] - positions[:, 1]
        horizontal = np.where(
            delta_x < -self.tolerance,
            -1,
            np.where(
                delta_x > self.tolerance,
                1,
                0,
            ),
        )
        vertical = np.where(
            delta_y < -self.tolerance,
            -1,
            np.where(
                delta_y > self.tolerance,
                1,
                0,
            ),
        )
        if self.arrival_latching and arrived is not None:
            horizontal = np.where(arrived, 0, horizontal)
            vertical = np.where(arrived, 0, vertical)
        return _ACTION_INDEX_GRID[horizontal + 1, vertical + 1]

    def native_action_index_for_position_velocity(
        self,
        position: tuple[float, float] | np.ndarray,
        velocity: tuple[float, float] | np.ndarray,
        target: tuple[float, float] | np.ndarray,
        step_frames: int = 3,
    ) -> int:
        """Return one velocity-aware native action for a single lane.

        ``position``, ``velocity``, and ``target`` are native coordinate pairs
        in pixels per native frame.  The target is expected to be a waypoint
        center, normally obtained from :meth:`WaypointGrid.centered`.  This
        method is opt-in; legacy sign steering remains the default everywhere.
        """
        actions = self.native_action_indices_for_positions_and_velocities(
            np.asarray(position, dtype=np.float64).reshape(1, 2),
            np.asarray(velocity, dtype=np.float64).reshape(1, 2),
            np.asarray(target, dtype=np.float64).reshape(1, 2),
            step_frames,
        )
        return int(actions[0])

    def native_action_indices_for_positions_and_velocities(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        target_positions: np.ndarray,
        step_frames: int = 3,
    ) -> np.ndarray:
        """Choose native actions by forecasting the next native frame block.

        The three arrays must have shape ``(N, 2)`` and contain current native
        player centers, current native velocities, and centered target
        coordinates.  Each of the nine existing native inputs is simulated for
        ``step_frames`` using the native Q16.16 friction, acceleration, and
        position clamp.  The action with the smallest deterministic score is
        selected: endpoint L1 target error plus twice endpoint L1 residual
        velocity.  Ties use the existing ``ACTION_CHOICES`` order.

        No arrival latch or mutable controller state is needed.  A stopped
        player inside a target window selects neutral; a player still carrying
        momentum selects the native input that reduces its predicted residual
        velocity.  The live game state is never modified or reset.
        """
        step_frames = _validate_native_step_frames(step_frames)
        positions = _validated_waypoint_vectors(positions, "waypoint positions")
        velocities = _validated_waypoint_vectors(velocities, "waypoint velocities")
        target_positions = _validated_waypoint_vectors(
            target_positions, "waypoint target positions"
        )
        if velocities.shape != positions.shape:
            raise ValueError("waypoint velocities must match positions")
        if target_positions.shape != positions.shape:
            raise ValueError("waypoint target positions must match positions")
        return _native_action_indices_for_positions_and_velocities(
            positions,
            velocities,
            target_positions,
            step_frames,
        )

    def target_reached(self, x: float, y: float, target_cell: tuple[int, int]) -> bool:
        """Return whether a player center is inside the configured target window."""
        target = self.grid.point(target_cell)
        return (
            abs(target[0] - x) <= self.tolerance
            and abs(target[1] - y) <= self.tolerance
        )

    def target_reached_between(
        self,
        previous_x: float,
        previous_y: float,
        x: float,
        y: float,
        target_cell: tuple[int, int],
    ) -> bool:
        """Return whether one native step entered or crossed the target window.

        The native player has momentum, so a multi-frame step can move from one
        side of a waypoint window to the other without leaving a sampled
        position inside the window. Arrival latching must treat that as an
        arrival; otherwise the controller reverses toward the target on the
        following sample.
        """
        target = self.grid.point(target_cell)
        return _axis_reached_between(
            previous_x, x, target[0], self.tolerance
        ) and _axis_reached_between(previous_y, y, target[1], self.tolerance)

    def _steering_delta(
        self,
        x: float,
        y: float,
        target: tuple[float, float],
        *,
        arrived: bool,
    ) -> tuple[int, int]:
        if self.arrival_latching and arrived:
            return 0, 0
        return (
            _sign(target[0] - x, self.tolerance),
            _sign(target[1] - y, self.tolerance),
        )

    def _decision(
        self,
        state: RawState,
        waypoint_action_index: int,
        current_cell: tuple[int, int],
        target_cell: tuple[int, int],
        target: tuple[float, float],
        *,
        arrived: bool = False,
    ) -> WaypointDecision:
        if not 0 <= waypoint_action_index < len(ACTION_CHOICES):
            raise ValueError("waypoint action index is outside the nine-action space")
        horizontal, vertical = self._steering_delta(
            state.player.x,
            state.player.y,
            target,
            arrived=arrived,
        )
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


def _validate_native_step_frames(step_frames: int) -> int:
    if (
        isinstance(step_frames, bool)
        or not isinstance(step_frames, (int, np.integer))
        or not 3 <= step_frames <= 5
    ):
        raise ValueError("velocity-aware waypoint step_frames must be between 3 and 5")
    return int(step_frames)


def _validated_waypoint_vectors(value: np.ndarray, name: str) -> np.ndarray:
    try:
        vectors = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric pairs") from error
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.isfinite(vectors).all():
        raise ValueError(f"{name} must be finite")
    if np.any(np.abs(vectors) > _NATIVE_FIXED_MAX_ABS):
        raise ValueError(f"{name} is outside the native fixed-point range")
    return vectors


def _native_raw_vectors(vectors: np.ndarray) -> np.ndarray:
    return np.trunc(vectors * _NATIVE_FIXED_ONE).astype(np.int64)


def _native_action_indices_for_positions_and_velocities(
    positions: np.ndarray,
    velocities: np.ndarray,
    target_positions: np.ndarray,
    step_frames: int,
) -> np.ndarray:
    """Forecast the native player recurrence for every lane and action."""
    position_raw = _native_raw_vectors(positions)
    velocity_raw = _native_raw_vectors(velocities)
    target_raw = _native_raw_vectors(target_positions)

    # The second axis is the native action index.  All arithmetic stays in
    # int64 so the fixed-point product is safe before its arithmetic shift.
    candidate_x = np.repeat(position_raw[:, 0, None], len(ACTION_CHOICES), axis=1)
    candidate_y = np.repeat(position_raw[:, 1, None], len(ACTION_CHOICES), axis=1)
    candidate_vx = np.repeat(velocity_raw[:, 0, None], len(ACTION_CHOICES), axis=1)
    candidate_vy = np.repeat(velocity_raw[:, 1, None], len(ACTION_CHOICES), axis=1)
    action_x = _NATIVE_ACTION_X[None, :] * _NATIVE_PLAYER_SPEED_RAW
    action_y = _NATIVE_ACTION_Y[None, :] * _NATIVE_PLAYER_SPEED_RAW

    for _ in range(step_frames):
        candidate_vx = (
            candidate_vx * _NATIVE_PLAYER_FRICTION_RAW
        ) >> _NATIVE_FIXED_SHIFT
        candidate_vy = (
            candidate_vy * _NATIVE_PLAYER_FRICTION_RAW
        ) >> _NATIVE_FIXED_SHIFT
        candidate_vx += action_x
        candidate_vy += action_y
        candidate_x = np.clip(
            candidate_x + candidate_vx,
            _NATIVE_PLAYER_MIN_RAW,
            _NATIVE_PLAYER_MAX_RAW,
        )
        candidate_y = np.clip(
            candidate_y + candidate_vy,
            _NATIVE_PLAYER_MIN_RAW,
            _NATIVE_PLAYER_MAX_RAW,
        )

    endpoint_error = np.abs(candidate_x - target_raw[:, 0, None]) + np.abs(
        candidate_y - target_raw[:, 1, None]
    )
    residual_velocity = np.abs(candidate_vx) + np.abs(candidate_vy)
    score = endpoint_error + _NATIVE_VELOCITY_COST_WEIGHT * residual_velocity
    return np.argmin(score, axis=1).astype(np.uint8)


def _axis_reached_between(
    previous: float,
    current: float,
    target: float,
    tolerance: float,
) -> bool:
    """Return whether a scalar step entered or crossed a target interval."""
    if abs(target - previous) <= tolerance or abs(target - current) <= tolerance:
        return True
    return min(previous, current) < target < max(previous, current)


@dataclass(frozen=True, slots=True)
class WaypointFeasibilityConfig:
    """Full-state oracle settings for testing waypoint resolutions."""

    step_frames: int = 4
    max_episode_steps: int = 200
    lookahead_steps: int = 8
    hold_decisions: int = 8
    steering_tolerance: float = 2.0
    arrival_latching: bool = False
    ban_corner_nodes: bool = False
    native_lanes: int = 32
    execution: Literal["serial", "parallel"] = "parallel"

    def validate(self) -> None:
        if not 3 <= self.step_frames <= 5:
            raise ValueError("step frames must be between 3 and 5")
        if self.max_episode_steps < 1:
            raise ValueError("maximum episode steps must be positive")
        if self.lookahead_steps < 1:
            raise ValueError("counterfactual lookahead must be positive")
        if self.hold_decisions < 1:
            raise ValueError("waypoint hold decisions must be positive")
        if not math.isfinite(self.steering_tolerance) or self.steering_tolerance < 0:
            raise ValueError(
                "waypoint steering tolerance must be finite and non-negative"
            )
        if not isinstance(self.arrival_latching, bool):
            raise ValueError("waypoint arrival latching must be a boolean")
        if not isinstance(self.ban_corner_nodes, bool):
            raise ValueError("waypoint corner-node policy must be a boolean")
        if self.native_lanes < 1:
            raise ValueError("native lane count must be positive")
        if self.execution not in {"serial", "parallel"}:
            raise ValueError("execution must be serial or parallel")


def evaluate_waypoint_oracle(
    seeds: tuple[int, ...] | list[int],
    spacing: int,
    *,
    config: WaypointFeasibilityConfig | None = None,
) -> dict[str, object]:
    """Evaluate a full-state counterfactual waypoint controller."""
    if config is None:
        config = WaypointFeasibilityConfig()
    config.validate()
    if not seeds:
        raise ValueError("waypoint feasibility requires at least one seed")
    grid = WaypointGrid(spacing, ban_corner_nodes=config.ban_corner_nodes)
    controller = WaypointController(
        grid,
        tolerance=config.steering_tolerance,
        arrival_latching=config.arrival_latching,
    )
    survival: dict[int, int] = {}
    terminated: dict[int, bool] = {}
    for start in range(0, len(seeds), config.native_lanes):
        local_seeds = tuple(seeds[start : start + config.native_lanes])
        local_survival, local_terminated = _evaluate_waypoint_batch(
            local_seeds,
            controller,
            config,
        )
        survival.update(local_survival)
        terminated.update(local_terminated)
    ordered_survival = [survival[int(seed)] for seed in seeds]
    ordered_terminated = [terminated[int(seed)] for seed in seeds]
    return {
        "seeds": list(seeds),
        "survival_frames": ordered_survival,
        "terminated": ordered_terminated,
        "spacing": spacing,
        "grid_shape": list(grid.shape),
        "point_count": grid.point_count,
        "step_frames": config.step_frames,
        "lookahead_steps": config.lookahead_steps,
        "hold_decisions": config.hold_decisions,
        "decision_interval": config.hold_decisions,
        "steering_tolerance": config.steering_tolerance,
        "arrival_latching": config.arrival_latching,
        "ban_corner_nodes": config.ban_corner_nodes,
        "max_episode_steps": config.max_episode_steps,
        "execution": config.execution,
        "summary": summarize_evaluation(
            {
                "seeds": list(seeds),
                "survival_frames": ordered_survival,
                "terminated": ordered_terminated,
            }
        ),
    }


def _evaluate_waypoint_batch(
    seeds: tuple[int, ...],
    controller: WaypointController,
    config: WaypointFeasibilityConfig,
) -> tuple[dict[int, int], dict[int, bool]]:
    environment = NativeBatchEnvironment(
        step_frames=config.step_frames,
        execution=config.execution,
        full_state=True,
        pixels=False,
        board=False,
    )
    active = np.ones(len(seeds), dtype=bool)
    survival = np.zeros(len(seeds), dtype=np.int64)
    terminated = np.zeros(len(seeds), dtype=bool)
    hold_remaining = np.zeros(len(seeds), dtype=np.int64)
    arrived = np.zeros(len(seeds), dtype=bool)
    target_cells: list[tuple[int, int] | None] = [None] * len(seeds)
    current_snapshots: list[bytes | None]
    try:
        result = environment.reset_batch(np.asarray(seeds, dtype=np.uint32))
        current_snapshots = list(result.snapshot_bytes)
        for _ in range(config.max_episode_steps):
            active_indices = np.flatnonzero(active)
            if not len(active_indices):
                break
            states = {
                int(lane): _raw_state_from_snapshot(
                    _decode_snapshot(_snapshot_at(current_snapshots, int(lane)))
                )
                for lane in active_indices
            }
            replan_indices = [
                int(lane)
                for lane in active_indices
                if hold_remaining[int(lane)] <= 0 or target_cells[int(lane)] is None
            ]
            scores = (
                environment.score_actions(
                    [_snapshot_at(current_snapshots, lane) for lane in replan_indices],
                    lookahead_steps=config.lookahead_steps,
                )
                if replan_indices
                else None
            )
            actions = np.zeros(len(seeds), dtype=np.uint8)
            if replan_indices:
                if scores is None:
                    raise ControlRuntimeError(
                        "waypoint feasibility failed to score replans"
                    )
                for lane, score_row in zip(replan_indices, scores, strict=True):
                    _, target = _select_waypoint_target(
                        states[lane],
                        score_row,
                        controller,
                    )
                    target_cells[lane] = target
                    hold_remaining[lane] = config.hold_decisions
                    arrived[lane] = False
            for lane in active_indices:
                lane_index = int(lane)
                target_cell = target_cells[lane_index]
                if target_cell is None:
                    raise ControlRuntimeError(
                        "waypoint feasibility selected no target cell"
                    )
                decision = controller.steer_to_cell(
                    states[lane_index],
                    target_cell,
                    arrived=bool(arrived[lane_index]),
                )
                actions[lane_index] = decision.native_action_index
                hold_remaining[lane_index] -= 1
            previous_positions = {
                int(lane): (
                    states[int(lane)].player.x,
                    states[int(lane)].player.y,
                )
                for lane in active_indices
            }
            result = environment.step_batch(actions)
            current_snapshots = list(result.snapshot_bytes)
            completed: list[int] = [
                lane for lane, done in enumerate(result.done) if bool(done)
            ]
            for lane in active_indices:
                lane_index = int(lane)
                survival[lane_index] += int(round(result.rewards[lane_index]))
                if bool(result.done[lane_index]):
                    active[lane_index] = False
                    terminated[lane_index] = True
                elif controller.arrival_latching and not arrived[lane_index]:
                    selected_target = target_cells[lane_index]
                    if selected_target is None:
                        raise ControlRuntimeError(
                            "waypoint feasibility lost its active target cell"
                        )
                    state = _raw_state_from_snapshot(
                        _decode_snapshot(_snapshot_at(current_snapshots, lane_index))
                    )
                    previous_x, previous_y = previous_positions[lane_index]
                    if controller.target_reached_between(
                        previous_x,
                        previous_y,
                        state.player.x,
                        state.player.y,
                        selected_target,
                    ):
                        arrived[lane_index] = True
            if completed:
                reset = environment.reset_lanes(
                    np.asarray(completed, dtype=np.uint32),
                    np.zeros(len(completed), dtype=np.uint32),
                )
                for reset_index, lane in enumerate(completed):
                    current_snapshots[lane] = reset.snapshot_bytes[reset_index]
                    target_cells[lane] = None
                    hold_remaining[lane] = 0
                    arrived[lane] = False
        unfinished = np.flatnonzero(active)
        survival[unfinished] = config.max_episode_steps * config.step_frames
    finally:
        environment.close()
    return (
        {int(seed): int(survival[lane]) for lane, seed in enumerate(seeds)},
        {int(seed): bool(terminated[lane]) for lane, seed in enumerate(seeds)},
    )


def _select_waypoint_target(
    state: RawState,
    scores: np.ndarray,
    controller: WaypointController,
) -> tuple[int, tuple[int, int]]:
    best_waypoint = 0
    best_score = -1.0
    for waypoint_action in range(len(ACTION_CHOICES)):
        decision = controller.decide(state, waypoint_action)
        score = float(scores[decision.native_action_index])
        if score > best_score:
            best_score = score
            best_waypoint = waypoint_action
    decision = controller.decide(state, best_waypoint)
    return best_waypoint, decision.target_cell


def build_waypoint_feasibility(
    output_directory: Path,
    manifest: SeedManifest,
    *,
    resolutions: tuple[int, ...] = WAYPOINT_RESOLUTIONS,
    config: WaypointFeasibilityConfig | None = None,
) -> dict[str, object]:
    """Select a waypoint spacing on training evidence, then report holdout."""
    manifest.validate()
    if config is None:
        config = WaypointFeasibilityConfig()
    config.validate()
    if not resolutions:
        raise ValueError("waypoint feasibility requires at least one resolution")
    training_results = [
        evaluate_waypoint_oracle(manifest.training_seeds, spacing, config=config)
        for spacing in resolutions
    ]
    selected = max(
        training_results,
        key=lambda result: (
            float(result["summary"]["horizon_completion_fraction"]),
            float(result["summary"]["p10_survival_frames"]),
            float(result["summary"]["mean_survival_frames"]),
            -int(result["spacing"]),
        ),
    )
    selected_spacing = int(selected["spacing"])
    holdout_result = evaluate_waypoint_oracle(
        manifest.holdout_seeds,
        selected_spacing,
        config=config,
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "dodge_ng_waypoint_feasibility",
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.sha256,
        "sample_count": manifest.sample_count,
        "training_count": len(manifest.training_seeds),
        "holdout_count": len(manifest.holdout_seeds),
        "training_seeds": list(manifest.training_seeds),
        "holdout_seeds": list(manifest.holdout_seeds),
        "config": {
            "step_frames": config.step_frames,
            "max_episode_steps": config.max_episode_steps,
            "lookahead_steps": config.lookahead_steps,
            "hold_decisions": config.hold_decisions,
            "decision_interval": config.hold_decisions,
            "steering_tolerance": config.steering_tolerance,
            "arrival_latching": config.arrival_latching,
            "ban_corner_nodes": config.ban_corner_nodes,
            "native_lanes": config.native_lanes,
            "execution": config.execution,
        },
        "training_resolutions": training_results,
        "selected_spacing": selected_spacing,
        "selection": "training_only; completion, p10, mean, then smaller spacing",
        "holdout_selected_resolution": holdout_result,
        "target": {
            "survival_frames": config.max_episode_steps * config.step_frames,
            "training_gate": float(selected["summary"]["mean_survival_frames"])
            >= config.max_episode_steps * config.step_frames,
        },
    }
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "waypoint-feasibility.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / "WAYPOINT_FEASIBILITY.md").write_text(
        _feasibility_markdown(report),
        encoding="utf-8",
    )
    return report


def _snapshot_at(snapshots: list[bytes | None], lane: int) -> bytes:
    snapshot = snapshots[lane]
    if snapshot is None:
        raise ControlRuntimeError("waypoint feasibility lost a live native snapshot")
    return snapshot


def _feasibility_markdown(report: dict[str, object]) -> str:
    training_results = report["training_resolutions"]
    holdout = report["holdout_selected_resolution"]
    if not isinstance(training_results, list) or not isinstance(holdout, dict):
        raise ValueError("waypoint feasibility report shape is invalid")
    lines = [
        "# Dodge NG waypoint feasibility",
        "",
        f"Manifest: `{report['manifest_id']}`  ",
        f"Manifest SHA-256: `{report['manifest_sha256']}`  ",
        f"Seeds: {report['training_count']} train / "
        f"{report['holdout_count']} locked holdout  ",
        f"Selected spacing: `{report['selected_spacing']}` pixels  ",
        f"Selection: `{report['selection']}`",
        "",
        "| Spacing | Grid | Points | Train mean | Train p10 | Train complete |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in training_results:
        if not isinstance(result, dict) or not isinstance(result["summary"], dict):
            raise ValueError("waypoint training result is invalid")
        summary = result["summary"]
        lines.append(
            f"| {result['spacing']} | "
            f"{result['grid_shape'][0]}x{result['grid_shape'][1]} | "
            f"{result['point_count']} | {float(summary['mean_survival_frames']):.1f} | "
            f"{float(summary['p10_survival_frames']):.1f} | "
            f"{float(summary['horizon_completion_fraction']):.1%} |"
        )
    holdout_summary = holdout["summary"]
    if not isinstance(holdout_summary, dict):
        raise ValueError("waypoint holdout result is invalid")
    lines.extend(
        [
            "",
            "Selected-resolution holdout: "
            f"mean {float(holdout_summary['mean_survival_frames']):.1f}, "
            f"p10 {float(holdout_summary['p10_survival_frames']):.1f}, "
            f"complete {float(holdout_summary['horizon_completion_fraction']):.1%}.",
            "",
            "This full-state oracle uses native counterfactual action scores "
            "for target selection; "
            "it is a feasibility control, not a learned DQN result.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-waypoint-feasibility")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("history/dodge/ng/waypoint-feasibility"),
    )
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=list(WAYPOINT_RESOLUTIONS)
    )
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--lookahead-steps", type=int, default=8)
    parser.add_argument(
        "--hold-decisions",
        "--decision-interval",
        dest="hold_decisions",
        type=int,
        default=8,
    )
    parser.add_argument("--steering-tolerance", type=float, default=2.0)
    parser.add_argument("--arrival-latching", action="store_true")
    parser.add_argument("--ban-corner-nodes", action="store_true")
    parser.add_argument("--native-lanes", type=int, default=32)
    parser.add_argument(
        "--execution", choices=("serial", "parallel"), default="parallel"
    )
    arguments = parser.parse_args(argv)
    config = WaypointFeasibilityConfig(
        step_frames=arguments.step_frames,
        max_episode_steps=arguments.max_episode_steps,
        lookahead_steps=arguments.lookahead_steps,
        hold_decisions=arguments.hold_decisions,
        steering_tolerance=arguments.steering_tolerance,
        arrival_latching=arguments.arrival_latching,
        ban_corner_nodes=arguments.ban_corner_nodes,
        native_lanes=arguments.native_lanes,
        execution=arguments.execution,
    )
    try:
        report = build_waypoint_feasibility(
            arguments.output_dir,
            load_manifest(arguments.manifest),
            resolutions=tuple(arguments.resolutions),
            config=config,
        )
    except (ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-ng-waypoint-feasibility: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output_directory": str(arguments.output_dir),
                "manifest_sha256": report["manifest_sha256"],
                "selected_spacing": report["selected_spacing"],
                "training_gate": report["target"]["training_gate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
