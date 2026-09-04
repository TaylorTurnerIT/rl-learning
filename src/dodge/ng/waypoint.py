"""Discrete waypoint geometry and native-action steering for Dodge NG."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
    ) -> WaypointDecision:
        """Steer toward an already selected target until it is reached."""
        current_cell = self.grid.nearest_cell(state.player.x, state.player.y)
        target = self.grid.point(target_cell)
        return self._decision(
            state,
            waypoint_action_index,
            current_cell,
            target_cell,
            target,
        )

    def _decision(
        self,
        state: RawState,
        waypoint_action_index: int,
        current_cell: tuple[int, int],
        target_cell: tuple[int, int],
        target: tuple[float, float],
    ) -> WaypointDecision:
        if not 0 <= waypoint_action_index < len(ACTION_CHOICES):
            raise ValueError("waypoint action index is outside the nine-action space")
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


@dataclass(frozen=True, slots=True)
class WaypointFeasibilityConfig:
    """Full-state oracle settings for testing waypoint resolutions."""

    step_frames: int = 4
    max_episode_steps: int = 200
    lookahead_steps: int = 8
    hold_decisions: int = 8
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
    grid = WaypointGrid(spacing)
    controller = WaypointController(grid)
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
                )
                actions[lane_index] = decision.native_action_index
                hold_remaining[lane_index] -= 1
                if decision.target_reached:
                    hold_remaining[lane_index] = 0
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
            if completed:
                reset = environment.reset_lanes(
                    np.asarray(completed, dtype=np.uint32),
                    np.zeros(len(completed), dtype=np.uint32),
                )
                for reset_index, lane in enumerate(completed):
                    current_snapshots[lane] = reset.snapshot_bytes[reset_index]
                    target_cells[lane] = None
                    hold_remaining[lane] = 0
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
    parser.add_argument("--hold-decisions", type=int, default=8)
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
