from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from dodge.control import ControlInputError, ControlRuntimeError, parse_seed
from dodge.dataset import ACTION_CHOICES
from dodge.native.differential import NativeSnapshot, decode_native_snapshot
from dodge.neat.environment import EpisodeResult, Observation, Transition
from dodge.neat.state import (
    EntityState,
    ParticleState,
    PlayerState,
    RawState,
    project_state,
)

Execution = Literal["serial", "parallel"]
ML_OBSERVATION_SIZE = 225
HAZARD_CHANNELS = 14
HAZARD_SCALARS = 4
HAZARD_OBSERVATION_VERSION = 1
HAZARD_MAX_GRID_SIZE = 128


@dataclass(frozen=True, slots=True)
class NativeBatchResult:
    """Owned batch arrays returned by the Rust boundary."""

    lane_ids: np.ndarray
    frames: np.ndarray
    frames_advanced: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    seeds: np.ndarray
    state_hashes: np.ndarray
    pixel_hashes: np.ndarray
    modes: np.ndarray
    event_flags: np.ndarray
    pixels: np.ndarray | None
    board: np.ndarray | None
    ml_observation: np.ndarray | None
    player_positions: np.ndarray | None
    snapshot_bytes: tuple[bytes | None, ...]

    @property
    def lane_count(self) -> int:
        return int(self.frames.shape[0])


@dataclass(frozen=True, slots=True)
class NativeMlBatchResult:
    """Render-free ML result returned by the Rust batch boundary."""

    lane_ids: np.ndarray
    frames: np.ndarray
    frames_advanced: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    seeds: np.ndarray
    modes: np.ndarray
    ml_observation: np.ndarray
    player_positions: np.ndarray

    @property
    def lane_count(self) -> int:
        return int(self.frames.shape[0])


@dataclass(frozen=True, slots=True)
class NativePixelBatchResult:
    """Minimal rendered result returned by the native pixel boundary."""

    lane_ids: np.ndarray
    frames: np.ndarray
    frames_advanced: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    seeds: np.ndarray
    modes: np.ndarray
    pixels: np.ndarray
    player_positions: np.ndarray

    @property
    def lane_count(self) -> int:
        return int(self.frames.shape[0])


@dataclass(frozen=True, slots=True)
class NativeHazardBatchResult:
    """Slow/reference frozen-center hazard field from the Rust boundary."""

    lane_ids: np.ndarray
    frames: np.ndarray
    frames_advanced: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    seeds: np.ndarray
    modes: np.ndarray
    survival_frames: np.ndarray
    grid_size: int
    prediction_horizon_frames: int
    spawn_halo_radius: int
    hazard_channels: int
    hazard_scalars: int
    hazard_observation: np.ndarray
    ttc_reference: np.ndarray
    player_positions: np.ndarray

    @property
    def lane_count(self) -> int:
        return int(self.frames.shape[0])


class NativeBatchEnvironment:
    """Python owner for independent native Rust lanes.

    Arrays are allocated on the Python/NumPy side for each returned batch and
    do not borrow mutable state from the Rust environment. The canonical full
    state is available as binary snapshot bytes when ``full_state=True``.
    """

    def __init__(
        self,
        *,
        step_frames: int = 4,
        execution: Execution = "serial",
        full_state: bool = False,
        pixels: bool = False,
        board: bool = True,
        difficulty: int = 2,
        patterns_enabled: bool = True,
        powerups_enabled: bool = True,
        include_offscreen_board: bool = False,
        preserve_offscreen_coordinates: bool = False,
        ml: bool = False,
        ml_grid_spacing: int = 32,
    ) -> None:
        if not 3 <= step_frames <= 5:
            raise ValueError("step_frames must be between 3 and 5")
        if execution not in {"serial", "parallel"}:
            raise ValueError("execution must be serial or parallel")
        if not 1 <= difficulty <= 3:
            raise ValueError("difficulty must be between 1 and 3")
        if (
            isinstance(ml_grid_spacing, bool)
            or not isinstance(ml_grid_spacing, int)
            or ml_grid_spacing < 1
        ):
            raise ValueError("ml_grid_spacing must be a positive integer")
        try:
            import dodge_native
        except ModuleNotFoundError as error:
            raise ControlRuntimeError(
                "dodge_native is not installed; run `uv sync --extra native`"
            ) from error

        self._native = dodge_native.NativeBatchEnv(
            step_frames,
            execution,
            full_state,
            pixels,
            board,
            difficulty,
            patterns_enabled,
            powerups_enabled,
            include_offscreen_board,
            preserve_offscreen_coordinates,
            ml,
            ml_grid_spacing,
        )
        self.step_frames = step_frames
        self.execution = execution
        self.full_state_enabled = full_state
        self.pixels_enabled = pixels
        self.board_enabled = board
        self.include_offscreen_board = include_offscreen_board
        self.preserve_offscreen_coordinates = preserve_offscreen_coordinates
        self.ml_enabled = ml
        self.ml_grid_spacing = ml_grid_spacing
        self._last_result: NativeBatchResult | None = None
        self._last_ml_result: NativeMlBatchResult | None = None
        self._last_pixel_result: NativePixelBatchResult | None = None
        self._last_hazard_result: NativeHazardBatchResult | None = None
        self._closed = False

    @property
    def lane_count(self) -> int:
        self._ensure_open()
        return int(self._native.lane_count)

    @property
    def closed(self) -> bool:
        return self._closed

    def reset_batch(self, seeds: object) -> NativeBatchResult:
        self._ensure_open()
        values = _integer_array(seeds, "seeds", maximum=32_767)
        payload = self._native.reset_batch(values)
        result = _result_from_payload(payload)
        self._last_result = result
        self._last_ml_result = None
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def reset_batch_with_startup(self, seeds: object) -> NativeBatchResult:
        """Reset after native movement to the first on-screen enemy."""
        self._ensure_open()
        values = _integer_array(seeds, "seeds", maximum=32_767)
        payload = self._native.reset_batch_with_startup(values)
        result = _result_from_payload(payload)
        self._last_result = result
        self._last_ml_result = None
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def reset_lanes(self, lanes: object, seeds: object) -> NativeBatchResult:
        self._ensure_open()
        lane_values = _integer_array(lanes, "lanes", maximum=2**31 - 1)
        seed_values = _integer_array(seeds, "seeds", maximum=32_767)
        if lane_values.shape != seed_values.shape:
            raise ValueError("lanes and seeds must have the same length")
        payload = self._native.reset_lanes(lane_values, seed_values)
        result = _result_from_payload(payload)
        self._last_result = result
        self._last_ml_result = None
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def reset_lanes_with_startup(
        self,
        lanes: object,
        seeds: object,
    ) -> NativeBatchResult:
        """Reset selected lanes after the native AI startup sequence."""
        self._ensure_open()
        lane_values = _integer_array(lanes, "lanes", maximum=2**31 - 1)
        seed_values = _integer_array(seeds, "seeds", maximum=32_767)
        if lane_values.shape != seed_values.shape:
            raise ValueError("lanes and seeds must have the same length")
        payload = self._native.reset_lanes_with_startup(lane_values, seed_values)
        result = _result_from_payload(payload)
        self._last_result = result
        self._last_ml_result = None
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def step_batch(self, actions: object) -> NativeBatchResult:
        self._ensure_open()
        values = _integer_array(actions, "actions", maximum=8)
        payload = self._native.step_batch(values)
        result = _result_from_payload(payload)
        self._last_result = result
        self._last_ml_result = None
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def step_pixels(self, actions: object) -> NativePixelBatchResult:
        """Advance lanes through the minimal rendered pixel-only boundary."""
        self._ensure_open()
        values = _integer_array(actions, "actions", maximum=8)
        payload = self._native.step_pixels(values)
        result = _pixel_result_from_payload(payload)
        self._last_result = None
        self._last_ml_result = None
        self._last_pixel_result = result
        self._last_hazard_result = None
        return result

    def step_pixels_active(
        self, actions: object, active: object
    ) -> NativePixelBatchResult:
        """Step selected lanes, returning only their original lane IDs and pixels."""
        self._ensure_open()
        values = _integer_array(actions, "actions", maximum=8)
        selected = np.asarray(active)
        if selected.dtype != np.bool_ or selected.shape != values.shape:
            raise ValueError("active must be a boolean array matching actions")
        payload = self._native.step_pixels_active(
            values, np.ascontiguousarray(selected)
        )
        result = _pixel_result_from_payload(payload)
        self._last_result = None
        self._last_ml_result = None
        self._last_pixel_result = result
        self._last_hazard_result = None
        return result

    def reset_ml_batch(self, seeds: object) -> NativeMlBatchResult:
        """Reset lanes without rendering or materializing canonical snapshots."""
        self._ensure_open()
        values = _integer_array(seeds, "seeds", maximum=32_767)
        payload = self._native.reset_ml_batch(values)
        result = _ml_result_from_payload(payload)
        self._last_result = None
        self._last_ml_result = result
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def reset_ml_batch_with_startup(self, seeds: object) -> NativeMlBatchResult:
        """Reset ML lanes at an upward waypoint with an on-screen enemy."""
        self._ensure_open()
        values = _integer_array(seeds, "seeds", maximum=32_767)
        payload = self._native.reset_ml_batch_with_startup(values)
        result = _ml_result_from_payload(payload)
        self._last_result = None
        self._last_ml_result = result
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def reset_ml_batch_with_centered_startup(
        self,
        seeds: object,
        grid_size: int,
    ) -> NativeMlBatchResult:
        """Reset ML lanes using the fixed-N centered hazard waypoint grid."""
        self._ensure_open()
        values = _integer_array(seeds, "seeds", maximum=32_767)
        if (
            isinstance(grid_size, bool)
            or not isinstance(grid_size, int)
            or not 1 <= grid_size <= HAZARD_MAX_GRID_SIZE
        ):
            raise ValueError(
                f"grid_size must be an integer between 1 and {HAZARD_MAX_GRID_SIZE}"
            )
        payload = self._native.reset_ml_batch_with_centered_startup(values, grid_size)
        result = _ml_result_from_payload(payload)
        self._last_result = None
        self._last_ml_result = result
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def reset_ml_lanes(self, lanes: object, seeds: object) -> NativeMlBatchResult:
        """Reset selected lanes through the render-free ML boundary."""
        self._ensure_open()
        lane_values = _integer_array(lanes, "lanes", maximum=2**31 - 1)
        seed_values = _integer_array(seeds, "seeds", maximum=32_767)
        if lane_values.shape != seed_values.shape:
            raise ValueError("lanes and seeds must have the same length")
        payload = self._native.reset_ml_lanes(lane_values, seed_values)
        result = _ml_result_from_payload(payload)
        self._last_result = None
        self._last_ml_result = result
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def reset_ml_lanes_with_startup(
        self,
        lanes: object,
        seeds: object,
    ) -> NativeMlBatchResult:
        """Reset selected ML lanes through the native AI startup sequence."""
        self._ensure_open()
        lane_values = _integer_array(lanes, "lanes", maximum=2**31 - 1)
        seed_values = _integer_array(seeds, "seeds", maximum=32_767)
        if lane_values.shape != seed_values.shape:
            raise ValueError("lanes and seeds must have the same length")
        payload = self._native.reset_ml_lanes_with_startup(lane_values, seed_values)
        result = _ml_result_from_payload(payload)
        self._last_result = None
        self._last_ml_result = result
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def reset_ml_lanes_with_centered_startup(
        self,
        lanes: object,
        seeds: object,
        grid_size: int,
    ) -> NativeMlBatchResult:
        """Reset selected ML lanes using centered hazard waypoints."""
        self._ensure_open()
        lane_values = _integer_array(lanes, "lanes", maximum=2**31 - 1)
        seed_values = _integer_array(seeds, "seeds", maximum=32_767)
        if lane_values.shape != seed_values.shape:
            raise ValueError("lanes and seeds must have the same length")
        if (
            isinstance(grid_size, bool)
            or not isinstance(grid_size, int)
            or not 1 <= grid_size <= HAZARD_MAX_GRID_SIZE
        ):
            raise ValueError(
                f"grid_size must be an integer between 1 and {HAZARD_MAX_GRID_SIZE}"
            )
        payload = self._native.reset_ml_lanes_with_centered_startup(
            lane_values, seed_values, grid_size
        )
        result = _ml_result_from_payload(payload)
        self._last_result = None
        self._last_ml_result = result
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def step_ml_batch(self, actions: object) -> NativeMlBatchResult:
        """Advance lanes without rendering or materializing snapshots."""
        self._ensure_open()
        values = _integer_array(actions, "actions", maximum=8)
        payload = self._native.step_ml_batch(values)
        result = _ml_result_from_payload(payload)
        self._last_result = None
        self._last_ml_result = result
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def step_ml_positions_batch(
        self,
        actions: object,
        fallback_observations: object,
    ) -> NativeMlBatchResult:
        """Advance lanes while retaining only positions and transition fields."""
        self._ensure_open()
        values = _integer_array(actions, "actions", maximum=8)
        fallback = np.asarray(fallback_observations, dtype=np.float32)
        if fallback.shape != (self.lane_count, ML_OBSERVATION_SIZE):
            raise ValueError(
                "fallback observations must have shape "
                f"({self.lane_count}, {ML_OBSERVATION_SIZE})"
            )
        payload = self._native.step_ml_positions_batch(values)
        result = _ml_position_result_from_payload(payload, fallback)
        self._last_ml_result = result
        self._last_result = None
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def observe_ml_batch(self) -> NativeMlBatchResult:
        """Materialize the current ML observation without advancing lanes."""
        self._ensure_open()
        payload = self._native.observe_ml_batch()
        result = _ml_result_from_payload(payload)
        self._last_ml_result = result
        self._last_result = None
        self._last_pixel_result = None
        self._last_hazard_result = None
        return result

    def hazard_observations(
        self,
        grid_size: int,
        *,
        prediction_horizon_frames: int = 32,
        spawn_halo_radius: int = 1,
    ) -> NativeHazardBatchResult:
        """Return the finite slow/reference frozen-center hazard field."""
        self._ensure_open()
        if (
            isinstance(grid_size, bool)
            or not isinstance(grid_size, int)
            or not 1 <= grid_size <= HAZARD_MAX_GRID_SIZE
        ):
            raise ValueError(
                f"grid_size must be an integer between 1 and {HAZARD_MAX_GRID_SIZE}"
            )
        if (
            isinstance(prediction_horizon_frames, bool)
            or not isinstance(prediction_horizon_frames, int)
            or prediction_horizon_frames < 1
        ):
            raise ValueError("prediction_horizon_frames must be a positive integer")
        if (
            isinstance(spawn_halo_radius, bool)
            or not isinstance(spawn_halo_radius, int)
            or spawn_halo_radius < 0
        ):
            raise ValueError("spawn_halo_radius must be a non-negative integer")
        context = self._last_ml_result
        payload = self._native.hazard_observations(
            grid_size,
            prediction_horizon_frames,
            spawn_halo_radius,
        )
        result = _hazard_result_from_payload(payload, context)
        self._last_result = None
        self._last_ml_result = None
        self._last_pixel_result = None
        self._last_hazard_result = result
        return result

    def score_actions(
        self, snapshots: Sequence[bytes], lookahead_steps: int
    ) -> np.ndarray:
        """Score all nine actions from independent canonical snapshots.

        The native scorer restores each snapshot into a private game, so this
        call never advances the environment's active lanes. Scores are
        additional survival frames over the fixed native lookahead.
        """
        self._ensure_open()
        if isinstance(lookahead_steps, bool) or lookahead_steps < 1:
            raise ValueError("lookahead_steps must be a positive integer")
        values = list(snapshots)
        if not values or any(
            not isinstance(value, bytes) or not value for value in values
        ):
            raise ValueError("snapshots must be a non-empty sequence of bytes")
        payload = self._native.score_actions(values, lookahead_steps)
        scores = payload.get("scores")
        if not isinstance(scores, np.ndarray):
            raise ControlRuntimeError("native counterfactual result has no scores")
        expected_shape = (len(values), len(ACTION_CHOICES))
        if scores.shape != expected_shape:
            raise ControlRuntimeError(
                "native counterfactual scores have unexpected shape: "
                f"expected {expected_shape}, got {scores.shape}"
            )
        return np.asarray(scores, dtype=np.float32)

    def score_waypoint_actions(
        self,
        snapshots: Sequence[bytes],
        hold_decisions: int,
        *,
        grid_spacing: int = 32,
        tolerance: float = 2.0,
        arrival_latching: bool = False,
        ban_corner_nodes: bool = False,
    ) -> np.ndarray:
        """Score exact relative waypoint targets without mutating live lanes."""
        values = list(snapshots)
        if not values or any(
            not isinstance(value, bytes) or not value for value in values
        ):
            raise ValueError("snapshots must be a non-empty sequence of bytes")
        payload = self._native.score_waypoint_actions(
            values,
            hold_decisions,
            grid_spacing,
            tolerance,
            arrival_latching,
            ban_corner_nodes,
        )
        scores = payload.get("scores")
        expected = (len(values), len(ACTION_CHOICES))
        if not isinstance(scores, np.ndarray) or scores.shape != expected:
            raise ControlRuntimeError(
                f"native waypoint scores have unexpected shape: expected {expected}"
            )
        return np.asarray(scores, dtype=np.float32)

    def observe_full_state(self) -> tuple[NativeSnapshot, ...]:
        self._ensure_open()
        result = self._require_result()
        if not self.full_state_enabled:
            raise ControlRuntimeError("full-state observation was not enabled")
        return tuple(_decode_snapshot(value) for value in result.snapshot_bytes)

    def observe_pixels(self) -> np.ndarray:
        self._ensure_open()
        if self._last_pixel_result is not None:
            return self._last_pixel_result.pixels
        result = self._require_result()
        if result.pixels is None:
            raise ControlRuntimeError("pixel observation was not enabled")
        return result.pixels

    def observe_board_19x16(self) -> np.ndarray:
        self._ensure_open()
        result = self._require_result()
        if result.board is None:
            raise ControlRuntimeError("board observation was not enabled")
        return result.board

    def observe_ml(self) -> np.ndarray:
        """Return native waypoint features with shape ``(lanes, 225)``."""
        self._ensure_open()
        result = self._require_observation_result()
        if result.ml_observation is None:
            raise ControlRuntimeError("ML observation was not enabled")
        return result.ml_observation

    def observe_player_positions(self) -> np.ndarray:
        """Return native player-center coordinates with shape ``(lanes, 2)``."""
        self._ensure_open()
        if self._last_pixel_result is not None:
            return self._last_pixel_result.player_positions
        if self._last_hazard_result is not None:
            return self._last_hazard_result.player_positions
        result = self._require_observation_result()
        if result.player_positions is None:
            raise ControlRuntimeError("ML player positions were not enabled")
        return result.player_positions

    @property
    def last_result(self) -> NativeBatchResult:
        self._ensure_open()
        return self._require_result()

    @property
    def last_pixel_result(self) -> NativePixelBatchResult:
        self._ensure_open()
        if self._last_pixel_result is None:
            raise ControlRuntimeError(
                "call step_pixels() before observing the pixel-only result"
            )
        return self._last_pixel_result

    def close(self) -> None:
        self._closed = True
        self._last_result = None
        self._last_ml_result = None
        self._last_pixel_result = None
        self._last_hazard_result = None

    def _require_result(self) -> NativeBatchResult:
        if self._last_result is None:
            raise ControlRuntimeError("call reset_batch() before observing state")
        return self._last_result

    def _require_observation_result(self) -> NativeBatchResult | NativeMlBatchResult:
        if self._last_ml_result is not None:
            return self._last_ml_result
        if self._last_result is not None:
            return self._last_result
        raise ControlRuntimeError(
            "call reset_batch() or reset_ml_batch() before observing state"
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ControlRuntimeError("native batch environment is closed")

    def __enter__(self) -> NativeBatchEnvironment:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class NativeDodgeEnv:
    """Single-lane `DodgeEnv` contract backed by the native batch boundary."""

    def __init__(
        self,
        *,
        step_frames: int = 4,
        enemy_slots: int = 16,
        aoe_slots: int = 8,
        include_time_to_intersection: bool = False,
        execution: Execution = "serial",
    ) -> None:
        self.step_frames = step_frames
        self.enemy_slots = enemy_slots
        self.aoe_slots = aoe_slots
        self.include_time_to_intersection = include_time_to_intersection
        self._step_frames = step_frames
        self._batch = self._new_batch(step_frames, execution)
        self._execution = execution
        self._seed: int | None = None
        self._terminated = False

    @staticmethod
    def _new_batch(step_frames: int, execution: Execution) -> NativeBatchEnvironment:
        return NativeBatchEnvironment(
            step_frames=step_frames,
            execution=execution,
            full_state=True,
            pixels=True,
            board=True,
        )

    def reset(self, seed: int | None = None) -> Observation:
        if self._batch.closed:
            self._batch = self._new_batch(self._step_frames, self._execution)
        selected_seed = (
            np.random.default_rng().integers(0, 32_768)
            if seed is None
            else parse_seed(str(seed))
        )
        result = self._batch.reset_batch([int(selected_seed)])
        self._seed = int(selected_seed)
        self._terminated = False
        return self._observation(_decode_snapshot(_only_snapshot(result)))

    def step(self, action: str) -> Transition:
        if self._seed is None:
            raise RuntimeError("call reset() before step()")
        if self._terminated:
            raise RuntimeError("episode is complete; call reset() before step()")
        if action not in ACTION_CHOICES:
            raise ControlInputError(f"unsupported Dodge action: {action!r}")
        action_index = ACTION_CHOICES.index(cast(Any, action))
        result = self._batch.step_batch([action_index])
        snapshot = _decode_snapshot(_only_snapshot(result))
        observation = self._observation(snapshot)
        done = bool(result.done[0])
        episode_result = None
        if done:
            self._terminated = True
            episode_result = EpisodeResult(
                score=_fixed_float(snapshot.score),
                frames=snapshot.frame,
                survival_frames=snapshot.survival_frames,
                seed=snapshot.seed,
            )
        return Transition(
            observation=observation,
            reward=float(result.rewards[0]),
            done=done,
            result=episode_result,
        )

    def close(self) -> None:
        self._batch.close()
        self._seed = None
        self._terminated = True

    def __enter__(self) -> NativeDodgeEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _observation(self, snapshot: NativeSnapshot) -> Observation:
        raw_state = _raw_state_from_snapshot(snapshot)
        return Observation(
            raw_state=raw_state,
            projected=project_state(
                raw_state,
                enemy_slots=self.enemy_slots,
                aoe_slots=self.aoe_slots,
                include_time_to_intersection=self.include_time_to_intersection,
            ),
        )


def _result_from_payload(payload: MappingLike) -> NativeBatchResult:
    snapshots = tuple(
        value if isinstance(value, bytes) else None
        for value in payload["snapshot_bytes"]
    )
    return NativeBatchResult(
        lane_ids=_array(payload, "lane_ids"),
        frames=_array(payload, "frames"),
        frames_advanced=_array(payload, "frames_advanced"),
        rewards=_array(payload, "rewards"),
        done=_array(payload, "done"),
        seeds=_array(payload, "seeds"),
        state_hashes=_array(payload, "state_hashes"),
        pixel_hashes=_array(payload, "pixel_hashes"),
        modes=_array(payload, "modes"),
        event_flags=_array(payload, "event_flags"),
        pixels=_optional_array(payload, "pixels"),
        board=_optional_array(payload, "board"),
        ml_observation=_optional_array(payload, "ml_observation"),
        player_positions=_optional_array(payload, "player_positions"),
        snapshot_bytes=snapshots,
    )


def _ml_result_from_payload(payload: MappingLike) -> NativeMlBatchResult:
    return NativeMlBatchResult(
        lane_ids=_array(payload, "lane_ids"),
        frames=_array(payload, "frames"),
        frames_advanced=_array(payload, "frames_advanced"),
        rewards=_array(payload, "rewards"),
        done=_array(payload, "done"),
        seeds=_array(payload, "seeds"),
        modes=_array(payload, "modes"),
        ml_observation=_array(payload, "ml_observation"),
        player_positions=_array(payload, "player_positions"),
    )


def _ml_position_result_from_payload(
    payload: MappingLike,
    fallback_observations: np.ndarray,
) -> NativeMlBatchResult:
    """Adapt the compact native step while preserving the current observation."""
    return NativeMlBatchResult(
        lane_ids=_array(payload, "lane_ids"),
        frames=_array(payload, "frames"),
        frames_advanced=_array(payload, "frames_advanced"),
        rewards=_array(payload, "rewards"),
        done=_array(payload, "done"),
        seeds=_array(payload, "seeds"),
        modes=_array(payload, "modes"),
        ml_observation=fallback_observations,
        player_positions=_array(payload, "player_positions"),
    )


def _hazard_result_from_payload(
    payload: MappingLike,
    context: NativeMlBatchResult | None,
) -> NativeHazardBatchResult:
    lane_ids = _array(payload, "lane_ids")
    frames = _array(payload, "frames")
    lane_count = int(frames.shape[0])
    grid_size = payload.get("grid_size")
    horizon = payload.get("prediction_horizon_frames")
    halo = payload.get("spawn_halo_radius")
    channels = payload.get("hazard_channels")
    scalars = payload.get("hazard_scalars")
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (
            grid_size,
            horizon,
            halo,
            channels,
            scalars,
        )
    ):
        raise ControlRuntimeError("native hazard metadata is invalid")
    if channels != HAZARD_CHANNELS or scalars != HAZARD_SCALARS:
        raise ControlRuntimeError("native hazard channel schema is unsupported")
    expected_shape = (
        lane_count,
        HAZARD_CHANNELS * grid_size * grid_size + HAZARD_SCALARS,
    )
    hazard_observation = _array(payload, "hazard_observation")
    if hazard_observation.shape != expected_shape:
        raise ControlRuntimeError(
            "native hazard observations have unexpected shape: "
            f"expected {expected_shape}, got {hazard_observation.shape}"
        )
    if (
        hazard_observation.dtype != np.float32
        or not np.isfinite(hazard_observation).all()
    ):
        raise ControlRuntimeError("native hazard observations must be finite float32")
    player_positions = _array(payload, "player_positions")
    if player_positions.shape != (lane_count, 2):
        raise ControlRuntimeError(
            "native hazard player positions have unexpected shape"
        )
    if player_positions.dtype != np.float32 or not np.isfinite(player_positions).all():
        raise ControlRuntimeError(
            "native hazard player positions must be finite float32"
        )
    ttc_reference = _array(payload, "ttc_reference")
    if ttc_reference.shape != (lane_count, grid_size * grid_size):
        raise ControlRuntimeError("native hazard TTC reference shape is invalid")
    if ttc_reference.dtype != np.float32 or np.isnan(ttc_reference).any():
        raise ControlRuntimeError(
            "native hazard TTC reference must be float32 without NaN"
        )
    if np.any((ttc_reference < 0) & ~np.isinf(ttc_reference)):
        raise ControlRuntimeError(
            "native hazard TTC reference contains negative values"
        )
    if context is not None and context.lane_count == lane_count:
        frames_advanced = context.frames_advanced.copy()
        rewards = context.rewards.copy()
    else:
        frames_advanced = np.zeros(lane_count, dtype=np.uint32)
        rewards = np.zeros(lane_count, dtype=np.float32)
    return NativeHazardBatchResult(
        lane_ids=lane_ids,
        frames=frames,
        frames_advanced=frames_advanced,
        rewards=rewards,
        done=_array(payload, "done"),
        seeds=_array(payload, "seeds"),
        modes=_array(payload, "modes"),
        survival_frames=_array(payload, "survival_frames"),
        grid_size=grid_size,
        prediction_horizon_frames=horizon,
        spawn_halo_radius=halo,
        hazard_channels=channels,
        hazard_scalars=scalars,
        hazard_observation=hazard_observation,
        ttc_reference=ttc_reference,
        player_positions=player_positions,
    )


def _pixel_result_from_payload(payload: MappingLike) -> NativePixelBatchResult:
    return NativePixelBatchResult(
        lane_ids=_array(payload, "lane_ids"),
        frames=_array(payload, "frames"),
        frames_advanced=_array(payload, "frames_advanced"),
        rewards=_array(payload, "rewards"),
        done=_array(payload, "done"),
        seeds=_array(payload, "seeds"),
        modes=_array(payload, "modes"),
        pixels=_array(payload, "pixels"),
        player_positions=_array(payload, "player_positions"),
    )


MappingLike = dict[str, object]


def _array(payload: MappingLike, key: str) -> np.ndarray:
    value = payload.get(key)
    if not isinstance(value, np.ndarray):
        raise ControlRuntimeError(f"native batch field {key!r} is not a NumPy array")
    return value


def _optional_array(payload: MappingLike, key: str) -> np.ndarray | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, np.ndarray):
        raise ControlRuntimeError(f"native batch field {key!r} is not a NumPy array")
    return value


def _integer_array(value: object, name: str, *, maximum: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    dtype = np.uint32 if name in {"lanes", "seeds"} else np.uint8
    if np.issubdtype(array.dtype, np.unsignedinteger):
        if np.any(array > maximum):
            raise ValueError(f"{name} values must be between 0 and {maximum}")
        values = np.asarray(array, dtype=dtype)
    else:
        values = np.asarray(array, dtype=np.int64)
        if np.any(values < 0) or np.any(values > maximum):
            raise ValueError(f"{name} values must be between 0 and {maximum}")
        values = np.asarray(values, dtype=dtype)
    return np.ascontiguousarray(values)


def _only_snapshot(result: NativeBatchResult) -> bytes:
    value = result.snapshot_bytes[0]
    if value is None:
        raise ControlRuntimeError("native compatibility adapter requires full state")
    return value


def _decode_snapshot(value: bytes | None) -> NativeSnapshot:
    if value is None:
        raise ControlRuntimeError("native batch result has no full-state snapshot")
    return decode_native_snapshot(value.hex())


def _fixed_float(raw: int) -> float:
    return raw / float(1 << 16)


def _raw_state_from_snapshot(snapshot: NativeSnapshot) -> RawState:
    player_values = tuple(_fixed_float(value) for value in snapshot.player)
    player = PlayerState(*player_values)
    enemies: list[EntityState] = []
    aoes: list[EntityState] = []
    for enemy in snapshot.enemies:
        size = _fixed_float(enemy.size)
        width = 8.0 if enemy.personality >= 2 else size
        entity = EntityState(
            x=_fixed_float(enemy.x),
            y=_fixed_float(enemy.y),
            vx=_fixed_float(enemy.vx),
            vy=_fixed_float(enemy.vy),
            width=width,
            height=width,
            kind="explosion" if enemy.personality == -1 else "enemy",
            stage=0.0,
        )
        (aoes if enemy.personality == -1 else enemies).append(entity)
    if snapshot.active_pattern is not None:
        pattern = snapshot.patterns[snapshot.active_pattern]
        for rect in pattern.rects:
            aoes.append(
                EntityState(
                    x=_fixed_float(rect.x),
                    y=_fixed_float(rect.y),
                    vx=_fixed_float(rect.dx),
                    vy=_fixed_float(rect.dy),
                    width=_fixed_float(rect.width),
                    height=_fixed_float(rect.height),
                    kind="pattern",
                    stage=_fixed_float(rect.sh),
                )
            )
    particles = tuple(
        ParticleState(
            x=_fixed_float(particle.x),
            y=_fixed_float(particle.y),
            dx=_fixed_float(particle.dx),
            dy=_fixed_float(particle.dy),
            radius=_fixed_float(particle.radius),
            kind=particle.kind,
            max_age=_fixed_float(particle.max_age),
            age=float(particle.age),
            color=particle.color,
            colors=particle.colors[: particle.color_count],
        )
        for particle in snapshot.particles
    )
    return RawState(snapshot.frame, player, tuple(enemies), tuple(aoes), particles)
