"""Fixed-N frozen-center hazard observations for waypoint DDQN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from dodge.control import ControlRuntimeError
from dodge.native.batch import (
    HAZARD_CHANNELS,
    HAZARD_MAX_GRID_SIZE,
    HAZARD_OBSERVATION_VERSION,
    HAZARD_SCALARS,
)
from dodge.ng.waypoint import WaypointGrid

HAZARD_TTC: Final[int] = 0
HAZARD_ENEMY_PRESENCE: Final[int] = 1
HAZARD_ENEMY_VX: Final[int] = 2
HAZARD_ENEMY_VY: Final[int] = 3
HAZARD_ENEMY_TYPE: Final[int] = 4
HAZARD_AOE_PRESENCE: Final[int] = 5
HAZARD_AOE_VX: Final[int] = 6
HAZARD_AOE_VY: Final[int] = 7
HAZARD_AOE_STAGE: Final[int] = 8
HAZARD_AOE_EXPLOSION: Final[int] = 9
HAZARD_AOE_PATTERN: Final[int] = 10
HAZARD_SPAWN_CORNER_MASK: Final[int] = 11
HAZARD_SPAWN_HALO: Final[int] = 12
HAZARD_PLAYER_PRESENCE: Final[int] = 13
HAZARD_PLAYER_VELOCITY_SCALE: Final[float] = 128.0
HAZARD_LEARNER_OBSERVATION_VERSION: Final[int] = 2
HazardObservationEncoding = Literal["native", "normalized_v2"]
HAZARD_OBSERVATION_ENCODINGS: Final[tuple[HazardObservationEncoding, ...]] = (
    "native",
    "normalized_v2",
)

# The normalized learner view retains the native fields that are already
# bounded, adds an explicit censored TTC bit, and expands the two packed
# categorical fields into binary/one-hot planes.  The final four values remain
# the native player x/y/vx/vy scalars so controller and replay code can share
# one tail layout.
NORMALIZED_CHANNEL_NAMES: Final[tuple[str, ...]] = (
    "ttc_normalized",
    "ttc_censored",
    "enemy_presence",
    "enemy_vx_normalized",
    "enemy_vy_normalized",
    "enemy_type_0",
    "enemy_type_1",
    "enemy_type_2",
    "enemy_type_3",
    "enemy_type_4",
    "aoe_presence",
    "aoe_vx_normalized",
    "aoe_vy_normalized",
    "aoe_stage",
    "aoe_explosion",
    "aoe_pattern",
    "spawn_corner_left_top",
    "spawn_corner_right_top",
    "spawn_corner_left_bottom",
    "spawn_corner_right_bottom",
    "spawn_halo",
    "player_presence",
)
NORMALIZED_HAZARD_CHANNELS: Final[int] = len(NORMALIZED_CHANNEL_NAMES)

CHANNEL_NAMES: Final[tuple[str, ...]] = (
    "ttc_native_frames",
    "enemy_presence",
    "enemy_vx_normalized",
    "enemy_vy_normalized",
    "enemy_type",
    "aoe_presence",
    "aoe_vx_normalized",
    "aoe_vy_normalized",
    "aoe_stage",
    "aoe_explosion",
    "aoe_pattern",
    "spawn_corner_mask",
    "spawn_halo",
    "player_presence",
)


def hazard_observation_size(
    grid_size: int,
    encoding: HazardObservationEncoding = "native",
) -> int:
    """Return flattened channel-major observation length for fixed ``N``."""
    if (
        isinstance(grid_size, bool)
        or not isinstance(grid_size, int)
        or not 1 <= grid_size <= HAZARD_MAX_GRID_SIZE
    ):
        raise ValueError(
            f"grid_size must be an integer between 1 and {HAZARD_MAX_GRID_SIZE}"
        )
    if encoding not in HAZARD_OBSERVATION_ENCODINGS:
        raise ValueError(
            "hazard observation encoding must be native or normalized_v2"
        )
    channel_count = (
        HAZARD_CHANNELS
        if encoding == "native"
        else NORMALIZED_HAZARD_CHANNELS
    )
    return channel_count * grid_size * grid_size + HAZARD_SCALARS


@dataclass(frozen=True, slots=True)
class HazardObservationSpec:
    """Fixed checkpoint schema for one hazard DDQN observation."""

    grid_size: int = 16
    prediction_horizon_frames: int = 32
    spawn_halo_radius: int = 1
    version: int = HAZARD_OBSERVATION_VERSION

    def __post_init__(self) -> None:
        hazard_observation_size(self.grid_size)
        if (
            isinstance(self.prediction_horizon_frames, bool)
            or not isinstance(self.prediction_horizon_frames, int)
            or self.prediction_horizon_frames < 1
        ):
            raise ValueError("prediction_horizon_frames must be a positive integer")
        if (
            isinstance(self.spawn_halo_radius, bool)
            or not isinstance(self.spawn_halo_radius, int)
            or self.spawn_halo_radius < 0
        ):
            raise ValueError("spawn_halo_radius must be a non-negative integer")
        if self.version != HAZARD_OBSERVATION_VERSION:
            raise ValueError("hazard observation version is unsupported")

    @property
    def observation_size(self) -> int:
        return hazard_observation_size(self.grid_size)

    def learner_observation_size(self, encoding: HazardObservationEncoding) -> int:
        """Return the model width for a native or normalized learner view."""
        return hazard_observation_size(self.grid_size, encoding)

    @property
    def grid(self) -> WaypointGrid:
        return WaypointGrid.centered(self.grid_size)

    def to_json(self) -> dict[str, object]:
        return {
            "version": self.version,
            "grid_size": self.grid_size,
            "prediction_horizon_frames": self.prediction_horizon_frames,
            "spawn_halo_radius": self.spawn_halo_radius,
            "channels": list(CHANNEL_NAMES),
            "channel_count": HAZARD_CHANNELS,
            "scalar_count": HAZARD_SCALARS,
            "encoding": "float32_channel_major_ttc_frames_spawn_mask_raw",
        }


@dataclass(frozen=True, slots=True)
class HazardObservationView:
    """Validated view of one flattened native hazard observation."""

    fields: np.ndarray
    player_scalars: np.ndarray


def decode_hazard_observation(
    observation: np.ndarray,
    spec: HazardObservationSpec,
) -> HazardObservationView:
    """Validate and split one finite native reference observation."""
    values = np.asarray(observation)
    expected = (spec.observation_size,)
    if values.shape != expected:
        raise ControlRuntimeError(
            "hazard observation has unexpected shape: "
            f"expected {expected}, got {values.shape}"
        )
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ControlRuntimeError("hazard observation must be finite float32")
    field_size = HAZARD_CHANNELS * spec.grid_size * spec.grid_size
    fields = values[:field_size].reshape(
        HAZARD_CHANNELS,
        spec.grid_size,
        spec.grid_size,
    )
    player_scalars = values[field_size:]
    return HazardObservationView(fields=fields, player_scalars=player_scalars)


def validate_native_hazard_result(
    result: object,
    spec: HazardObservationSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated field and position arrays from a native result."""
    for name, expected in (
        ("grid_size", spec.grid_size),
        ("prediction_horizon_frames", spec.prediction_horizon_frames),
        ("spawn_halo_radius", spec.spawn_halo_radius),
        ("hazard_channels", HAZARD_CHANNELS),
        ("hazard_scalars", HAZARD_SCALARS),
    ):
        if getattr(result, name, None) != expected:
            raise ControlRuntimeError(f"native hazard result has mismatched {name}")
    observations = getattr(result, "hazard_observation", None)
    ttc_reference = getattr(result, "ttc_reference", None)
    positions = getattr(result, "player_positions", None)
    lane_count = getattr(result, "lane_count", None)
    if (
        not isinstance(observations, np.ndarray)
        or not isinstance(positions, np.ndarray)
        or not isinstance(lane_count, int)
    ):
        raise ControlRuntimeError("native hazard result has invalid arrays")
    if observations.shape != (lane_count, spec.observation_size):
        raise ControlRuntimeError("native hazard observation batch shape is invalid")
    if positions.shape != (lane_count, 2):
        raise ControlRuntimeError("native hazard position batch shape is invalid")
    if not isinstance(ttc_reference, np.ndarray) or ttc_reference.shape != (
        lane_count,
        spec.grid_size * spec.grid_size,
    ):
        raise ControlRuntimeError("native hazard TTC reference batch shape is invalid")
    if observations.dtype != np.float32 or positions.dtype != np.float32:
        raise ControlRuntimeError("native hazard arrays must use float32")
    if (
        ttc_reference.dtype != np.float32
        or np.isnan(ttc_reference).any()
        or np.any((ttc_reference < 0) & ~np.isinf(ttc_reference))
    ):
        raise ControlRuntimeError(
            "native TTC reference must be float32 without invalid values"
        )
    if not np.isfinite(observations).all() or not np.isfinite(positions).all():
        raise ControlRuntimeError("native hazard arrays must be finite")
    return observations, positions


def spawn_corner_bits(view: HazardObservationView) -> np.ndarray:
    """Return raw four-bit corner labels without changing native encoding."""
    return np.rint(view.fields[HAZARD_SPAWN_CORNER_MASK]).astype(np.uint8)


def encode_hazard_observations(
    observations: np.ndarray,
    spec: HazardObservationSpec,
    encoding: HazardObservationEncoding = "native",
) -> np.ndarray:
    """Convert native hazard rows into the selected learner representation.

    Native rows use ``H + 1`` for a TTC censored at the prediction horizon.
    The normalized representation maps finite TTC to ``[0, 1]`` and emits a
    separate censored plane, avoiding the false ordering between a real hit at
    the horizon and a no-hit result.  Packed corner masks and the native
    five-level enemy kind are decoded without changing the native result.
    """
    values = np.asarray(observations)
    native_size = spec.observation_size
    if values.ndim != 2 or values.shape[1] != native_size:
        raise ControlRuntimeError(
            "hazard observation batch has unexpected shape: "
            f"expected (N, {native_size}), got {values.shape}"
        )
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ControlRuntimeError("hazard observation batch must be finite float32")
    if encoding not in HAZARD_OBSERVATION_ENCODINGS:
        raise ValueError(
            "hazard observation encoding must be native or normalized_v2"
        )
    if encoding == "native":
        return values.copy()

    cell_count = spec.grid_size * spec.grid_size
    native_fields = values[:, : HAZARD_CHANNELS * cell_count].reshape(
        len(values), HAZARD_CHANNELS, spec.grid_size, spec.grid_size
    )
    normalized_fields = np.zeros(
        (len(values), NORMALIZED_HAZARD_CHANNELS, spec.grid_size, spec.grid_size),
        dtype=np.float32,
    )
    raw_ttc = native_fields[:, HAZARD_TTC]
    normalized_fields[:, 0] = np.clip(
        raw_ttc,
        0.0,
        float(spec.prediction_horizon_frames),
    ) / float(spec.prediction_horizon_frames)
    normalized_fields[:, 1] = (
        raw_ttc > float(spec.prediction_horizon_frames)
    ).astype(np.float32)
    normalized_fields[:, 2] = native_fields[:, HAZARD_ENEMY_PRESENCE]
    normalized_fields[:, 3] = native_fields[:, HAZARD_ENEMY_VX]
    normalized_fields[:, 4] = native_fields[:, HAZARD_ENEMY_VY]

    enemy_presence = native_fields[:, HAZARD_ENEMY_PRESENCE] > 0.5
    enemy_type = np.rint(native_fields[:, HAZARD_ENEMY_TYPE] * 5.0).astype(
        np.int16
    ) - 1
    for category in range(5):
        normalized_fields[:, 5 + category] = (
            enemy_presence & (enemy_type == category)
        ).astype(np.float32)

    normalized_fields[:, 10] = native_fields[:, HAZARD_AOE_PRESENCE]
    normalized_fields[:, 11] = native_fields[:, HAZARD_AOE_VX]
    normalized_fields[:, 12] = native_fields[:, HAZARD_AOE_VY]
    normalized_fields[:, 13] = native_fields[:, HAZARD_AOE_STAGE]
    normalized_fields[:, 14] = native_fields[:, HAZARD_AOE_EXPLOSION]
    normalized_fields[:, 15] = native_fields[:, HAZARD_AOE_PATTERN]

    corner_mask = np.rint(
        native_fields[:, HAZARD_SPAWN_CORNER_MASK]
    ).astype(np.uint8)
    for bit in range(4):
        normalized_fields[:, 16 + bit] = ((corner_mask >> bit) & 1).astype(
            np.float32
        )
    normalized_fields[:, 20] = native_fields[:, HAZARD_SPAWN_HALO]
    normalized_fields[:, 21] = native_fields[:, HAZARD_PLAYER_PRESENCE]

    scalars = values[:, -HAZARD_SCALARS:]
    return np.concatenate(
        (normalized_fields.reshape(len(values), -1), scalars),
        axis=1,
    ).astype(np.float32, copy=False)


def player_velocities_from_hazard_observations(
    observations: np.ndarray,
) -> np.ndarray:
    """Decode native player velocities from the shared normalized scalar tail.

    The native hazard boundary stores player ``vx``/``vy`` divided by 128.
    The velocity-aware waypoint controller consumes native pixels per frame.
    """
    values = np.asarray(observations)
    if values.ndim != 2 or values.shape[1] < HAZARD_SCALARS:
        raise ControlRuntimeError("hazard observations have no scalar tail")
    scalars = values[:, -HAZARD_SCALARS:]
    if values.dtype != np.float32 or not np.isfinite(scalars).all():
        raise ControlRuntimeError("hazard scalar tail must be finite float32")
    return (
        scalars[:, 2:4].astype(np.float64, copy=False)
        * HAZARD_PLAYER_VELOCITY_SCALE
    )
