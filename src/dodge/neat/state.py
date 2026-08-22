from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Literal

from dodge.control import ControlRuntimeError

EntityKind = Literal["enemy", "explosion", "pattern"]
SCREEN_SIZE = 128.0
ENEMY_SLOT_COUNT = 16
AOE_SLOT_COUNT = 8
PLAYER_FEATURE_COUNT = 5
ENTITY_FEATURE_COUNT = 8
ENTITY_FEATURE_COUNT_WITH_TIME_TO_INTERSECTION = 9
OBSERVATION_SIZE = (
    PLAYER_FEATURE_COUNT + (ENEMY_SLOT_COUNT + AOE_SLOT_COUNT) * ENTITY_FEATURE_COUNT
)
OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION = (
    PLAYER_FEATURE_COUNT
    + (ENEMY_SLOT_COUNT + AOE_SLOT_COUNT)
    * ENTITY_FEATURE_COUNT_WITH_TIME_TO_INTERSECTION
)
TIME_TO_INTERSECTION_HORIZON = 120.0


@dataclass(frozen=True, slots=True)
class PlayerState:
    x: float
    y: float
    vx: float
    vy: float
    size: float


@dataclass(frozen=True, slots=True)
class EntityState:
    x: float
    y: float
    vx: float
    vy: float
    width: float
    height: float
    kind: EntityKind
    stage: float


@dataclass(frozen=True, slots=True)
class RawState:
    frame: int
    player: PlayerState
    enemies: tuple[EntityState, ...]
    aoes: tuple[EntityState, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "player": asdict(self.player),
            "enemies": [asdict(entity) for entity in self.enemies],
            "aoes": [asdict(entity) for entity in self.aoes],
        }


@dataclass(frozen=True, slots=True)
class ProjectedObservation:
    values: tuple[float, ...]
    enemy_overflow: bool
    aoe_overflow: bool


def parse_raw_state(line: str, *, prefix: str) -> RawState:
    values = line.removeprefix(prefix).split("|")
    if len(values) != 4:
        raise ControlRuntimeError("invalid Dodge raw state field count")
    try:
        frame = int(values[0])
        player_values = _numbers(values[1], 5)
        player = PlayerState(*player_values)
        enemies = _parse_entities(values[2], "enemy")
        aoes = _parse_entities(values[3], "aoe")
    except ValueError as error:
        raise ControlRuntimeError("invalid Dodge raw state values") from error
    return RawState(frame, player, enemies, aoes)


def project_state(
    state: RawState,
    *,
    enemy_slots: int = ENEMY_SLOT_COUNT,
    aoe_slots: int = AOE_SLOT_COUNT,
    include_time_to_intersection: bool = False,
) -> ProjectedObservation:
    if enemy_slots < 1 or aoe_slots < 1:
        raise ValueError("observation slot counts must be positive")

    enemies = _danger_order(state.player, state.enemies)
    aoes = _danger_order(state.player, state.aoes)
    values = [
        state.player.x / SCREEN_SIZE,
        state.player.y / SCREEN_SIZE,
        state.player.vx / SCREEN_SIZE,
        state.player.vy / SCREEN_SIZE,
        state.player.size / SCREEN_SIZE,
    ]
    values.extend(
        _entity_features(
            state.player,
            enemies,
            enemy_slots,
            include_time_to_intersection=include_time_to_intersection,
        )
    )
    values.extend(
        _entity_features(
            state.player,
            aoes,
            aoe_slots,
            include_time_to_intersection=include_time_to_intersection,
        )
    )
    return ProjectedObservation(
        tuple(values),
        enemy_overflow=len(enemies) > enemy_slots,
        aoe_overflow=len(aoes) > aoe_slots,
    )


def _parse_entities(
    value: str, category: Literal["enemy", "aoe"]
) -> tuple[EntityState, ...]:
    if not value:
        return ()
    return tuple(_parse_entity(entry, category) for entry in value.split(";"))


def _parse_entity(value: str, category: Literal["enemy", "aoe"]) -> EntityState:
    x, y, vx, vy, width, height, kind_code, stage = _numbers(value, 8)
    kind: EntityKind
    if category == "enemy":
        kind = "enemy"
    elif kind_code == -1:
        kind = "explosion"
    else:
        kind = "pattern"
    return EntityState(x, y, vx, vy, width, height, kind, stage)


def _numbers(value: str, expected_count: int) -> tuple[float, ...]:
    result = tuple(float(part) for part in value.split(","))
    if len(result) != expected_count:
        raise ValueError("unexpected number count")
    return result


def _danger_order(
    player: PlayerState, entities: tuple[EntityState, ...]
) -> tuple[EntityState, ...]:
    return tuple(
        sorted(
            entities,
            key=lambda entity: (
                _time_to_intersection(player, entity),
                _center_distance(player, entity),
            ),
        )
    )


def _time_to_intersection(player: PlayerState, entity: EntityState) -> float:
    player_half = player.size / 2
    entity_half_x = entity.width / 2
    entity_half_y = entity.height / 2
    return max(
        _axis_intersection_time(
            entity.x - player.x,
            entity.vx - player.vx,
            player_half + entity_half_x,
        ),
        _axis_intersection_time(
            entity.y - player.y,
            entity.vy - player.vy,
            player_half + entity_half_y,
        ),
    )


def _axis_intersection_time(
    distance: float, velocity: float, combined_half: float
) -> float:
    if abs(distance) <= combined_half:
        return 0.0
    if velocity == 0 or distance * velocity >= 0:
        return inf
    return max(0.0, (abs(distance) - combined_half) / abs(velocity))


def _center_distance(player: PlayerState, entity: EntityState) -> float:
    return (entity.x - player.x) ** 2 + (entity.y - player.y) ** 2


def _entity_features(
    player: PlayerState,
    entities: tuple[EntityState, ...],
    slot_count: int,
    *,
    include_time_to_intersection: bool,
) -> list[float]:
    values: list[float] = []
    for entity in entities[:slot_count]:
        values.append(1.0)
        if include_time_to_intersection:
            values.append(_normalized_time_to_intersection(player, entity))
        values.extend(
            (
                (entity.x - player.x) / SCREEN_SIZE,
                (entity.y - player.y) / SCREEN_SIZE,
                entity.vx / SCREEN_SIZE,
                entity.vy / SCREEN_SIZE,
                entity.width / SCREEN_SIZE,
                entity.height / SCREEN_SIZE,
                entity.stage / 2,
            )
        )
    feature_count = (
        ENTITY_FEATURE_COUNT_WITH_TIME_TO_INTERSECTION
        if include_time_to_intersection
        else ENTITY_FEATURE_COUNT
    )
    values.extend([0.0] * feature_count * (slot_count - min(len(entities), slot_count)))
    return values


def _normalized_time_to_intersection(player: PlayerState, entity: EntityState) -> float:
    time_to_intersection = _time_to_intersection(player, entity)
    if time_to_intersection == inf:
        return 1.0
    return min(time_to_intersection / TIME_TO_INTERSECTION_HORIZON, 1.0)
