from __future__ import annotations

import json
import math
from json import JSONDecodeError
from typing import cast

import numpy as np

from dodge.control import ControlRuntimeError
from dodge.neat.state import (
    SCREEN_SIZE,
    EntityKind,
    EntityState,
    PlayerState,
    RawState,
)

BOARD_SIZE = 16
PLAYER_CHANNEL_NAMES = (
    "player_presence",
    "player_vx",
    "player_vy",
    "player_width",
    "player_height",
)
ENEMY_CHANNEL_NAMES = (
    "enemy_presence",
    "enemy_vx",
    "enemy_vy",
    "enemy_width",
    "enemy_height",
    "enemy_stage",
)
AOE_CHANNEL_NAMES = (
    "aoe_presence",
    "aoe_vx",
    "aoe_vy",
    "aoe_width",
    "aoe_height",
    "aoe_stage",
    "aoe_explosion",
    "aoe_pattern",
)
BOARD_CHANNEL_NAMES = (
    *PLAYER_CHANNEL_NAMES,
    *ENEMY_CHANNEL_NAMES,
    *AOE_CHANNEL_NAMES,
)
BOARD_CHANNELS = len(BOARD_CHANNEL_NAMES)
BOARD_SHAPE = (BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE)
FULL_BOARD_POSITION_CHANNEL_NAMES = (
    "enemy_x",
    "enemy_y",
    "aoe_x",
    "aoe_y",
)
FULL_BOARD_CHANNEL_NAMES = (*BOARD_CHANNEL_NAMES, *FULL_BOARD_POSITION_CHANNEL_NAMES)
FULL_BOARD_CHANNELS = len(FULL_BOARD_CHANNEL_NAMES)
FULL_BOARD_SHAPE = (FULL_BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE)

PLAYER_PRESENCE, PLAYER_VX, PLAYER_VY, PLAYER_WIDTH, PLAYER_HEIGHT = range(5)
(
    ENEMY_PRESENCE,
    ENEMY_VX,
    ENEMY_VY,
    ENEMY_WIDTH,
    ENEMY_HEIGHT,
    ENEMY_STAGE,
) = range(5, 11)
(
    AOE_PRESENCE,
    AOE_VX,
    AOE_VY,
    AOE_WIDTH,
    AOE_HEIGHT,
    AOE_STAGE,
    AOE_EXPLOSION,
    AOE_PATTERN,
) = range(11, 19)
ENEMY_X, ENEMY_Y, AOE_X, AOE_Y = range(19, 23)


def raw_state_from_json(encoded: str) -> RawState:
    """Decode the raw state JSON stored beside each learned decision."""
    try:
        payload = _mapping(json.loads(encoded))
        player = _parse_player(payload["player"])
        enemies = _parse_entities(payload["enemies"])
        aoes = _parse_entities(payload["aoes"])
        frame = _integer(payload["frame"])
    except (JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ControlRuntimeError("collector raw state JSON is invalid") from error
    return RawState(frame, player, enemies, aoes)


def encode_board(
    state: RawState,
    *,
    include_offscreen: bool = False,
    preserve_coordinates: bool = False,
) -> np.ndarray:
    """Rasterize the complete raw board state into CNN-friendly channels."""
    if preserve_coordinates and not include_offscreen:
        raise ValueError("coordinate preservation requires offscreen entities")
    board = np.zeros(
        FULL_BOARD_SHAPE if preserve_coordinates else BOARD_SHAPE,
        dtype=np.float32,
    )
    velocity_sums = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    velocity_counts = np.zeros_like(velocity_sums)

    _paint_entity(
        board,
        velocity_sums,
        velocity_counts,
        x=state.player.x,
        y=state.player.y,
        width=state.player.size,
        height=state.player.size,
        vx=state.player.vx,
        vy=state.player.vy,
        stage=None,
        kind_channel=None,
        presence_channel=PLAYER_PRESENCE,
        vx_channel=PLAYER_VX,
        vy_channel=PLAYER_VY,
        width_channel=PLAYER_WIDTH,
        height_channel=PLAYER_HEIGHT,
        stage_channel=None,
        velocity_slot=0,
        include_offscreen=include_offscreen,
        position_channels=None,
    )
    for entity in state.enemies:
        _paint_entity(
            board,
            velocity_sums,
            velocity_counts,
            x=entity.x,
            y=entity.y,
            width=entity.width,
            height=entity.height,
            vx=entity.vx,
            vy=entity.vy,
            stage=entity.stage,
            kind_channel=None,
            presence_channel=ENEMY_PRESENCE,
            vx_channel=ENEMY_VX,
            vy_channel=ENEMY_VY,
            width_channel=ENEMY_WIDTH,
            height_channel=ENEMY_HEIGHT,
            stage_channel=ENEMY_STAGE,
            velocity_slot=2,
            include_offscreen=include_offscreen,
            position_channels=(ENEMY_X, ENEMY_Y) if preserve_coordinates else None,
        )
    for entity in state.aoes:
        _paint_entity(
            board,
            velocity_sums,
            velocity_counts,
            x=entity.x,
            y=entity.y,
            width=entity.width,
            height=entity.height,
            vx=entity.vx,
            vy=entity.vy,
            stage=entity.stage,
            kind_channel=(AOE_EXPLOSION if entity.kind == "explosion" else AOE_PATTERN),
            presence_channel=AOE_PRESENCE,
            vx_channel=AOE_VX,
            vy_channel=AOE_VY,
            width_channel=AOE_WIDTH,
            height_channel=AOE_HEIGHT,
            stage_channel=AOE_STAGE,
            velocity_slot=4,
            include_offscreen=include_offscreen,
            position_channels=(AOE_X, AOE_Y) if preserve_coordinates else None,
        )

    for channel, slot in (
        (PLAYER_VX, 0),
        (PLAYER_VY, 1),
        (ENEMY_VX, 2),
        (ENEMY_VY, 3),
        (AOE_VX, 4),
        (AOE_VY, 5),
    ):
        np.divide(
            velocity_sums[slot],
            velocity_counts[slot],
            out=board[channel],
            where=velocity_counts[slot] > 0,
        )
    return board


def _paint_entity(
    board: np.ndarray,
    velocity_sums: np.ndarray,
    velocity_counts: np.ndarray,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    vx: float,
    vy: float,
    stage: float | None,
    kind_channel: int | None,
    presence_channel: int,
    vx_channel: int,
    vy_channel: int,
    width_channel: int,
    height_channel: int,
    stage_channel: int | None,
    velocity_slot: int,
    include_offscreen: bool,
    position_channels: tuple[int, int] | None,
) -> None:
    bounds = _cell_bounds(x, y, width, height, include_offscreen=include_offscreen)
    if bounds is None:
        return
    top, bottom, left, right = bounds
    normalized_width = width / SCREEN_SIZE
    normalized_height = height / SCREEN_SIZE
    normalized_vx = vx / SCREEN_SIZE
    normalized_vy = vy / SCREEN_SIZE
    normalized_stage = None if stage is None else stage / 2
    normalized_x = 0.5 + x / SCREEN_SIZE
    normalized_y = 0.5 + y / SCREEN_SIZE
    for row in range(top, bottom + 1):
        for column in range(left, right + 1):
            board[presence_channel, row, column] = 1.0
            board[width_channel, row, column] = max(
                board[width_channel, row, column], normalized_width
            )
            board[height_channel, row, column] = max(
                board[height_channel, row, column], normalized_height
            )
            velocity_sums[velocity_slot, row, column] += normalized_vx
            velocity_sums[velocity_slot + 1, row, column] += normalized_vy
            velocity_counts[velocity_slot, row, column] += 1
            velocity_counts[velocity_slot + 1, row, column] += 1
            if stage_channel is not None and normalized_stage is not None:
                board[stage_channel, row, column] = max(
                    board[stage_channel, row, column], normalized_stage
                )
            if kind_channel is not None:
                board[kind_channel, row, column] = 1.0
            if position_channels is not None:
                x_channel, y_channel = position_channels
                board[x_channel, row, column] = max(
                    board[x_channel, row, column], normalized_x
                )
                board[y_channel, row, column] = max(
                    board[y_channel, row, column], normalized_y
                )


def _cell_bounds(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    include_offscreen: bool = False,
) -> tuple[int, int, int, int] | None:
    scale = BOARD_SIZE / SCREEN_SIZE
    left = math.floor((x - width / 2) * scale)
    right = math.floor((x + width / 2) * scale)
    top = math.floor((y - height / 2) * scale)
    bottom = math.floor((y + height / 2) * scale)
    if not include_offscreen and (
        right < 0 or left >= BOARD_SIZE or bottom < 0 or top >= BOARD_SIZE
    ):
        return None
    if right < 0:
        left = right = 0
    elif left >= BOARD_SIZE:
        left = right = BOARD_SIZE - 1
    else:
        left = max(0, left)
        right = min(BOARD_SIZE - 1, right)
    if bottom < 0:
        top = bottom = 0
    elif top >= BOARD_SIZE:
        top = bottom = BOARD_SIZE - 1
    else:
        top = max(0, top)
        bottom = min(BOARD_SIZE - 1, bottom)
    return (
        top,
        bottom,
        left,
        right,
    )


def _parse_player(value: object) -> PlayerState:
    payload = _mapping(value)
    return PlayerState(
        _number(payload["x"]),
        _number(payload["y"]),
        _number(payload["vx"]),
        _number(payload["vy"]),
        _number(payload["size"]),
    )


def _parse_entities(value: object) -> tuple[EntityState, ...]:
    return tuple(_parse_entity(item) for item in _sequence(value))


def _parse_entity(value: object) -> EntityState:
    payload = _mapping(value)
    kind_value = payload["kind"]
    if kind_value not in {"enemy", "explosion", "pattern"}:
        raise ValueError("unknown entity kind")
    return EntityState(
        _number(payload["x"]),
        _number(payload["y"]),
        _number(payload["vx"]),
        _number(payload["vy"]),
        _number(payload["width"]),
        _number(payload["height"]),
        cast(EntityKind, kind_value),
        _number(payload["stage"]),
    )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected JSON list")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("expected JSON number")
    return float(value)


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected JSON integer")
    return value
