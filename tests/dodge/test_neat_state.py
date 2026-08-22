from __future__ import annotations

from dodge.neat.state import (
    ENTITY_FEATURE_COUNT,
    OBSERVATION_SIZE,
    RawState,
    parse_raw_state,
    project_state,
)


def test_parse_raw_state_exposes_player_enemies_and_aoes() -> None:
    state = parse_raw_state(
        "__state__12|64,32,1,-0.5,4|20,30,1,2,6,6,0,0|10,11,0,0,30,30,-1,0;1,2,3,4,5,6,-2,1",
        prefix="__state__",
    )

    assert state.frame == 12
    assert state.player.x == 64
    assert state.player.vy == -0.5
    assert state.enemies[0].width == 6
    assert state.aoes[0].kind == "explosion"
    assert state.aoes[1].kind == "pattern"
    assert state.to_json()["player"] == {
        "x": 64.0,
        "y": 32.0,
        "vx": 1.0,
        "vy": -0.5,
        "size": 4.0,
    }


def test_project_state_danger_orders_and_zero_pads_slots() -> None:
    player = parse_raw_state("__x__0|0,0,0,0,4||", prefix="__x__").player
    enemies = parse_raw_state(
        "__x__0|0,0,0,0,4|40,0,1,0,4,4,0,0;10,0,-1,0,4,4,0,0|",
        prefix="__x__",
    ).enemies
    state = RawState(frame=1, player=player, enemies=enemies, aoes=())

    projected = project_state(state, enemy_slots=2, aoe_slots=1)

    assert projected.values[5] == 1.0
    assert projected.values[6] == 10 / 128
    assert projected.values[5 + ENTITY_FEATURE_COUNT] == 1.0
    assert projected.values[-ENTITY_FEATURE_COUNT:] == (0.0,) * ENTITY_FEATURE_COUNT
    assert projected.enemy_overflow is False
    assert projected.aoe_overflow is False
    assert OBSERVATION_SIZE == 197


def test_project_state_reports_overflow_without_failing() -> None:
    state = parse_raw_state(
        "__x__0|0,0,0,0,4|1,0,0,0,1,1,0,0;2,0,0,0,1,1,0,0|3,0,0,0,1,1,-1,0;4,0,0,0,1,1,-2,0",
        prefix="__x__",
    )

    projected = project_state(state, enemy_slots=1, aoe_slots=1)

    assert projected.enemy_overflow is True
    assert projected.aoe_overflow is True
