from __future__ import annotations

import io
import json

import pytest

from dodge.control import (
    DIRECTION_KEYS,
    ControlInputError,
    MovementCommand,
    load_commands,
    parse_commands,
)


def test_all_direction_mappings() -> None:
    assert DIRECTION_KEYS == {
        "neutral": (),
        "left": ("Left",),
        "right": ("Right",),
        "up": ("Up",),
        "down": ("Down",),
        "up_left": ("Up", "Left"),
        "up_right": ("Up", "Right"),
        "down_left": ("Down", "Left"),
        "down_right": ("Down", "Right"),
    }


def test_parse_commands_preserves_order() -> None:
    assert parse_commands(
        [
            {"move": "left", "duration_ms": 250},
            {"move": "up_right", "duration_ms": 400},
            {"move": "neutral", "duration_ms": 50},
        ]
    ) == [
        MovementCommand("left", 250),
        MovementCommand("up_right", 400),
        MovementCommand("neutral", 50),
    ]


@pytest.mark.parametrize(
    "value",
    [
        {},
        ["left"],
        [{"move": "left"}],
        [{"move": "left", "duration_ms": 1, "extra": True}],
        [{"move": "sideways", "duration_ms": 1}],
        [{"move": "left", "duration_ms": True}],
        [{"move": "left", "duration_ms": 0}],
        [{"move": "left", "duration_ms": 60_001}],
    ],
)
def test_invalid_command_schema_is_rejected(value: object) -> None:
    with pytest.raises(ControlInputError):
        parse_commands(value)


def test_load_commands_reads_stdin() -> None:
    source = io.StringIO(json.dumps([{"move": "down", "duration_ms": 25}]))

    assert load_commands("-", stdin=source) == [MovementCommand("down", 25)]
