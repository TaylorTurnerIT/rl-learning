from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


class ControlInputError(ValueError):
    """The movement command input does not match the public JSON interface."""


DIRECTION_KEYS: dict[str, tuple[str, ...]] = {
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


@dataclass(frozen=True, slots=True)
class MovementCommand:
    move: str
    duration_ms: int

    @property
    def keys(self) -> tuple[str, ...]:
        return DIRECTION_KEYS[self.move]


def parse_commands(value: object) -> list[MovementCommand]:
    if not isinstance(value, list):
        raise ControlInputError("commands must be a JSON list")

    commands: list[MovementCommand] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ControlInputError(f"command {index} must be an object")
        if set(item) != {"move", "duration_ms"}:
            raise ControlInputError(
                f"command {index} must contain exactly 'move' and 'duration_ms'"
            )

        move = item["move"]
        if not isinstance(move, str) or move not in DIRECTION_KEYS:
            choices = ", ".join(DIRECTION_KEYS)
            raise ControlInputError(f"command {index} move must be one of: {choices}")

        duration_ms = item["duration_ms"]
        if (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, int)
            or not 1 <= duration_ms <= 60_000
        ):
            raise ControlInputError(
                f"command {index} duration_ms must be an integer from 1 to 60000"
            )

        commands.append(MovementCommand(move=move, duration_ms=duration_ms))

    return commands


def load_commands(source: str, *, stdin: TextIO = sys.stdin) -> list[MovementCommand]:
    try:
        raw = stdin.read() if source == "-" else Path(source).read_text()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ControlInputError(f"could not read commands: {error}") from error

    return parse_commands(value)
