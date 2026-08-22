from __future__ import annotations

import pytest

from dodge.control import ControlRuntimeError
from dodge.neat.bridge import (
    ACCEPT_PREFIX,
    ACTION_KEYS,
    READY_PREFIX,
    instrument_step_cartridge,
)


def test_instrument_step_cartridge_creates_exact_step_harness() -> None:
    source = "function _init()\nend\n__gfx__\n"

    result = instrument_step_cartridge(source, seed=123, step_frames=4)

    assert "srand(123)" in result
    assert "__dodge_remaining=4" in result
    assert READY_PREFIX in result
    assert ACCEPT_PREFIX in result
    assert "__dodge_game_btn=btn" in result
    assert "function _draw()\nend" in result


def test_step_actions_use_dedicated_directional_key_sets() -> None:
    assert ACTION_KEYS["up_left"] == ("Left", "Up")
    assert ACTION_KEYS["neutral"] == ("x",)
    assert ACTION_KEYS["down_right"] == ("Right", "Down")


@pytest.mark.parametrize("step_frames", [2, 6])
def test_instrument_step_cartridge_rejects_unsupported_step_frames(
    step_frames: int,
) -> None:
    with pytest.raises(ValueError, match="between 3 and 5"):
        instrument_step_cartridge(
            "function _init()\nend\n__gfx__\n", seed=1, step_frames=step_frames
        )


def test_instrument_step_cartridge_requires_cartridge_markers() -> None:
    with pytest.raises(ControlRuntimeError, match="exactly one _init"):
        instrument_step_cartridge("__gfx__\n", seed=1, step_frames=4)
