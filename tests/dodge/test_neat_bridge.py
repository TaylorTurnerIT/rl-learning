from __future__ import annotations

import pytest

from dodge.control import ControlRuntimeError
from dodge.neat.bridge import (
    ACCEPT_PREFIX,
    ACTION_KEYS,
    INPUT_PREFIX,
    READY_PREFIX,
    RELEASE_PREFIX,
    RESULT_PREFIX,
    STATE_PREFIX,
    InputAcknowledgementTimeout,
    PemsaStepBridge,
    instrument_step_cartridge,
)
from dodge.neat.state import parse_raw_state


def test_instrument_step_cartridge_creates_exact_step_harness() -> None:
    source = "function _init()\nend\n__gfx__\n"

    result = instrument_step_cartridge(source, seed=123, step_frames=4)

    assert "srand(123)" in result
    assert "__dodge_remaining=4" in result
    assert READY_PREFIX in result
    assert ACCEPT_PREFIX in result
    assert INPUT_PREFIX in result
    assert RELEASE_PREFIX in result
    assert STATE_PREFIX in result
    assert RESULT_PREFIX in result
    assert "__dodge_survival_frames" in result
    assert "__dodge_max_enemies" in result
    assert "__dodge_collecting=false\n    __dodge_physical_held=false" in result
    assert "__dodge_game_btn=btn" in result
    assert "function _draw()\nend" in result


def test_step_actions_use_dedicated_directional_key_sets() -> None:
    assert ACTION_KEYS["up_left"] == ("x", "Left", "Up", "x")
    assert ACTION_KEYS["neutral"] == ("x", "x")
    assert ACTION_KEYS["down_right"] == ("x", "Right", "Down", "x")


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


def test_v17_accepted_final_action_tolerates_destroyed_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = PemsaStepBridge(seed=1)
    bridge._pemsa = object()  # type: ignore[assignment]
    bridge._window_id = "window"
    keyup_count = 0
    state = parse_raw_state("__state__0|0,0,0,0,4||", prefix="__state__")

    def key(command: str, *_: str) -> None:
        nonlocal keyup_count
        if command == "keyup":
            keyup_count += 1
            if keyup_count == 2:
                raise ControlRuntimeError("BadWindow")

    monkeypatch.setattr(bridge, "_key", key)
    monkeypatch.setattr(bridge, "_wait_for", lambda _prefix, **_: None)
    monkeypatch.setattr(bridge, "_wait_for_update", lambda: state)

    assert bridge.step("neutral") == state


def test_v17_pre_accept_key_error_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = PemsaStepBridge(seed=1)
    bridge._pemsa = object()  # type: ignore[assignment]
    bridge._window_id = "window"

    def key(command: str, *_: str) -> None:
        if command == "keyup":
            raise ControlRuntimeError("BadWindow")

    monkeypatch.setattr(bridge, "_key", key)
    monkeypatch.setattr(bridge, "_wait_for", lambda _prefix, **_: None)

    with pytest.raises(ControlRuntimeError, match="BadWindow"):
        bridge.step("neutral")


def test_v21_retries_one_missing_input_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = PemsaStepBridge(seed=1)
    bridge._pemsa = object()  # type: ignore[assignment]
    bridge._window_id = "window"
    keys: list[tuple[str, str]] = []
    waits = 0

    def key(command: str, value: str) -> None:
        keys.append((command, value))

    def wait(prefix: str, **_: object) -> None:
        nonlocal waits
        waits += 1
        if waits == 1:
            raise InputAcknowledgementTimeout(f"timed out: {prefix}")

    monkeypatch.setattr(bridge, "_key", key)
    monkeypatch.setattr(bridge, "_wait_for", wait)

    bridge._send_key_and_wait("x", INPUT_PREFIX)

    assert keys == [
        ("keydown", "x"),
        ("keyup", "x"),
        ("keydown", "x"),
        ("keyup", "x"),
    ]
