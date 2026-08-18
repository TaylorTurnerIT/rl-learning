from __future__ import annotations

import io
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dodge.control import (
    DIRECTION_KEYS,
    ControlInputError,
    ControlRuntimeError,
    MovementCommand,
    XDoToolKeyboard,
    execute_commands,
    load_commands,
    parse_commands,
)


@dataclass
class FakeProcess:
    pid: int = 42
    running: bool = True
    terminated: bool = False
    killed: bool = False
    waits: list[float | None] = field(default_factory=list)

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.terminated = True
        self.running = False

    def kill(self) -> None:
        self.killed = True
        self.running = False

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        return 0


@dataclass
class FakeKeyboard:
    window_id: str = "9001"
    events: list[tuple[str, ...]] = field(default_factory=list)
    wait_error: Exception | None = None
    fail_on_down: str | None = None

    def wait_for_window(self, pid: int, timeout: float) -> str:
        self.events.append(("wait", str(pid), str(timeout)))
        if self.wait_error is not None:
            raise self.wait_error
        return self.window_id

    def focus(self, window_id: str) -> None:
        self.events.append(("focus", window_id))

    def tap(self, window_id: str, key: str) -> None:
        self.events.append(("tap", window_id, key))

    def key_down(self, window_id: str, key: str) -> None:
        self.events.append(("down", window_id, key))
        if self.fail_on_down == key:
            raise ControlRuntimeError("injected failure")

    def key_up(self, window_id: str, key: str) -> None:
        self.events.append(("up", window_id, key))


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


def test_event_sequence_targets_window_and_releases_between_commands() -> None:
    process = FakeProcess()
    keyboard = FakeKeyboard()
    sleeps: list[float] = []

    execute_commands(
        [MovementCommand("up_right", 250), MovementCommand("neutral", 50)],
        keyboard=keyboard,
        launcher=lambda: process,
        sleep=sleeps.append,
    )

    assert keyboard.events == [
        ("wait", "42", "5.0"),
        ("wait", "42", "5.0"),
        ("focus", "9001"),
        ("tap", "9001", "x"),
        ("down", "9001", "Up"),
        ("down", "9001", "Right"),
        ("up", "9001", "Right"),
        ("up", "9001", "Up"),
    ]
    assert sleeps == [0.5, 0.75, 0.25, 0.05]
    assert process.terminated
    assert process.waits == [2.0]


def test_partial_keydown_failure_releases_held_key_and_reaps_process() -> None:
    process = FakeProcess()
    keyboard = FakeKeyboard(fail_on_down="Right")

    with pytest.raises(ControlRuntimeError, match="injected failure"):
        execute_commands(
            [MovementCommand("up_right", 10)],
            keyboard=keyboard,
            launcher=lambda: process,
            sleep=lambda _: None,
        )

    assert ("up", "9001", "Up") in keyboard.events
    assert process.terminated
    assert process.waits == [2.0]


def test_keyboard_interrupt_releases_keys_and_reaps_process() -> None:
    process = FakeProcess()
    keyboard = FakeKeyboard()
    calls = 0

    def interrupt_on_movement(_: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_commands(
            [MovementCommand("left", 10)],
            keyboard=keyboard,
            launcher=lambda: process,
            sleep=interrupt_on_movement,
        )

    assert ("up", "9001", "Left") in keyboard.events
    assert process.terminated
    assert process.waits == [2.0]


def test_window_timeout_reaps_process_without_key_events() -> None:
    process = FakeProcess()
    keyboard = FakeKeyboard(wait_error=ControlRuntimeError("window timeout"))

    with pytest.raises(ControlRuntimeError, match="window timeout"):
        execute_commands([], keyboard=keyboard, launcher=lambda: process)

    assert keyboard.events == [("wait", "42", "5.0")]
    assert process.terminated
    assert process.waits == [2.0]


def test_startup_settle_delay_precedes_first_key_event() -> None:
    process = FakeProcess()
    keyboard = FakeKeyboard()

    def record_sleep(seconds: float) -> None:
        keyboard.events.append(("sleep", str(seconds)))

    execute_commands(
        [],
        keyboard=keyboard,
        launcher=lambda: process,
        sleep=record_sleep,
    )

    assert keyboard.events == [
        ("wait", "42", "5.0"),
        ("sleep", "0.5"),
        ("wait", "42", "5.0"),
        ("focus", "9001"),
        ("tap", "9001", "x"),
        ("sleep", "0.75"),
    ]


def test_unresponsive_process_is_killed_and_reaped() -> None:
    class UnresponsiveProcess(FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            self.waits.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired("pemsa", timeout)
            return 0

    process = UnresponsiveProcess()

    execute_commands(
        [],
        keyboard=FakeKeyboard(),
        launcher=lambda: process,
        sleep=lambda _: None,
    )

    assert process.terminated
    assert process.killed
    assert process.waits == [2.0, None]


def test_xdotool_commands_always_include_target_window() -> None:
    calls: list[list[str]] = []

    def run(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    keyboard = XDoToolKeyboard(run=run)
    keyboard.focus("123")
    keyboard.tap("123", "x")
    keyboard.key_down("123", "Left")
    keyboard.key_up("123", "Left")

    assert calls == [
        ["xdotool", "windowactivate", "--sync", "123"],
        ["xdotool", "key", "--window", "123", "x"],
        ["xdotool", "keydown", "--window", "123", "Left"],
        ["xdotool", "keyup", "--window", "123", "Left"],
    ]


def test_main_validates_full_input_before_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    command_file = tmp_path / "invalid.json"
    command_file.write_text(
        json.dumps(
            [
                {"move": "left", "duration_ms": 10},
                {"move": "invalid", "duration_ms": 10},
            ]
        )
    )
    launched = False

    def unexpected_execute(*_: object, **__: object) -> None:
        nonlocal launched
        launched = True

    monkeypatch.setattr("dodge.control.execute_commands", unexpected_execute)

    from dodge.control import main

    assert main([str(command_file)]) == 1
    assert not launched
