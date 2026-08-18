from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO


class ControlInputError(ValueError):
    """The movement command input does not match the public JSON interface."""


class ControlRuntimeError(RuntimeError):
    """The game runner or keyboard injection backend failed."""


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


class Process(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class Keyboard(Protocol):
    def wait_for_window(self, pid: int, timeout: float) -> str: ...

    def focus(self, window_id: str) -> None: ...

    def tap(self, window_id: str, key: str) -> None: ...

    def key_down(self, window_id: str, key: str) -> None: ...

    def key_up(self, window_id: str, key: str) -> None: ...


class XDoToolKeyboard:
    def __init__(
        self,
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._run = run
        self._monotonic = monotonic
        self._sleep = sleep

    def wait_for_window(self, pid: int, timeout: float) -> str:
        deadline = self._monotonic() + timeout
        while self._monotonic() < deadline:
            result = self._run(
                ["xdotool", "search", "--onlyvisible", "--pid", str(pid)],
                check=False,
                capture_output=True,
                text=True,
            )
            window_ids = result.stdout.split()
            if result.returncode == 0 and window_ids:
                return window_ids[0]
            self._sleep(0.05)

        raise ControlRuntimeError(
            f"timed out after {timeout:g}s waiting for Pemsa window"
        )

    def focus(self, window_id: str) -> None:
        self._call("windowactivate", "--sync", window_id)

    def tap(self, window_id: str, key: str) -> None:
        self._call("key", "--window", window_id, key)

    def key_down(self, window_id: str, key: str) -> None:
        self._call("keydown", "--window", window_id, key)

    def key_up(self, window_id: str, key: str) -> None:
        self._call("keyup", "--window", window_id, key)

    def _call(self, *arguments: str) -> None:
        try:
            self._run(
                ["xdotool", *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            command = " ".join(arguments)
            raise ControlRuntimeError(f"xdotool {command} failed") from error


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


def launch_pemsa() -> subprocess.Popen[bytes]:
    project_root = Path(__file__).resolve().parents[2]
    executable = project_root / "src/dodge/runtime/pemsa"
    cartridge = project_root / "src/dodge/game/dodge.p8"
    environment = os.environ.copy()
    environment["SDL_VIDEODRIVER"] = "x11"
    try:
        return subprocess.Popen(
            [
                executable,
                cartridge,
                "--no-splash",
                "--no-fullscreen",
            ],
            env=environment,
        )
    except OSError as error:
        raise ControlRuntimeError(f"could not launch Pemsa: {error}") from error


def execute_commands(
    commands: list[MovementCommand],
    *,
    keyboard: Keyboard,
    launcher: Callable[[], Process] = launch_pemsa,
    sleep: Callable[[float], None] = time.sleep,
    window_timeout: float = 5.0,
    start_delay: float = 0.75,
) -> None:
    process = launcher()
    window_id: str | None = None
    held_keys: list[str] = []
    try:
        window_id = keyboard.wait_for_window(process.pid, window_timeout)
        keyboard.focus(window_id)
        keyboard.tap(window_id, "x")
        sleep(start_delay)

        for command in commands:
            for key in command.keys:
                keyboard.key_down(window_id, key)
                held_keys.append(key)
            try:
                sleep(command.duration_ms / 1_000)
            finally:
                _release_keys(keyboard, window_id, held_keys)
    finally:
        if window_id is not None:
            _release_keys(keyboard, window_id, held_keys)
        _terminate_process(process)


def _release_keys(keyboard: Keyboard, window_id: str, held_keys: list[str]) -> None:
    release_error: Exception | None = None
    while held_keys:
        key = held_keys.pop()
        try:
            keyboard.key_up(window_id, key)
        except Exception as error:  # Continue releasing every tracked key.
            release_error = release_error or error
    if release_error is not None:
        raise release_error


def _terminate_process(process: Process) -> None:
    if process.poll() is not None:
        process.wait()
        return

    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-control",
        description="Run Dodge from a JSON list of timed movement commands.",
    )
    parser.add_argument("source", help="JSON command file, or - to read stdin")
    arguments = parser.parse_args(argv)

    try:
        commands = load_commands(arguments.source)
        execute_commands(commands, keyboard=XDoToolKeyboard())
    except (ControlInputError, ControlRuntimeError) as error:
        print(f"dodge-control: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("dodge-control: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
