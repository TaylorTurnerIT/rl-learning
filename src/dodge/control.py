from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO


class ControlInputError(ValueError):
    """The movement command input does not match the public JSON interface."""


class ControlRuntimeError(RuntimeError):
    """The game runner or keyboard injection backend failed."""


DIRECTION_KEYS: dict[str, tuple[str, ...]] = {
    "x": ("x",),
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARTRIDGE_PATH = PROJECT_ROOT / "src/dodge/game/dodge.p8"
PEMSA_PATH = PROJECT_ROOT / "src/dodge/runtime/pemsa"


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

    if not commands:
        raise ControlInputError("commands must not be empty")

    if commands[0].move != "x":
        raise ControlInputError("commands must start with an x move")

    return commands


def load_commands(source: str, *, stdin: TextIO = sys.stdin) -> list[MovementCommand]:
    try:
        raw = stdin.read() if source == "-" else Path(source).read_text()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ControlInputError(f"could not read commands: {error}") from error

    return parse_commands(value)


def parse_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as error:
        raise ControlInputError("seed must be an integer from 0 to 32767") from error
    if not 0 <= seed <= 32_767:
        raise ControlInputError("seed must be an integer from 0 to 32767")
    return seed


@contextmanager
def seeded_cartridge(
    seed: int | None, *, source: Path = CARTRIDGE_PATH
) -> Iterator[Path]:
    if seed is None:
        yield source
        return

    try:
        cartridge = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ControlRuntimeError(f"could not read cartridge: {error}") from error

    init_marker = "function _init()\n"
    if cartridge.count(init_marker) != 1:
        raise ControlRuntimeError("cartridge must contain exactly one _init function")
    seeded = cartridge.replace(
        init_marker,
        f"{init_marker} srand({seed})\n",
        1,
    )

    with tempfile.TemporaryDirectory(prefix="dodge-control-") as directory:
        generated = Path(directory) / "dodge-seeded.p8"
        try:
            generated.write_text(seeded, encoding="utf-8")
        except OSError as error:
            message = f"could not create seeded cartridge: {error}"
            raise ControlRuntimeError(message) from error
        yield generated


def launch_pemsa(cartridge: Path = CARTRIDGE_PATH) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment["SDL_VIDEODRIVER"] = "x11"
    try:
        return subprocess.Popen(
            [
                PEMSA_PATH,
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
    window_settle_delay: float = 0.5,
) -> None:
    process = launcher()
    window_id: str | None = None
    held_keys: list[str] = []
    try:
        window_id = keyboard.wait_for_window(process.pid, window_timeout)
        sleep(window_settle_delay)
        window_id = keyboard.wait_for_window(process.pid, window_timeout)
        keyboard.focus(window_id)

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


def control(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-control",
        description="Run Dodge from a JSON list of timed movement commands.",
    )
    parser.add_argument("source", help="JSON command file, or - to read stdin")
    parser.add_argument(
        "--seed",
        default="42",
        help="PICO-8 random seed from 0 to 32767 (default: 42)",
    )
    arguments = parser.parse_args(argv)

    try:
        seed = parse_seed(arguments.seed)
        commands = load_commands(arguments.source)
        with seeded_cartridge(seed) as cartridge:
            execute_commands(
                commands,
                keyboard=XDoToolKeyboard(),
                launcher=lambda: launch_pemsa(cartridge),
            )
    except (ControlInputError, ControlRuntimeError) as error:
        print(f"dodge-control: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("dodge-control: interrupted", file=sys.stderr)
        return 130
    return 0


def main(argv: list[str] | None = None) -> int:
    return control(argv)


if __name__ == "__main__":
    raise SystemExit(control())
