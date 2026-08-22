from __future__ import annotations

import os
import queue
import select
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Thread
from typing import Literal, TextIO

from dodge.control import CARTRIDGE_PATH, PEMSA_PATH, ControlRuntimeError

Direction = Literal[
    "neutral",
    "left",
    "right",
    "up",
    "down",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
]

READY_PREFIX = "__dodge_neat_ready__"
ACCEPT_PREFIX = "__dodge_neat_accept__"
ACTION_KEYS: dict[Direction, tuple[str, ...]] = {
    "up_left": ("Left", "Up"),
    "up": ("Up",),
    "up_right": ("Right", "Up"),
    "left": ("Left",),
    "neutral": ("x",),
    "right": ("Right",),
    "down_left": ("Left", "Down"),
    "down": ("Down",),
    "down_right": ("Right", "Down"),
}


def instrument_step_cartridge(source: str, *, seed: int, step_frames: int) -> str:
    if not 3 <= step_frames <= 5:
        raise ValueError("step_frames must be between 3 and 5")

    init_marker = "function _init()\n"
    gfx_marker = "__gfx__\n"
    if source.count(init_marker) != 1:
        raise ControlRuntimeError("cartridge must contain exactly one _init function")
    if source.count(gfx_marker) != 1:
        raise ControlRuntimeError("cartridge must contain exactly one __gfx__ section")

    seeded = source.replace(init_marker, f"{init_marker} srand({seed})\n", 1)
    harness = f'''__dodge_game_update60=_update60
__dodge_game_btn=btn
__dodge_mask=0
__dodge_previous_mask=0
__dodge_waiting=false
__dodge_remaining=0
__dodge_started=false
__dodge_frames=0

function btn(i)
 return flr(__dodge_mask/(2^i))%2==1
end

function btnp(i)
 return btn(i) and flr(__dodge_previous_mask/(2^i))%2!=1
end

function _draw()
end

function __dodge_advance_transition()
 trsy+=(target==2 and -10 or 10)
 if (target==2 and trsy<=-128) or (target!=2 and trsy>=128) then
  trsdone=true
  if target==0 then
   _upd,_drw=updatemenu,drawmenu
  elseif target==1 then
   _upd,_drw=updategame,drawgame
  else
   _upd,_drw=updatesettings,drawsettings
  end
 else
  trsdone=false
 end
end

function __dodge_ready()
 printh("{READY_PREFIX}"..tostr(__dodge_frames))
end

function _update60()
 if _upd!=updategame then
  __dodge_mask=32
  __dodge_game_update60()
  if _upd==updatetransition then __dodge_advance_transition() end
  __dodge_previous_mask=__dodge_mask
  __dodge_frames+=1
  return
 end

 if not __dodge_started then
  __dodge_started=true
  __dodge_waiting=true
  __dodge_mask=0
  __dodge_ready()
  return
 end

 if __dodge_waiting then
  local physical_mask=__dodge_game_btn()
  if physical_mask==0 then return end
  __dodge_mask=physical_mask==32 and 0 or physical_mask
  __dodge_remaining={step_frames}
  __dodge_waiting=false
  printh("{ACCEPT_PREFIX}"..tostr(__dodge_frames))
 end

 __dodge_game_update60()
 __dodge_previous_mask=__dodge_mask
 __dodge_frames+=1
 __dodge_remaining-=1
 if isdead then exit() end
 if __dodge_remaining<=0 then
  __dodge_mask=0
  __dodge_previous_mask=0
  __dodge_waiting=true
  __dodge_ready()
 end
end

'''
    return seeded.replace(gfx_marker, f"{harness}{gfx_marker}", 1)


class PemsaStepBridge:
    def __init__(
        self,
        *,
        seed: int,
        step_frames: int = 4,
        source: Path = CARTRIDGE_PATH,
        startup_timeout: float = 10.0,
    ) -> None:
        self.seed = seed
        self.step_frames = step_frames
        self.source = source
        self.startup_timeout = startup_timeout
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._xvfb: subprocess.Popen[str] | None = None
        self._pemsa: subprocess.Popen[str] | None = None
        self._window_id: str | None = None
        self._display_value: str | None = None
        self._lines: queue.Queue[str] = queue.Queue()

    def start(self) -> int:
        if self._pemsa is not None:
            raise ControlRuntimeError("Pemsa step bridge already started")
        try:
            original = self.source.read_text(encoding="utf-8")
            instrumented = instrument_step_cartridge(
                original,
                seed=self.seed,
                step_frames=self.step_frames,
            )
            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="dodge-neat-"
            )
            workspace = Path(self._temporary_directory.name)
            cartridge = workspace / "dodge-neat.p8"
            cartridge.write_text(instrumented, encoding="utf-8")
            display = self._start_xvfb()
            self._display_value = display
            self._pemsa = subprocess.Popen(
                [
                    "stdbuf",
                    "-oL",
                    PEMSA_PATH,
                    cartridge,
                    "--no-splash",
                    "--no-fullscreen",
                ],
                cwd=workspace,
                env=self._environment(display),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self._start_reader()
            self._window_id = self._wait_for_window(display)
            return self._wait_for_ready()
        except Exception:
            self.close()
            raise

    def step(self, action: Direction) -> int:
        if self._pemsa is None or self._window_id is None:
            raise ControlRuntimeError("Pemsa step bridge is not started")
        keys = ACTION_KEYS[action]
        self._key("keydown", *keys)
        try:
            self._wait_for_accept()
        finally:
            self._key("keyup", *reversed(keys))
        return self._wait_for_ready()

    def close(self) -> None:
        self._terminate(self._pemsa)
        self._pemsa = None
        self._terminate(self._xvfb)
        self._xvfb = None
        self._window_id = None
        self._display_value = None
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> PemsaStepBridge:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _start_xvfb(self) -> str:
        self._xvfb = subprocess.Popen(
            [
                "Xvfb",
                "-displayfd",
                "1",
                "-screen",
                "0",
                "1280x720x24",
                "-nolisten",
                "tcp",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if self._xvfb.stdout is None:
            raise ControlRuntimeError("Xvfb did not provide display output")
        display_number = self._read_line(self._xvfb.stdout)
        if not display_number.isdigit():
            raise ControlRuntimeError(
                f"Xvfb returned invalid display: {display_number!r}"
            )
        return f":{display_number}"

    def _start_reader(self) -> None:
        if self._pemsa is None or self._pemsa.stdout is None:
            raise ControlRuntimeError("Pemsa did not provide stdout")

        def read() -> None:
            for line in self._pemsa.stdout:
                self._lines.put(line.strip())

        Thread(target=read, daemon=True).start()

    def _wait_for_window(self, display: str) -> str:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._pemsa is None:
                break
            result = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--pid", str(self._pemsa.pid)],
                env=self._environment(display),
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.split():
                return result.stdout.split()[0]
            self._raise_if_stopped()
            time.sleep(0.05)
        raise ControlRuntimeError("timed out waiting for hidden Pemsa window")

    def _wait_for_ready(self) -> int:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            self._raise_if_stopped()
            try:
                line = self._lines.get(timeout=0.05)
            except queue.Empty:
                continue
            if line.startswith(READY_PREFIX):
                try:
                    return int(line.removeprefix(READY_PREFIX))
                except ValueError as error:
                    raise ControlRuntimeError(
                        f"invalid Pemsa ready line: {line!r}"
                    ) from error
        raise ControlRuntimeError("timed out waiting for Pemsa step boundary")

    def _wait_for_accept(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            self._raise_if_stopped()
            try:
                line = self._lines.get(timeout=0.05)
            except queue.Empty:
                continue
            if line.startswith(ACCEPT_PREFIX):
                return
        raise ControlRuntimeError("timed out waiting for Pemsa action acknowledgement")

    def _key(self, command: str, *keys: str) -> None:
        if self._window_id is None:
            raise ControlRuntimeError("hidden Pemsa window is unavailable")
        try:
            subprocess.run(
                ["xdotool", command, "--window", self._window_id, *keys],
                env=self._environment(self._display()),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            raise ControlRuntimeError(
                f"could not inject hidden Pemsa key: {error.stderr.strip()}"
            ) from error

    def _raise_if_stopped(self) -> None:
        if self._pemsa is not None and self._pemsa.poll() is not None:
            stderr = self._pemsa.stderr.read() if self._pemsa.stderr else ""
            raise ControlRuntimeError(f"hidden Pemsa exited: {stderr.strip()}")

    def _display(self) -> str:
        if self._xvfb is None or self._display_value is None:
            raise ControlRuntimeError("Xvfb is not started")
        return self._display_value

    @staticmethod
    def _terminate(process: subprocess.Popen[str] | None) -> None:
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

    @staticmethod
    def _read_line(stream: TextIO) -> str:
        ready, _, _ = select.select([stream], [], [], 10)
        if not ready:
            raise ControlRuntimeError("timed out waiting for Xvfb display")
        return stream.readline().strip()

    @staticmethod
    def _environment(display: str) -> dict[str, str]:
        environment = os.environ.copy()
        environment["DISPLAY"] = display
        environment["SDL_VIDEODRIVER"] = "x11"
        environment["SDL_AUDIODRIVER"] = "dummy"
        return environment
