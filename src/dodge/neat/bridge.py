from __future__ import annotations

import os
import queue
import select
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Literal, TextIO

from dodge.control import CARTRIDGE_PATH, PEMSA_PATH, ControlRuntimeError
from dodge.neat.state import RawState, parse_raw_state

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
INPUT_PREFIX = "__dodge_neat_input__"
RELEASE_PREFIX = "__dodge_neat_release__"
STATE_PREFIX = "__dodge_neat_state__"
RESULT_PREFIX = "__dodge_neat_result__"
ACTION_KEYS: dict[Direction, tuple[str, ...]] = {
    "up_left": ("x", "Left", "Up", "x"),
    "up": ("x", "Up", "x"),
    "up_right": ("x", "Right", "Up", "x"),
    "left": ("x", "Left", "x"),
    "neutral": ("x", "x"),
    "right": ("x", "Right", "x"),
    "down_left": ("x", "Left", "Down", "x"),
    "down": ("x", "Down", "x"),
    "down_right": ("x", "Right", "Down", "x"),
}
KEY_ACK_ATTEMPTS = 3


class InputAcknowledgementTimeout(ControlRuntimeError):
    """A key event was not observed by the paused Pemsa bridge."""


@dataclass(frozen=True, slots=True)
class BridgeResult:
    state: RawState
    score: float
    frames: int
    survival_frames: int
    seed: int
    max_visible_enemies: int
    max_visible_aoes: int
    enemy_overflow_frames: int
    aoe_overflow_frames: int


def instrument_step_cartridge(
    source: str,
    *,
    seed: int,
    step_frames: int,
    enemy_slots: int = 16,
    aoe_slots: int = 8,
) -> str:
    if not 3 <= step_frames <= 5:
        raise ValueError("step_frames must be between 3 and 5")
    if enemy_slots < 1 or aoe_slots < 1:
        raise ValueError("observation slot counts must be positive")

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
__dodge_survival_frames=0
__dodge_previous_px=0
__dodge_previous_py=0
__dodge_player_vx=0
__dodge_player_vy=0
__dodge_collecting=false
__dodge_pending_mask=0
__dodge_physical_held=false
__dodge_max_enemies=0
__dodge_max_aoes=0
__dodge_enemy_overflow_frames=0
__dodge_aoe_overflow_frames=0

function btn(i)
 return flr(__dodge_mask/(2^i))%2==1
end

function btnp(i)
 return btn(i) and flr(__dodge_previous_mask/(2^i))%2!=1
end

__dodge_game_stat=stat
function stat(i)
 if i==32 or i==33 or i==34 then return 0 end
 return __dodge_game_stat(i)
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
 __dodge_emit_state()
 printh("{READY_PREFIX}"..tostr(__dodge_frames))
end

function __dodge_observe_counts()
 local enemy_count=0
 local aoe_count=0
 for entity in all(enemies) do
  if entity.p==-1 then aoe_count+=1 else enemy_count+=1 end
 end
 if cp and cp.rects then
  for rect in all(cp.rects) do aoe_count+=1 end
 end
 __dodge_max_enemies=max(__dodge_max_enemies,enemy_count)
 __dodge_max_aoes=max(__dodge_max_aoes,aoe_count)
 if enemy_count>{enemy_slots} then __dodge_enemy_overflow_frames+=1 end
 if aoe_count>{aoe_slots} then __dodge_aoe_overflow_frames+=1 end
end

function __dodge_finish()
 __dodge_emit_state()
 local result="{RESULT_PREFIX}"..tostr(score).."|"..tostr(__dodge_frames)
 result=result.."|"..tostr(__dodge_survival_frames).."|"..tostr({seed})
 result=result.."|"..tostr(__dodge_max_enemies).."|"..tostr(__dodge_max_aoes)
 result=result.."|"..tostr(__dodge_enemy_overflow_frames)
 result=result.."|"..tostr(__dodge_aoe_overflow_frames)
 printh(result)
 exit()
end

function __dodge_join(values)
 local result=""
 for value in all(values) do
  result=result..(result!="" and ";" or "")..value
 end
 return result
end

function __dodge_entity(x,y,vx,vy,w,h,kind,stage)
 return tostr(x)..","..tostr(y)..","..tostr(vx)..","..
  tostr(vy)..","..tostr(w)..","..tostr(h)..","..
  tostr(kind)..","..tostr(stage)
end

function __dodge_emit_state()
 local enemy_state={{}}
 local aoe_state={{}}
 for entity in all(enemies) do
  local width=entity.p>=2 and 8 or entity.s
  local height=entity.p>=2 and 8 or entity.s
  local value=__dodge_entity(
   entity.x,entity.y,entity.vx,entity.vy,width,height,entity.p,0)
  if entity.p==-1 then add(aoe_state,value) else add(enemy_state,value) end
 end
 if cp and cp.rects then
  for rect in all(cp.rects) do
   add(aoe_state,__dodge_entity(
    rect.x,rect.y,rect.dx or 0,rect.dy or 0,rect.w,rect.h,-2,rect.sh or 0))
  end
 end
 local player=tostr(px)..","..tostr(py)..","..tostr(__dodge_player_vx)..","..
  tostr(__dodge_player_vy)..","..tostr(size)
 printh("{STATE_PREFIX}"..tostr(__dodge_frames).."|"..player.."|"..
  __dodge_join(enemy_state).."|"..__dodge_join(aoe_state))
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
  __dodge_previous_px=px
  __dodge_previous_py=py
  __dodge_waiting=true
  __dodge_mask=0
  __dodge_ready()
  return
 end

 if __dodge_waiting then
  local physical_mask=__dodge_game_btn()
  if physical_mask==0 then
   if __dodge_physical_held then
    __dodge_physical_held=false
    printh("{RELEASE_PREFIX}"..tostr(__dodge_frames))
   end
   return
  end
  if __dodge_physical_held then return end
  __dodge_physical_held=true
  if physical_mask==32 then
   if __dodge_collecting then
    __dodge_mask=__dodge_pending_mask
    __dodge_remaining={step_frames}
    __dodge_waiting=false
    __dodge_collecting=false
    __dodge_physical_held=false
    printh("{ACCEPT_PREFIX}"..tostr(__dodge_frames))
   else
    __dodge_collecting=true
    __dodge_pending_mask=0
    printh("{INPUT_PREFIX}"..tostr(__dodge_frames))
   end
  elseif __dodge_collecting then
   __dodge_pending_mask+=physical_mask
   printh("{INPUT_PREFIX}"..tostr(__dodge_frames))
  end
  if __dodge_waiting then return end
 end

 __dodge_game_update60()
 __dodge_player_vx=px-__dodge_previous_px
 __dodge_player_vy=py-__dodge_previous_py
 __dodge_previous_px=px
 __dodge_previous_py=py
 __dodge_observe_counts()
 __dodge_previous_mask=__dodge_mask
 __dodge_frames+=1
 if not isdead then __dodge_survival_frames+=1 end
 __dodge_remaining-=1
 if isdead then __dodge_finish() end
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
        enemy_slots: int = 16,
        aoe_slots: int = 8,
        source: Path = CARTRIDGE_PATH,
        startup_timeout: float = 10.0,
    ) -> None:
        self.seed = seed
        self.step_frames = step_frames
        self.enemy_slots = enemy_slots
        self.aoe_slots = aoe_slots
        self.source = source
        self.startup_timeout = startup_timeout
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._xvfb: subprocess.Popen[str] | None = None
        self._pemsa: subprocess.Popen[str] | None = None
        self._window_id: str | None = None
        self._display_value: str | None = None
        self._lines: queue.Queue[str] = queue.Queue()

    def start(self) -> RawState:
        if self._pemsa is not None:
            raise ControlRuntimeError("Pemsa step bridge already started")
        try:
            original = self.source.read_text(encoding="utf-8")
            instrumented = instrument_step_cartridge(
                original,
                seed=self.seed,
                step_frames=self.step_frames,
                enemy_slots=self.enemy_slots,
                aoe_slots=self.aoe_slots,
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
            state = self._wait_for_ready()
            self._window_id = self._wait_for_window(display)
            return state
        except Exception:
            self.close()
            raise

    def step(self, action: Direction) -> RawState | BridgeResult:
        if self._pemsa is None or self._window_id is None:
            raise ControlRuntimeError("Pemsa step bridge is not started")
        keys = ACTION_KEYS[action]
        for index, key in enumerate(keys):
            prefix = ACCEPT_PREFIX if index == len(keys) - 1 else INPUT_PREFIX
            self._send_key_and_wait(key, prefix)
            if index != len(keys) - 1:
                self._wait_for(RELEASE_PREFIX)
        return self._wait_for_update()

    def _send_key_and_wait(self, key: str, prefix: str) -> None:
        timeout = self.startup_timeout / KEY_ACK_ATTEMPTS
        for attempt in range(KEY_ACK_ATTEMPTS):
            acknowledged = False
            terminal_accepted = False
            self._key("keydown", key)
            try:
                self._wait_for(prefix, timeout=timeout)
                acknowledged = True
                terminal_accepted = prefix == ACCEPT_PREFIX
            except InputAcknowledgementTimeout:
                if attempt == KEY_ACK_ATTEMPTS - 1:
                    raise
            finally:
                try:
                    self._key("keyup", key)
                except ControlRuntimeError:
                    if not terminal_accepted:
                        raise
            if acknowledged:
                return
        raise AssertionError("input acknowledgement retry loop exhausted")

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
                return result.stdout.split()[-1]
            self._raise_if_stopped()
            time.sleep(0.05)
        raise ControlRuntimeError("timed out waiting for hidden Pemsa window")

    def _wait_for_ready(self) -> RawState:
        update = self._wait_for_update()
        if isinstance(update, BridgeResult):
            raise ControlRuntimeError("Pemsa ended before it reached a step boundary")
        return update

    def _wait_for_update(self) -> RawState | BridgeResult:
        deadline = time.monotonic() + self.startup_timeout
        state: RawState | None = None
        while time.monotonic() < deadline:
            try:
                line = self._lines.get(timeout=0.05)
            except queue.Empty:
                self._raise_if_stopped()
                continue
            if line.startswith(STATE_PREFIX):
                state = parse_raw_state(line, prefix=STATE_PREFIX)
                continue
            if line.startswith(READY_PREFIX):
                try:
                    int(line.removeprefix(READY_PREFIX))
                except ValueError as error:
                    raise ControlRuntimeError(
                        f"invalid Pemsa ready line: {line!r}"
                    ) from error
                if state is None:
                    raise ControlRuntimeError(
                        "Pemsa step boundary did not include state"
                    )
                return state
            if line.startswith(RESULT_PREFIX):
                if state is None:
                    raise ControlRuntimeError("Pemsa result did not include state")
                return self._parse_result(line, state)
        raise ControlRuntimeError("timed out waiting for Pemsa step boundary")

    def _wait_for(self, prefix: str, *, timeout: float | None = None) -> None:
        wait_timeout = self.startup_timeout if timeout is None else timeout
        deadline = time.monotonic() + wait_timeout
        while time.monotonic() < deadline:
            self._raise_if_stopped()
            try:
                line = self._lines.get(timeout=0.05)
            except queue.Empty:
                continue
            if line.startswith(prefix):
                return
        raise InputAcknowledgementTimeout(
            f"timed out waiting for Pemsa protocol line: {prefix}"
        )

    @staticmethod
    def _parse_result(line: str, state: RawState) -> BridgeResult:
        values = line.removeprefix(RESULT_PREFIX).split("|")
        if len(values) != 8:
            raise ControlRuntimeError("invalid Pemsa terminal result field count")
        try:
            (
                score_raw,
                frames_raw,
                survival_frames_raw,
                seed_raw,
                max_enemies_raw,
                max_aoes_raw,
                enemy_overflow_raw,
                aoe_overflow_raw,
            ) = values
            result = BridgeResult(
                state=state,
                score=float(score_raw),
                frames=int(frames_raw),
                survival_frames=int(survival_frames_raw),
                seed=int(seed_raw),
                max_visible_enemies=int(max_enemies_raw),
                max_visible_aoes=int(max_aoes_raw),
                enemy_overflow_frames=int(enemy_overflow_raw),
                aoe_overflow_frames=int(aoe_overflow_raw),
            )
        except ValueError as error:
            raise ControlRuntimeError("invalid Pemsa terminal result values") from error
        if (
            result.frames < 0
            or result.survival_frames < 0
            or result.max_visible_enemies < 0
            or result.max_visible_aoes < 0
            or result.enemy_overflow_frames < 0
            or result.aoe_overflow_frames < 0
        ):
            raise ControlRuntimeError("Pemsa terminal result cannot be negative")
        return result

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
