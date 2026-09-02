from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dodge.control import (
    CARTRIDGE_PATH,
    PEMSA_PATH,
    ControlInputError,
    ControlRuntimeError,
    MovementCommand,
    load_commands,
    parse_seed,
)
from dodge.neat.state import RawState, parse_raw_state

RESULT_PREFIX = "__dodge_result__"
STATE_PREFIX = "__dodge_state__"
PIXEL_PREFIX = "__dodge_pixel__"
FRAME_PREFIX = "__dodge_frame__"

COMMAND_MASKS: dict[str, int] = {
    "x": 32,
    "neutral": 0,
    "left": 1,
    "right": 2,
    "up": 4,
    "down": 8,
    "up_left": 5,
    "up_right": 6,
    "down_left": 9,
    "down_right": 10,
}

HeadlessResult = dict[str, int | float | bool]
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class HeadlessTrace:
    result: HeadlessResult
    states: tuple[RawState, ...]


def duration_to_frames(duration_ms: int) -> int:
    return max(1, (duration_ms * 60 + 999) // 1_000)


def instrument_cartridge(
    source: str,
    commands: list[MovementCommand],
    *,
    seed: int,
    render: bool = False,
    wait_for_game_start: bool = False,
    legacy_mouse_input: bool = False,
    capture_states: bool = False,
    capture_pixels: bool = False,
) -> str:
    if not commands:
        raise ControlInputError("headless commands must not be empty")
    if capture_pixels and not render:
        raise ControlInputError("pixel capture requires render mode")
    capture_frames = capture_states or capture_pixels

    init_marker = "function _init()\n"
    gfx_marker = "__gfx__\n"
    if source.count(init_marker) != 1:
        raise ControlRuntimeError("cartridge must contain exactly one _init function")
    if source.count(gfx_marker) != 1:
        raise ControlRuntimeError("cartridge must contain exactly one __gfx__ section")

    seeded = source.replace(init_marker, f"{init_marker} srand({seed})\n", 1)
    encoded_commands = ",".join(
        f"{{{COMMAND_MASKS[command.move]},{duration_to_frames(command.duration_ms)}}}"
        for command in commands
    )
    draw_override = (
        """__dodge_game_draw=_draw
__dodge_game_rnd=rnd
function __dodge_draw_rnd(max)
 if max then return max/2 end
 return 0.5
end
function _draw()
 if _upd!=updatetransition then
  rnd=__dodge_draw_rnd
  __dodge_game_draw()
  rnd=__dodge_game_rnd
 end
end

"""
        if render
        else "function _draw()\nend\n\n"
    )
    if capture_pixels:
        draw_override = f"""__dodge_game_draw=_draw
function __dodge_record_event(event)
 __dodge_events=__dodge_events..(__dodge_events!="" and "," or "")..event
end

__dodge_game_collide=collide
function collide(_e)
 __dodge_record_event("collision")
 __dodge_game_collide(_e)
end

__dodge_game_die=die
function die()
 __dodge_record_event("death")
 __dodge_game_die()
end

__dodge_game_addenemy=addenemy
function addenemy(_x,_y,_isdying,_s,_ms,_ep)
 __dodge_record_event("enemy_spawn")
 __dodge_game_addenemy(_x,_y,_isdying,_s,_ms,_ep)
end

function __dodge_emit_frame()
 local mode=0
 if _upd and _upd==updatemenu then mode=1
 elseif _upd and _upd==updategame then mode=2
 elseif _upd and _upd==updatesettings then mode=3
 elseif _upd and _upd==updatetransition then mode=4
 end
 local events=__dodge_events
 if cp then events=events..(events!="" and "," or "").."pattern_active" end
 printh("{FRAME_PREFIX}"..tostr(__dodge_frames).."|"..
  tostr(__dodge_mask).."|"..tostr(__dodge_previous_mask).."|"..
  tostr(mode).."|"..tostr(isdead and 1 or 0).."|"..events)
end

function __dodge_emit_pixels()
 local row=""
 for y=0,127 do
  row=""
  for x=0,127 do
   row=row..(x==0 and "" or ",")..tostr(pget(x,y))
  end
  printh("{PIXEL_PREFIX}"..tostr(__dodge_frames).."|"..tostr(y).."|"..row)
 end
end
function _draw()
 __dodge_game_draw()
 if __dodge_capture_started and __dodge_last_draw_frame!=__dodge_frames then
  __dodge_last_draw_frame=__dodge_frames
  __dodge_emit_frame()
  __dodge_emit_state()
  __dodge_emit_pixels()
  __dodge_events=""
  if __dodge_done then __dodge_emit_result() end
 end
end

"""
    transition_harness = """function __dodge_advance_transition()
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

"""
    transition_tick = (
        " if _upd and _upd==updatetransition then\n"
        "  __dodge_advance_transition()\n"
        " end\n"
    )
    state_harness = (
        f'''function __dodge_join(values)
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
 local player=tostr(px)..","..tostr(py)..","..tostr(dpx)..","..
  tostr(dpy)..","..tostr(size)
 printh("{STATE_PREFIX}"..tostr(__dodge_frames).."|"..player.."|"..
  __dodge_join(enemy_state).."|"..__dodge_join(aoe_state))
end

'''
        if capture_frames
        else ""
    )
    capture_terminal = (
        " if __dodge_capture_started then __dodge_emit_state() end\n"
        if capture_states and not capture_pixels
        else ""
    )
    capture_game_start = (
        "  __dodge_capture_started=true\n"
        + ("  __dodge_emit_state()\n" if capture_states and not capture_pixels else "")
        if capture_frames
        else ""
    )
    capture_action_complete = (
        "   if __dodge_capture_started then __dodge_emit_state() end\n"
        if capture_states and not capture_pixels
        else ""
    )
    capture_boot = " __dodge_capture_started=true\n" if capture_pixels else ""
    if capture_pixels:
        finish_definition = f'''function __dodge_emit_result()
 if __dodge_result_emitted then return end
 __dodge_result_emitted=true
 local started=hasplayed and "true" or "false"
 local result="{RESULT_PREFIX}"..tostr(score).."|"..tostr(__dodge_frames)
 result=result.."|"..tostr(__dodge_survival_frames).."|"..tostr({seed})
 result=result.."|"..started.."|true"
 printh(result)
 exit()
end

function __dodge_finish()
 __dodge_done=true
end

'''
    else:
        finish_definition = f'''function __dodge_finish()
{capture_terminal} local started=hasplayed and "true" or "false"
 local result="{RESULT_PREFIX}"..tostr(score).."|"..tostr(__dodge_frames)
 result=result.."|"..tostr(__dodge_survival_frames).."|"..tostr({seed})
 result=result.."|"..started.."|true"
 printh(result)
 exit()
end

'''
    harness = f"""__dodge_game_update60=_update60
__dodge_commands={{{encoded_commands}}}
__dodge_command=1
__dodge_remaining=__dodge_commands[1][2]
__dodge_mask=0
__dodge_previous_mask=0
__dodge_frames=0
__dodge_survival_frames=0
__dodge_wait_for_game_start={str(wait_for_game_start).lower()}
__dodge_mouse_x={64 if legacy_mouse_input else 0}
__dodge_mouse_y={64 if legacy_mouse_input else 0}
__dodge_fast_forward={str(not render).lower()}
__dodge_capture_started=false
__dodge_last_draw_frame=-1
__dodge_events=""
__dodge_done=false
__dodge_result_emitted=false

function btn(i)
 return flr(__dodge_mask/(2^i))%2==1
end

function btnp(i)
 return btn(i) and flr(__dodge_previous_mask/(2^i))%2!=1
end

__dodge_game_stat=stat
function stat(i)
 if i==32 then return __dodge_mouse_x end
 if i==33 then return __dodge_mouse_y end
 if i==34 then return 0 end
 return __dodge_game_stat(i)
end

{draw_override}{transition_harness}{state_harness}{finish_definition}

function __dodge_step()
 local command=__dodge_commands[__dodge_command]
 __dodge_mask=command and command[1] or 0
{capture_boot} if __dodge_wait_for_game_start and _upd==updategame then
  __dodge_wait_for_game_start=false
  __dodge_mask=0
  __dodge_previous_mask=0
  __dodge_command+=1
  local next_command=__dodge_commands[__dodge_command]
  if next_command then __dodge_remaining=next_command[2] end
{capture_game_start}  return
 end
 local game_frame=_upd==updategame and not isdead
 __dodge_game_update60()
 if game_frame and not isdead then
  __dodge_survival_frames+=1
 end
{transition_tick}
 __dodge_frames+=1
 __dodge_previous_mask=__dodge_mask
 if isdead then
  __dodge_finish()
  return
 end
 if command and not __dodge_wait_for_game_start then
  __dodge_remaining-=1
  if __dodge_remaining<=0 then
   __dodge_mask=0
   __dodge_previous_mask=0
   __dodge_command+=1
   local next_command=__dodge_commands[__dodge_command]
   if next_command then
    __dodge_remaining=next_command[2]
   end
{capture_action_complete}
  end
 end
end

function _update60()
 if __dodge_fast_forward then
  while not isdead do
   __dodge_step()
  end
 else
  __dodge_step()
 end
end

"""
    return seeded.replace(gfx_marker, f"{harness}{gfx_marker}", 1)


def run_headless(
    commands: list[MovementCommand],
    *,
    seed: int,
    source: Path = CARTRIDGE_PATH,
    runner: Runner = subprocess.run,
    timeout: float | None = None,
    render: bool = False,
    wait_for_game_start: bool = False,
    legacy_mouse_input: bool = False,
) -> HeadlessResult:
    try:
        original = source.read_text(encoding="utf-8")
        instrumented = instrument_cartridge(
            original,
            commands,
            seed=seed,
            render=render,
            wait_for_game_start=wait_for_game_start,
            legacy_mouse_input=legacy_mouse_input,
        )
    except OSError as error:
        raise ControlRuntimeError(f"could not read cartridge: {error}") from error

    stdout = _run_pemsa(instrumented, render=render, runner=runner, timeout=timeout)

    result_lines = [
        line.strip()
        for line in stdout.splitlines()
        if line.strip().startswith(RESULT_PREFIX)
    ]
    if len(result_lines) != 1:
        raise ControlRuntimeError("headless Pemsa did not produce a result")

    payload = result_lines[0][len(RESULT_PREFIX) :].strip()
    try:
        (
            score_raw,
            frames_raw,
            survival_frames_raw,
            seed_raw,
            started_raw,
            died_raw,
        ) = payload.split("|")
        score = json.loads(score_raw)
        frames = int(frames_raw)
        survival_frames = int(survival_frames_raw)
        result_seed = int(seed_raw)
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError("score is not numeric")
        if survival_frames < 0:
            raise ValueError("survival_frames is negative")
        if started_raw not in {"true", "false"}:
            raise ValueError("started is not boolean")
        if died_raw != "true":
            raise ValueError("died is not true")
    except (ValueError, json.JSONDecodeError) as error:
        message = "headless Pemsa produced an invalid result"
        raise ControlRuntimeError(message) from error

    return {
        "score": score,
        "frames": frames,
        "survival_frames": survival_frames,
        "seed": result_seed,
        "started": started_raw == "true",
        "died": True,
    }


def run_headless_trace(
    commands: list[MovementCommand],
    *,
    seed: int,
    source: Path = CARTRIDGE_PATH,
    runner: Runner = subprocess.run,
    timeout: float | None = None,
) -> HeadlessTrace:
    """Run an unpaced command trace and capture states at game-ready boundaries."""
    try:
        original = source.read_text(encoding="utf-8")
        instrumented = instrument_cartridge(
            original,
            commands,
            seed=seed,
            wait_for_game_start=True,
            capture_states=True,
        )
    except OSError as error:
        raise ControlRuntimeError(f"could not read cartridge: {error}") from error

    stdout = _run_pemsa(instrumented, render=False, runner=runner, timeout=timeout)
    result = _parse_result(stdout)
    states = tuple(
        parse_raw_state(line.strip(), prefix=STATE_PREFIX)
        for line in stdout.splitlines()
        if line.strip().startswith(STATE_PREFIX)
    )
    if len(states) < 2:
        raise ControlRuntimeError("headless Pemsa did not produce a state trace")
    return HeadlessTrace(result, states)


def _run_pemsa(
    instrumented: str,
    *,
    render: bool,
    runner: Runner,
    timeout: float | None,
) -> str:
    environment = os.environ.copy()
    environment["SDL_VIDEODRIVER"] = "x11" if render else "dummy"
    if render:
        environment.pop("SDL_AUDIODRIVER", None)
    else:
        environment["SDL_AUDIODRIVER"] = "dummy"
        environment["SDL_RENDER_DRIVER"] = "software"

    mode = "replay" if render else "headless"
    with tempfile.TemporaryDirectory(prefix=f"dodge-{mode}-") as directory:
        workspace = Path(directory)
        cartridge = workspace / f"dodge-{mode}.p8"
        try:
            cartridge.write_text(instrumented, encoding="utf-8")
            completed = runner(
                [PEMSA_PATH, cartridge, "--no-splash", "--no-fullscreen"],
                cwd=workspace,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stderr = _timeout_text(error.stderr)
            stdout = _timeout_text(error.stdout)
            detail = stderr.strip() or stdout.strip()
            suffix = f": {detail}" if detail else ""
            timeout_text = f"{timeout:g}" if timeout is not None else "unbounded"
            raise ControlRuntimeError(
                f"Dodge run timed out after {timeout_text}s{suffix}"
            ) from error
        except OSError as error:
            raise ControlRuntimeError(
                f"could not run headless Pemsa: {error}"
            ) from error

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ControlRuntimeError(
            f"headless Pemsa exited {completed.returncode}: {detail}"
        )
    return completed.stdout


def _parse_result(stdout: str) -> HeadlessResult:
    result_lines = [
        line.strip()
        for line in stdout.splitlines()
        if line.strip().startswith(RESULT_PREFIX)
    ]
    if len(result_lines) != 1:
        raise ControlRuntimeError("headless Pemsa did not produce a result")

    payload = result_lines[0][len(RESULT_PREFIX) :].strip()
    try:
        (
            score_raw,
            frames_raw,
            survival_frames_raw,
            seed_raw,
            started_raw,
            died_raw,
        ) = payload.split("|")
        score = json.loads(score_raw)
        frames = int(frames_raw)
        survival_frames = int(survival_frames_raw)
        result_seed = int(seed_raw)
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError("score is not numeric")
        if survival_frames < 0:
            raise ValueError("survival_frames is negative")
        if started_raw not in {"true", "false"}:
            raise ValueError("started is not boolean")
        if died_raw != "true":
            raise ValueError("died is not true")
    except (ValueError, json.JSONDecodeError) as error:
        raise ControlRuntimeError(
            "headless Pemsa produced an invalid result"
        ) from error
    return {
        "score": score,
        "frames": frames,
        "survival_frames": survival_frames,
        "seed": result_seed,
        "started": started_raw == "true",
        "died": True,
    }


def replay_commands(
    commands: list[MovementCommand],
    *,
    seed: int,
    source: Path = CARTRIDGE_PATH,
    runner: Runner = subprocess.run,
    wait_for_game_start: bool = False,
    legacy_mouse_input: bool = False,
) -> HeadlessResult:
    """Show an input-simulated winner replay; physical keyboard input is ignored."""
    return run_headless(
        commands,
        seed=seed,
        source=source,
        runner=runner,
        render=True,
        wait_for_game_start=wait_for_game_start,
        legacy_mouse_input=legacy_mouse_input,
    )


def _timeout_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-headless",
        description="Run Dodge without rendering and print final score JSON.",
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
        result = run_headless(commands, seed=seed)
    except (ControlInputError, ControlRuntimeError) as error:
        print(f"dodge-headless: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
