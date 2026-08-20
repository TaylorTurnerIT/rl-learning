from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
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

RESULT_PREFIX = "__dodge_result__"

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


def duration_to_frames(duration_ms: int) -> int:
    return max(1, (duration_ms * 60 + 999) // 1_000)


def instrument_cartridge(
    source: str,
    commands: list[MovementCommand],
    *,
    seed: int,
    render: bool = False,
) -> str:
    if not commands:
        raise ControlInputError("headless commands must not be empty")

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
        " if _upd==updatetransition then\n  __dodge_advance_transition()\n end\n"
    )
    harness = f'''__dodge_game_update60=_update60
__dodge_commands={{{encoded_commands}}}
__dodge_command=1
__dodge_remaining=__dodge_commands[1][2]
__dodge_mask=0
__dodge_previous_mask=0
__dodge_frames=0
__dodge_survival_frames=0

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

{draw_override}{transition_harness}function __dodge_finish()
 local started=hasplayed and "true" or "false"
 local result="{RESULT_PREFIX}"..tostr(score).."|"..tostr(__dodge_frames)
 result=result.."|"..tostr(__dodge_survival_frames).."|"..tostr({seed})
 result=result.."|"..started.."|true"
 printh(result)
 exit()
end

function _update60()
 local command=__dodge_commands[__dodge_command]
 __dodge_mask=command and command[1] or 0
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
 end
 if command then
  __dodge_remaining-=1
  if __dodge_remaining<=0 then
   __dodge_command+=1
   local next_command=__dodge_commands[__dodge_command]
   if next_command then
    __dodge_remaining=next_command[2]
   end
  end
 end
end

'''
    return seeded.replace(gfx_marker, f"{harness}{gfx_marker}", 1)


def run_headless(
    commands: list[MovementCommand],
    *,
    seed: int,
    source: Path = CARTRIDGE_PATH,
    runner: Runner = subprocess.run,
    timeout: float | None = None,
    render: bool = False,
) -> HeadlessResult:
    try:
        original = source.read_text(encoding="utf-8")
        instrumented = instrument_cartridge(
            original, commands, seed=seed, render=render
        )
    except OSError as error:
        raise ControlRuntimeError(f"could not read cartridge: {error}") from error

    environment = os.environ.copy()
    environment["SDL_VIDEODRIVER"] = "x11" if render else "dummy"
    if render:
        environment.pop("SDL_AUDIODRIVER", None)
    else:
        environment["SDL_AUDIODRIVER"] = "dummy"

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
            message = f"could not run headless Pemsa: {error}"
            raise ControlRuntimeError(message) from error

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ControlRuntimeError(
            f"headless Pemsa exited {completed.returncode}: {detail}"
        )

    result_lines = [
        line.strip()
        for line in completed.stdout.splitlines()
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


def replay_commands(
    commands: list[MovementCommand],
    *,
    seed: int,
    source: Path = CARTRIDGE_PATH,
    runner: Runner = subprocess.run,
) -> HeadlessResult:
    """Show an input-simulated winner replay; physical keyboard input is ignored."""
    return run_headless(commands, seed=seed, source=source, runner=runner, render=True)


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
