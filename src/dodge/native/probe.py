from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dodge.control import (
    ControlInputError,
    ControlRuntimeError,
    MovementCommand,
    parse_seed,
)
from dodge.headless import _run_pemsa, instrument_cartridge

PROBE_PREFIX = "__dodge_probe__"


def run_semantics_probe(
    *,
    seed: int,
    runner=None,
) -> dict[str, list[str]]:
    """Measure compatibility primitives directly in Pemsa's PICO-8 runtime."""
    source = semantics_probe_cartridge(seed)
    stdout = _run_pemsa(
        source,
        render=False,
        runner=runner or subprocess.run,
        timeout=10.0,
    )
    return parse_probe_output(stdout)


def run_input_probe(*, runner=None) -> dict[str, list[str]]:
    """Record the button and button-pressed masks for a known action schedule."""
    commands = [
        MovementCommand("x", 1),
        MovementCommand("left", 1),
        MovementCommand("neutral", 1),
        MovementCommand("up_right", 1),
        MovementCommand("down", 1),
    ]
    source = instrument_cartridge(
        input_probe_cartridge(),
        commands,
        seed=42,
        render=False,
    )
    stdout = _run_pemsa(
        source,
        render=False,
        runner=runner or subprocess.run,
        timeout=10.0,
    )
    return parse_probe_output(stdout)


def semantics_probe_cartridge(seed: int) -> str:
    return f"""pico-8 cartridge
version 42
__lua__
function _init()
 srand({seed})
 printh("{PROBE_PREFIX}|rng_first|"..tostr(rnd()))
 printh("{PROBE_PREFIX}|rng_limit|"..tostr(rnd(10)))
 printh("{PROBE_PREFIX}|numeric_floor|"..tostr(flr(-1.2)))
 printh("{PROBE_PREFIX}|numeric_ceil|"..tostr(ceil(-1.2)))
 printh("{PROBE_PREFIX}|numeric_mid|"..tostr(mid(0,9,4)))
 printh("{PROBE_PREFIX}|numeric_mod|"..tostr(7%4))
 local values={{1,2,3}}
 del(values,2)
 add(values,4)
 printh("{PROBE_PREFIX}|list_len|"..tostr(#values))
 printh("{PROBE_PREFIX}|list_1|"..tostr(values[1]))
 printh("{PROBE_PREFIX}|list_2|"..tostr(values[2]))
 printh("{PROBE_PREFIX}|list_3|"..tostr(values[3]))
 camera(3,4)
 pset(3,4,5)
 camera()
 pal(1,2)
 printh("{PROBE_PREFIX}|draw|"..tostr(pget(0,0)))
 cls(0)
 fillp(1)
 rectfill(0,0,3,3,7)
 for y=0,3 do
  for x=0,3 do
   printh("{PROBE_PREFIX}|fill_"..tostr(x).."_"..tostr(y).."|"..
    tostr(pget(x,y)))
  end
 end
 fillp()
 exit()
end
__gfx__
"""


def input_probe_cartridge() -> str:
    return f"""pico-8 cartridge
version 42
__lua__
function _init()
 score=0
 hasplayed=true
 isdead=false
 __probe_frame=0
end
function __probe_bool(value)
 return value and 1 or 0
end
function _update60()
 __probe_frame+=1
 printh("{PROBE_PREFIX}|input_frame|"..tostr(__probe_frame))
 printh("{PROBE_PREFIX}|input_btn0|"..tostr(__probe_bool(btn(0))))
 printh("{PROBE_PREFIX}|input_btn1|"..tostr(__probe_bool(btn(1))))
 printh("{PROBE_PREFIX}|input_btn2|"..tostr(__probe_bool(btn(2))))
 printh("{PROBE_PREFIX}|input_btn3|"..tostr(__probe_bool(btn(3))))
 printh("{PROBE_PREFIX}|input_btnp0|"..tostr(__probe_bool(btnp(0))))
 printh("{PROBE_PREFIX}|input_btnp1|"..tostr(__probe_bool(btnp(1))))
 printh("{PROBE_PREFIX}|input_btnp2|"..tostr(__probe_bool(btnp(2))))
 printh("{PROBE_PREFIX}|input_btnp3|"..tostr(__probe_bool(btnp(3))))
 if __probe_frame>=5 then isdead=true end
end
function _draw()
end
__gfx__
"""


def parse_probe_output(stdout: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith(PROBE_PREFIX):
            continue
        fields = line.removeprefix(PROBE_PREFIX).removeprefix("|").split("|")
        if len(fields) < 2 or not fields[0]:
            raise ControlRuntimeError("invalid compatibility probe record")
        result.setdefault(fields[0], []).append("|".join(fields[1:]))
    if not result:
        raise ControlRuntimeError("Pemsa produced no compatibility probe records")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-native-probe",
        description="Probe PICO-8 numeric, RNG, list, draw, and input semantics.",
    )
    parser.add_argument("--seed", default="42", help="PICO-8 random seed")
    parser.add_argument(
        "--output", type=Path, help="optional JSON output path; stdout by default"
    )
    arguments = parser.parse_args(argv)

    try:
        seed = parse_seed(arguments.seed)
        result = run_semantics_probe(seed=seed)
        result["input"] = run_input_probe()
        payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
        if arguments.output is None:
            sys.stdout.write(payload)
        else:
            arguments.output.write_text(payload, encoding="utf-8")
    except (ControlInputError, ControlRuntimeError, OSError) as error:
        print(f"dodge-native-probe: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
