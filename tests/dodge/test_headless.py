from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dodge.control import ControlRuntimeError, MovementCommand
from dodge.headless import (
    COMMAND_MASKS,
    RESULT_PREFIX,
    duration_to_frames,
    instrument_cartridge,
    run_headless,
)

COMMANDS = [
    MovementCommand("x", 50),
    MovementCommand("neutral", 750),
    MovementCommand("up_left", 100),
]


def test_duration_to_frames_rounds_up() -> None:
    assert duration_to_frames(1) == 1
    assert duration_to_frames(50) == 3
    assert duration_to_frames(100) == 6
    assert duration_to_frames(101) == 7


def test_command_masks_match_pico8_buttons() -> None:
    assert COMMAND_MASKS == {
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


def test_instrumented_cartridge_seeds_inputs_and_disables_draw() -> None:
    source = (
        "pico-8 cartridge // http://www.pico-8.com\n"
        "version 42\n"
        "__lua__\n"
        "function _init()\n score=0\nend\n"
        "function _update60()\n score+=1\nend\n"
        "function _draw()\n cls()\nend\n"
        "__gfx__\n"
    )

    result = instrument_cartridge(source, COMMANDS, seed=42)

    assert "function _init()\n srand(42)\n" in result
    assert "__dodge_game_update60=_update60" in result
    assert "__dodge_commands={{32,3},{0,45},{5,6}}" in result
    assert "function __dodge_advance_transition()" in result
    assert "_upd=updategame" in result
    assert "function _draw()\nend" in result
    assert RESULT_PREFIX in result
    assert result.endswith("__gfx__\n")


def test_run_headless_uses_dummy_drivers_isolated_cwd_and_parses_score(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.p8"
    original = (
        "pico-8 cartridge\nversion 42\n__lua__\n"
        "function _init()\n score=0\nend\n"
        "function _update60()\nend\n__gfx__\n"
    )
    source.write_text(original)
    observed: dict[str, object] = {}

    def runner(
        arguments: list[object], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        cartridge = Path(arguments[1])
        cwd = Path(str(kwargs["cwd"]))
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        observed["workspace"] = cwd
        observed["cartridge"] = cartridge
        assert cartridge.exists()
        assert cartridge.parent == cwd
        assert environment["SDL_VIDEODRIVER"] == "dummy"
        assert environment["SDL_AUDIODRIVER"] == "dummy"
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        return subprocess.CompletedProcess(
            arguments,
            0,
            f"noise\n{RESULT_PREFIX}1.5|54|12|42|true\t\n",
            "",
        )

    result = run_headless(COMMANDS, seed=42, source=source, runner=runner)

    assert result == {
        "score": 1.5,
        "frames": 54,
        "survival_frames": 12,
        "seed": 42,
        "started": True,
    }
    assert source.read_text() == original
    assert not Path(str(observed["workspace"])).exists()
    assert not Path(str(observed["cartridge"])).exists()


def test_run_headless_rejects_missing_result(tmp_path: Path) -> None:
    source = tmp_path / "source.p8"
    source.write_text(
        "pico-8 cartridge\nversion 42\n__lua__\n"
        "function _init()\nend\nfunction _update60()\nend\n__gfx__\n"
    )

    def runner(
        arguments: list[object], **_: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 0, "", "")

    with pytest.raises(ControlRuntimeError, match="did not produce a result"):
        run_headless(COMMANDS, seed=42, source=source, runner=runner)


def test_headless_result_is_json_serializable() -> None:
    result = {
        "score": 2.5,
        "frames": 100,
        "survival_frames": 40,
        "seed": 42,
        "started": True,
    }

    assert json.loads(json.dumps(result)) == result
