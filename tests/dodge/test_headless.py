from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dodge.control import ControlInputError, ControlRuntimeError, MovementCommand
from dodge.headless import (
    COMMAND_MASKS,
    PIXEL_PREFIX,
    RESULT_PREFIX,
    STATE_PREFIX,
    duration_to_frames,
    instrument_cartridge,
    replay_commands,
    run_headless,
    run_headless_trace,
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
        "o": 16,
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
    assert "__dodge_fast_forward=true" in result
    assert "function __dodge_step()" in result
    assert "while not isdead do\n   __dodge_step()\n  end" in result
    assert "function __dodge_advance_transition()" in result
    assert "_upd,_drw=updategame,drawgame" in result
    assert "function _draw()\nend" in result
    assert RESULT_PREFIX in result
    assert result.endswith("__gfx__\n")


def test_instrumented_cartridge_preserves_draw_for_visible_replay() -> None:
    source = (
        "pico-8 cartridge\nversion 42\n__lua__\n"
        "function _init()\n score=0\nend\n"
        "function _update60()\nend\n"
        "function _draw()\n cls()\nend\n__gfx__\n"
    )

    result = instrument_cartridge(source, COMMANDS, seed=42, render=True)

    assert result.count("function _draw()") == 2
    assert "__dodge_fast_forward=false" in result
    assert "else\n  __dodge_step()\n end" in result
    assert "__dodge_game_draw=_draw" in result
    assert "__dodge_game_rnd=rnd" in result
    assert "function __dodge_draw_rnd(max)" in result
    assert "__dodge_game_stat=stat" in result
    assert "function __dodge_advance_transition()" in result
    assert "if _upd and _upd==updatetransition then" in result
    assert "if _upd!=updatetransition then" in result


def test_instrumented_cartridge_full_draw_capture_reads_indexed_pixels() -> None:
    source = (
        "pico-8 cartridge\nversion 42\n__lua__\n"
        "function _init()\nend\nfunction _update60()\nend\n"
        "function _draw()\nend\n__gfx__\n"
    )

    result = instrument_cartridge(
        source,
        COMMANDS,
        seed=42,
        render=True,
        capture_pixels=True,
    )

    assert PIXEL_PREFIX in result
    assert "for y=0,127 do" in result
    assert "for x=0,127 do" in result
    assert "pget(x,y)" in result
    assert "__dodge_done=true" in result
    assert "__dodge_last_draw_frame=-1" in result
    assert "__dodge_record_event(event)" in result
    assert "if __dodge_done then __dodge_emit_result() end" in result
    assert "__dodge_capture_started=true" in result

    bounded = instrument_cartridge(
        source,
        COMMANDS,
        seed=42,
        render=True,
        capture_pixels=True,
        capture_frame_limit=1,
    )
    assert "__dodge_capture_frame_limit=1" in bounded
    with pytest.raises(ControlInputError, match="capture frame limit must be positive"):
        instrument_cartridge(
            source,
            COMMANDS,
            seed=42,
            render=True,
            capture_pixels=True,
            capture_frame_limit=0,
        )

    with pytest.raises(ControlInputError, match="pixel capture requires render"):
        instrument_cartridge(source, COMMANDS, seed=42, capture_pixels=True)


def test_instrumented_neat_replay_waits_for_the_game_before_actions() -> None:
    source = (
        "pico-8 cartridge\nversion 42\n__lua__\n"
        "function _init()\n score=0\nend\n"
        "function _update60()\nend\n"
        "function _draw()\nend\n__gfx__\n"
    )

    result = instrument_cartridge(source, COMMANDS, seed=42, wait_for_game_start=True)

    assert "__dodge_wait_for_game_start=true" in result
    assert "if __dodge_wait_for_game_start and _upd==updategame then" in result
    assert "if command and not __dodge_wait_for_game_start then" in result
    legacy_result = instrument_cartridge(
        source, COMMANDS, seed=42, legacy_mouse_input=True
    )
    assert "__dodge_mouse_x=64" in legacy_result
    assert "__dodge_mouse_y=64" in legacy_result


def test_instrumented_original_replay_can_match_native_startup_and_sample_frames() -> (
    None
):
    source = (
        "pico-8 cartridge\nversion 42\n__lua__\n"
        "function _init()\nend\nfunction _update60()\nend\n"
        "function _draw()\nend\n__gfx__\n"
    )

    result = instrument_cartridge(
        source,
        COMMANDS,
        seed=42,
        render=True,
        wait_for_game_start=True,
        capture_pixels=True,
        native_startup_grid_spacing=24,
        capture_frame_indices=[8, 4, 8],
    )

    assert "__dodge_startup_points={2,26,50,74,98,122,125}" in result
    assert "__dodge_capture_frames={[4]=true,[8]=true}" in result
    assert "__dodge_capture_frames[__dodge_frames] or __dodge_done" in result
    with pytest.raises(ControlInputError, match="requires wait_for_game_start"):
        instrument_cartridge(
            source,
            COMMANDS,
            seed=42,
            native_startup_grid_spacing=24,
        )


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
        assert environment["SDL_RENDER_DRIVER"] == "software"
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        assert kwargs["timeout"] is None
        return subprocess.CompletedProcess(
            arguments,
            0,
            f"noise\n{RESULT_PREFIX}1.5|54|12|42|true|true\t\n",
            "",
        )

    result = run_headless(COMMANDS, seed=42, source=source, runner=runner)

    assert result == {
        "score": 1.5,
        "frames": 54,
        "survival_frames": 12,
        "seed": 42,
        "started": True,
        "died": True,
    }
    assert source.read_text() == original
    assert not Path(str(observed["workspace"])).exists()
    assert not Path(str(observed["cartridge"])).exists()


def test_replay_commands_uses_visible_instrumented_input(tmp_path: Path) -> None:
    source = tmp_path / "source.p8"
    source.write_text(
        "pico-8 cartridge\nversion 42\n__lua__\n"
        "function _init()\n score=0\nend\n"
        "function _update60()\nend\n"
        "function _draw()\nend\n__gfx__\n"
    )

    def runner(
        arguments: list[object], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["SDL_VIDEODRIVER"] == "x11"
        assert environment.get("SDL_AUDIODRIVER") != "dummy"
        cartridge = Path(arguments[1])
        assert "function _draw()\nend" in cartridge.read_text()
        return subprocess.CompletedProcess(
            arguments, 0, f"{RESULT_PREFIX}0|2|1|42|true|true\n", ""
        )

    assert replay_commands(COMMANDS, seed=42, source=source, runner=runner)["died"]


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


def test_run_headless_trace_captures_game_ready_raw_states(tmp_path: Path) -> None:
    source = tmp_path / "source.p8"
    source.write_text(
        "pico-8 cartridge\nversion 42\n__lua__\n"
        "function _init()\nend\nfunction _update60()\nend\n__gfx__\n"
    )

    def runner(
        arguments: list[object], **_: object
    ) -> subprocess.CompletedProcess[str]:
        instrumented = Path(arguments[1]).read_text()
        assert "__dodge_capture_started=true" in instrumented
        assert STATE_PREFIX in instrumented
        stdout = "\n".join(
            (
                f"{STATE_PREFIX}20|64,64,0,0,4||",
                f"{STATE_PREFIX}24|62,64,-1,0,4||",
                f"{RESULT_PREFIX}0|24|4|42|true|true",
            )
        )
        return subprocess.CompletedProcess(arguments, 0, stdout, "")

    trace = run_headless_trace(COMMANDS, seed=42, source=source, runner=runner)

    assert trace.result["survival_frames"] == 4
    assert [state.frame for state in trace.states] == [20, 24]


def test_headless_result_is_json_serializable() -> None:
    result = {
        "score": 2.5,
        "frames": 100,
        "survival_frames": 40,
        "seed": 42,
        "started": True,
        "died": True,
    }

    assert json.loads(json.dumps(result)) == result
