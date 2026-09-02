from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from dodge.control import ControlRuntimeError, MovementCommand
from dodge.native.oracle import XvfbProcess, run_oracle_trace, write_trace
from dodge.native.trace import (
    PIXEL_HEIGHT,
    PIXEL_WIDTH,
    parse_full_draw_stdout,
    parse_pixel_line,
)


def _pixel_rows(frame: int) -> list[str]:
    row = ",".join(str(index % 16) for index in range(PIXEL_WIDTH))
    return [f"__dodge_pixel__{frame}|{index}|{row}" for index in range(PIXEL_HEIGHT)]


def _stdout(*frames: int, terminal: int) -> str:
    lines: list[str] = []
    for frame in frames:
        reward = 0 if frame == terminal else 1
        lines.append(
            f"__dodge_frame__{frame}|32|0|2|{int(frame == terminal)}|{reward}|"
        )
        lines.append(f"__dodge_state__{frame}|64,64,0,0,4||")
        lines.extend(_pixel_rows(frame))
    lines.append(f"__dodge_result__0|{terminal}|{terminal - 1}|42|true|true")
    return "\n".join(lines) + "\n"


def test_parse_pixel_line_accepts_all_palette_indexes() -> None:
    frame, row, pixels = parse_pixel_line(_pixel_rows(12)[7])

    assert (frame, row) == (12, 7)
    assert len(pixels) == PIXEL_WIDTH
    assert pixels[:16] == tuple(range(16))


def test_parse_full_draw_stdout_builds_post_draw_terminal_frame() -> None:
    frames, result = parse_full_draw_stdout(_stdout(3, 8, terminal=8))

    assert result["frames"] == 8
    assert [frame.frame_index for frame in frames] == [3, 8]
    assert frames[0].done is False
    assert frames[0].reward == 1
    assert frames[1].done is True
    assert frames[1].reward == 0
    assert frames[1].input_mask == 32
    assert frames[1].mode == 2
    assert frames[1].dead is True
    assert frames[1].events == ("terminal",)
    assert len(frames[1].pixels) == 128 * 128
    assert frames[1].pixel_sha256 == hashlib.sha256(frames[1].pixels).hexdigest()


@pytest.mark.parametrize(
    ("bad_line", "message"),
    [
        ("__dodge_pixel__1|0|0", "128 values"),
        (
            "__dodge_pixel__1|0|" + ",".join("16" for _ in range(PIXEL_WIDTH)),
            "palette indexes",
        ),
    ],
)
def test_parse_pixel_line_rejects_noncanonical_rows(
    bad_line: str, message: str
) -> None:
    with pytest.raises(ControlRuntimeError, match=message):
        parse_pixel_line(bad_line)


def test_parse_full_draw_stdout_rejects_incomplete_frame() -> None:
    output = _stdout(3, terminal=3).replace(_pixel_rows(3)[-1] + "\n", "", 1)

    with pytest.raises(ControlRuntimeError, match="128 pixel rows"):
        parse_full_draw_stdout(output)


class _FakeXvfb:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.waited = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return self.returncode or 0


def test_run_oracle_trace_isolated_and_reaps_xvfb(tmp_path: Path) -> None:
    source = tmp_path / "source.p8"
    pemsa = tmp_path / "pemsa"
    source.write_text(
        "pico-8 cartridge\nversion 42\n__lua__\n"
        "function _init()\nend\nfunction _update60()\nend\n"
        "function _draw()\nend\n__gfx__\n"
    )
    pemsa.write_bytes(b"fake pemsa")
    fake_xvfb = _FakeXvfb()
    observed: dict[str, object] = {}

    def start_xvfb(timeout: float) -> XvfbProcess:
        observed["xvfb_timeout"] = timeout
        return XvfbProcess(fake_xvfb, ":99")  # type: ignore[arg-type]

    def runner(
        arguments: list[object], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        observed["cwd"] = kwargs["cwd"]
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert environment["DISPLAY"] == ":99"
        assert environment["SDL_VIDEODRIVER"] == "x11"
        assert environment["SDL_AUDIODRIVER"] == "dummy"
        assert environment["SDL_RENDER_DRIVER"] == "software"
        cartridge = Path(str(arguments[1]))
        assert cartridge.exists()
        assert cartridge.parent == kwargs["cwd"]
        assert "pget(x,y)" in cartridge.read_text()
        return subprocess.CompletedProcess(
            arguments,
            0,
            _stdout(7, terminal=7),
            "",
        )

    trace = run_oracle_trace(
        [MovementCommand("x", 50), MovementCommand("neutral", 100)],
        seed=42,
        source=source,
        pemsa=pemsa,
        runner=runner,
        start_xvfb=start_xvfb,
        timeout=12.0,
    )

    assert trace.provenance["capture_mode"] == "full_draw"
    assert trace.scenario["seed"] == 42
    assert len(trace.frames) == 1
    assert fake_xvfb.terminated is True
    assert fake_xvfb.waited is True
    assert not Path(str(observed["cwd"])).exists()

    output = tmp_path / "trace.json"
    write_trace(output, trace)
    parsed = json.loads(output.read_text())
    assert parsed["trace_type"] == "dodge.native.full_draw"
    assert parsed["frames"][0]["pixels"]["width"] == 128
    assert not list(tmp_path.glob(".trace.json.*.tmp"))


def test_run_oracle_trace_timeout_reaps_xvfb(tmp_path: Path) -> None:
    source = tmp_path / "source.p8"
    pemsa = tmp_path / "pemsa"
    source.write_text(
        "pico-8 cartridge\nversion 42\n__lua__\nfunction _init()\nend\n__gfx__\n"
    )
    pemsa.write_bytes(b"fake pemsa")
    fake_xvfb = _FakeXvfb()

    def runner(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired("pemsa", 0.01, stderr="timed out")

    with pytest.raises(ControlRuntimeError, match="timed out"):
        run_oracle_trace(
            [MovementCommand("x", 1)],
            seed=42,
            source=source,
            pemsa=pemsa,
            runner=runner,
            start_xvfb=lambda _: XvfbProcess(fake_xvfb, ":98"),  # type: ignore[arg-type]
            timeout=0.01,
        )

    assert fake_xvfb.terminated is True
    assert fake_xvfb.waited is True
