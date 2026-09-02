from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from dodge.control import ControlRuntimeError, MovementCommand
from dodge.headless import (
    FRAME_PREFIX,
    PIXEL_PREFIX,
    RESULT_PREFIX,
    STATE_PREFIX,
    _parse_result,
)
from dodge.native.manifest import canonical_json
from dodge.neat.state import RawState, parse_raw_state

TRACE_SCHEMA_VERSION = 1
TRACE_TYPE = "dodge.native.full_draw"
CAPTURE_MODE = "full_draw"
PIXEL_ENCODING = "palette_index_u8_row_major"
PIXEL_WIDTH = 128
PIXEL_HEIGHT = 128
PIXEL_COUNT = PIXEL_WIDTH * PIXEL_HEIGHT


@dataclass(frozen=True, slots=True)
class OracleFrame:
    frame_index: int
    state: RawState
    pixels: bytes
    done: bool
    reward: float = 0.0
    events: tuple[str, ...] = ()
    input_mask: int = 0
    previous_input_mask: int = 0
    mode: int = 0
    dead: bool = False

    @property
    def state_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json(self.state.to_json()).encode("utf-8")
        ).hexdigest()

    @property
    def pixel_sha256(self) -> str:
        return hashlib.sha256(self.pixels).hexdigest()

    def to_json(self) -> dict[str, object]:
        return {
            "frame": self.frame_index,
            "state": self.state.to_json(),
            "reward": self.reward,
            "done": self.done,
            "events": list(self.events),
            "input": {
                "mask": self.input_mask,
                "previous_mask": self.previous_input_mask,
                "mode": self.mode,
                "dead": self.dead,
            },
            "pixels": {
                "encoding": PIXEL_ENCODING,
                "width": PIXEL_WIDTH,
                "height": PIXEL_HEIGHT,
                "data_hex": self.pixels.hex(),
            },
            "hashes": {
                "state_sha256": self.state_sha256,
                "pixel_sha256": self.pixel_sha256,
            },
        }


@dataclass(frozen=True, slots=True)
class OracleTrace:
    provenance: dict[str, object]
    scenario: dict[str, object]
    frames: tuple[OracleFrame, ...]
    result: dict[str, int | float | bool]

    def to_json(self) -> dict[str, object]:
        frame_values = [frame.to_json() for frame in self.frames]
        initial = self.frames[0]
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_type": TRACE_TYPE,
            "provenance": self.provenance,
            "scenario": self.scenario,
            "initial_state": {
                "frame": initial.frame_index,
                "state": initial.state.to_json(),
                "pixel_sha256": initial.pixel_sha256,
            },
            "frames": frame_values,
            "result": self.result,
            "hashes": {
                "frames_sha256": hashlib.sha256(
                    canonical_json(frame_values).encode("utf-8")
                ).hexdigest()
            },
        }

    def canonical_bytes(self) -> bytes:
        return (canonical_json(self.to_json()) + "\n").encode("utf-8")


def parse_full_draw_stdout(
    stdout: str,
) -> tuple[tuple[OracleFrame, ...], dict[str, int | float | bool]]:
    states: dict[int, RawState] = {}
    pixel_rows: dict[int, dict[int, tuple[int, ...]]] = {}
    metadata: dict[int, tuple[int, int, int, bool, float, tuple[str, ...]]] = {}
    result_lines = 0

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith(STATE_PREFIX):
            state = parse_raw_state(line, prefix=STATE_PREFIX)
            if state.frame in states:
                raise ControlRuntimeError(
                    f"duplicate full-draw state for frame {state.frame}"
                )
            states[state.frame] = state
        elif line.startswith(FRAME_PREFIX):
            frame, mask, previous_mask, mode, dead, reward, events = parse_frame_line(
                line
            )
            if frame in metadata:
                raise ControlRuntimeError(
                    f"duplicate full-draw metadata for frame {frame}"
                )
            metadata[frame] = (mask, previous_mask, mode, dead, reward, events)
        elif line.startswith(PIXEL_PREFIX):
            frame, row, pixels = parse_pixel_line(line)
            frame_rows = pixel_rows.setdefault(frame, {})
            if row in frame_rows:
                raise ControlRuntimeError(
                    f"duplicate full-draw pixels for frame {frame}, row {row}"
                )
            frame_rows[row] = pixels
        elif line.startswith(RESULT_PREFIX):
            result_lines += 1

    result = _parse_result(stdout)
    if result_lines != 1:
        raise ControlRuntimeError("full-draw Pemsa produced an invalid result count")
    if not states:
        raise ControlRuntimeError("full-draw Pemsa did not produce state frames")
    if set(states) != set(pixel_rows):
        raise ControlRuntimeError("full-draw state/pixel frame sets differ")
    if set(states) != set(metadata):
        raise ControlRuntimeError("full-draw state/metadata frame sets differ")

    terminal_frame = result["frames"]
    frames: list[OracleFrame] = []
    previous_frame = -1
    for frame_index in sorted(states):
        if frame_index <= previous_frame:
            raise ControlRuntimeError("full-draw frame indexes are not monotonic")
        rows = pixel_rows[frame_index]
        if set(rows) != set(range(PIXEL_HEIGHT)):
            raise ControlRuntimeError(
                f"full-draw frame {frame_index} does not contain 128 pixel rows"
            )
        pixels = b"".join(bytes(rows[row]) for row in range(PIXEL_HEIGHT))
        if len(pixels) != PIXEL_COUNT:
            raise ControlRuntimeError(
                f"full-draw frame {frame_index} has an invalid pixel count"
            )
        done = frame_index == terminal_frame
        frames.append(
            OracleFrame(
                frame_index=frame_index,
                state=states[frame_index],
                pixels=pixels,
                done=done,
                reward=metadata[frame_index][4],
                events=(
                    (*metadata[frame_index][5], "terminal")
                    if done
                    else metadata[frame_index][5]
                ),
                input_mask=metadata[frame_index][0],
                previous_input_mask=metadata[frame_index][1],
                mode=metadata[frame_index][2],
                dead=metadata[frame_index][3],
            )
        )
        previous_frame = frame_index

    if not frames[-1].done:
        raise ControlRuntimeError("full-draw trace is missing its terminal frame")
    if sum(frame.done for frame in frames) != 1:
        raise ControlRuntimeError("full-draw trace has an invalid terminal frame count")
    return tuple(frames), result


def parse_pixel_line(line: str) -> tuple[int, int, tuple[int, ...]]:
    payload = line.removeprefix(PIXEL_PREFIX)
    values = payload.split("|", 2)
    if len(values) != 3:
        raise ControlRuntimeError("invalid full-draw pixel field count")
    try:
        frame = int(values[0])
        row = int(values[1])
        palette_values = tuple(int(value) for value in values[2].split(","))
    except ValueError as error:
        raise ControlRuntimeError("invalid full-draw pixel values") from error
    if frame < 0 or not 0 <= row < PIXEL_HEIGHT:
        raise ControlRuntimeError("invalid full-draw pixel coordinates")
    if len(palette_values) != PIXEL_WIDTH:
        raise ControlRuntimeError("full-draw pixel row must contain 128 values")
    if any(value < 0 or value > 15 for value in palette_values):
        raise ControlRuntimeError(
            "full-draw pixels must be palette indexes from 0 to 15"
        )
    return frame, row, palette_values


def parse_frame_line(
    line: str,
) -> tuple[int, int, int, int, bool, float, tuple[str, ...]]:
    values = line.removeprefix(FRAME_PREFIX).split("|")
    if len(values) not in {6, 7}:
        raise ControlRuntimeError("invalid full-draw metadata field count")
    try:
        frame, mask, previous_mask, mode, dead = (int(value) for value in values[:5])
        if len(values) == 7:
            reward = float(values[5])
            events_value = values[6]
        else:
            reward = 0.0
            events_value = values[5]
    except ValueError as error:
        raise ControlRuntimeError("invalid full-draw metadata values") from error
    if frame < 0 or not 0 <= mask <= 63 or not 0 <= previous_mask <= 63:
        raise ControlRuntimeError("invalid full-draw input mask")
    if not 0 <= mode <= 4 or dead not in {0, 1}:
        raise ControlRuntimeError("invalid full-draw lifecycle metadata")
    if not math.isfinite(reward) or reward < 0:
        raise ControlRuntimeError("invalid full-draw reward")
    events = tuple(event for event in events_value.split(",") if event)
    if any(not event.replace("_", "").isalnum() for event in events):
        raise ControlRuntimeError("invalid full-draw event name")
    return frame, mask, previous_mask, mode, bool(dead), reward, events


def command_schedule(commands: list[MovementCommand]) -> list[dict[str, object]]:
    from dodge.headless import duration_to_frames

    return [
        {
            "index": index,
            "move": command.move,
            "duration_ms": command.duration_ms,
            "duration_frames": duration_to_frames(command.duration_ms),
        }
        for index, command in enumerate(commands)
    ]
