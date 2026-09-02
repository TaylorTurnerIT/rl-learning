from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dodge.control import CARTRIDGE_PATH
from dodge.native.manifest import manifest_for_path

FRAME_WIDTH = 128
FRAME_HEIGHT = 128
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT
FIXED_SCALE = 1 << 16
SNAPSHOT_MAGIC = b"DGSN"
SNAPSHOT_WIRE_VERSION = 1
MAX_ENEMIES = 4_096

NATIVE_MODES = {
    0: "menu",
    1: "transition_to_game",
    2: "game",
    3: "terminal",
}
ORACLE_MODES = {
    1: "menu",
    2: "game",
    3: "settings",
    4: "transition_to_game",
}


class NativeDifferentialError(ValueError):
    """A native trace cannot be decoded or compared."""


@dataclass(frozen=True, slots=True)
class NativeEnemy:
    x: int
    y: int
    vx: int
    vy: int
    size: int
    max_size: int
    personality: int
    speed: int
    inside: bool
    is_dying: bool


@dataclass(frozen=True, slots=True)
class NativeSnapshot:
    source_sha256: str
    seed: int
    frame: int
    mode: str
    transition_y: int
    started: bool
    game_ready: bool
    dead: bool
    input_mask: int
    previous_input_mask: int
    player: tuple[int, int, int, int, int]
    enemies: tuple[NativeEnemy, ...]
    score: int
    survival_frames: int
    pixels: bytes


@dataclass(frozen=True, slots=True)
class DifferentialMismatch:
    frame: int
    path: str
    expected: object
    actual: object
    source: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "path": self.path,
            "expected": self.expected,
            "actual": self.actual,
            "source": self.source,
        }


def decode_native_snapshot(snapshot_hex: str) -> NativeSnapshot:
    if not isinstance(snapshot_hex, str):
        raise NativeDifferentialError("native snapshot_hex must be a string")
    try:
        data = bytes.fromhex(snapshot_hex)
    except ValueError as error:
        raise NativeDifferentialError(
            "native snapshot_hex is not hexadecimal"
        ) from error
    reader = _Reader(data)
    if reader.take(4) != SNAPSHOT_MAGIC:
        raise NativeDifferentialError("native snapshot magic mismatch")
    if reader.u32() != SNAPSHOT_WIRE_VERSION:
        raise NativeDifferentialError("native snapshot wire version mismatch")
    core_schema = reader.u32()
    if core_schema < 1:
        raise NativeDifferentialError("native snapshot core schema is invalid")
    source_sha256 = reader.take(32).hex()
    seed = reader.u32()
    frame = reader.u32()
    mode_tag = reader.u8()
    if mode_tag not in NATIVE_MODES:
        raise NativeDifferentialError("native snapshot mode is invalid")
    transition_y = reader.i16()
    started = reader.boolean()
    game_ready = reader.boolean()
    dead = reader.boolean()
    input_mask = reader.u8()
    previous_input_mask = reader.u8()
    if input_mask > 63 or previous_input_mask > 63:
        raise NativeDifferentialError("native snapshot input mask is invalid")

    reader.u32()
    reader.skip(31 * 4)
    front = reader.u8()
    rear = reader.u8()
    if front >= 31 or rear >= 31:
        raise NativeDifferentialError("native snapshot RNG checkpoint is invalid")

    player = tuple(reader.i32() for _ in range(5))
    enemy_count = reader.u32()
    if enemy_count > MAX_ENEMIES:
        raise NativeDifferentialError("native snapshot enemy count is invalid")
    enemies = tuple(_read_enemy(reader) for _ in range(enemy_count))

    reader.i32()
    reader.i32()
    reader.u32()
    reader.i32()
    reader.i32()
    reader.i32()
    reader.u32()
    reader.boolean()
    score = reader.i32()
    survival_frames = reader.u32()
    reader.i16()

    draw_color = reader.u8()
    reader.u16()
    draw_palette = reader.take(16)
    screen_palette = reader.take(16)
    transparent = reader.take(16)
    if draw_color >= 16 or any(color >= 16 for color in draw_palette + screen_palette):
        raise NativeDifferentialError("native snapshot palette is invalid")
    if any(value not in {0, 1} for value in transparent):
        raise NativeDifferentialError("native snapshot transparency is invalid")
    reader.i32()
    reader.i32()
    reader.i16()
    reader.i16()
    clip_width = reader.u16()
    clip_height = reader.u16()
    if clip_width > FRAME_WIDTH or clip_height > FRAME_HEIGHT:
        raise NativeDifferentialError("native snapshot clip is invalid")
    reader.i16()

    pixels = reader.take(FRAME_SIZE)
    if any(pixel >= 16 for pixel in pixels):
        raise NativeDifferentialError("native snapshot pixels are not palette indexes")
    reader.ensure_finished()
    return NativeSnapshot(
        source_sha256=source_sha256,
        seed=seed,
        frame=frame,
        mode=NATIVE_MODES[mode_tag],
        transition_y=transition_y,
        started=started,
        game_ready=game_ready,
        dead=dead,
        input_mask=input_mask,
        previous_input_mask=previous_input_mask,
        player=player,  # type: ignore[arg-type]
        enemies=enemies,
        score=score,
        survival_frames=survival_frames,
        pixels=pixels,
    )


def compare_native_to_oracle(
    native: Mapping[str, object],
    oracle: Mapping[str, object],
    *,
    source_map: Mapping[str, object] | None = None,
    compare_pixels: bool = True,
) -> dict[str, object]:
    native_frames = _list_value(native, "frames")
    oracle_frames = _list_value(oracle, "frames")
    for index, expected_frame_value in enumerate(oracle_frames):
        frame = _mapping_value(expected_frame_value, f"oracle frame {index}")
        if index >= len(native_frames):
            frame_number = _int_value(frame, "frame", index)
            mismatch = _mismatch(
                frame_number,
                "frames.present",
                frame_number,
                None,
                "lifecycle",
                source_map,
            )
            return _report(native, oracle, index, mismatch)
        actual_frame = _mapping_value(native_frames[index], f"native frame {index}")
        mismatch = _compare_frame(
            actual_frame,
            frame,
            source_map=source_map,
            compare_pixels=compare_pixels,
        )
        if mismatch is not None:
            return _report(native, oracle, index + 1, mismatch)

    if len(native_frames) > len(oracle_frames):
        actual_frame = _mapping_value(
            native_frames[len(oracle_frames)], "extra native frame"
        )
        frame_number = _int_value(actual_frame, "frame", len(oracle_frames) + 1)
        mismatch = _mismatch(
            frame_number,
            "frames.present",
            None,
            frame_number,
            "lifecycle",
            source_map,
        )
        return _report(native, oracle, len(oracle_frames), mismatch)

    mismatch = _compare_results(
        native,
        oracle,
        native_frames,
        source_map=source_map,
    )
    if mismatch is not None:
        return _report(native, oracle, len(native_frames), mismatch)
    return _report(native, oracle, len(native_frames), None)


def load_source_map(
    path: Path | None = None,
    *,
    source: Path = CARTRIDGE_PATH,
) -> dict[str, object]:
    if path is not None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise NativeDifferentialError(
                f"could not read source map: {error}"
            ) from error
        if not isinstance(value, dict):
            raise NativeDifferentialError("source map must be a JSON object")
        return value

    generated = Path("src/dodge/runtime/.native-assets-check-final/source_map.json")
    if generated.exists():
        try:
            value = json.loads(generated.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = None
        if isinstance(value, dict):
            return value

    manifest = manifest_for_path(source)
    functions = list(manifest.functions)
    mapped: list[dict[str, object]] = []
    for index, function in enumerate(functions):
        next_line = (
            functions[index + 1].line
            if index + 1 < len(functions)
            else max(section.end_line for section in manifest.sections) + 1
        )
        mapped.append(
            {
                "pico8_name": function.name,
                "source": {"section": "lua", "span": [function.line, next_line - 1]},
                "rust_target": f"dodge_core::{function.name}",
            }
        )
    return {"functions": mapped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-native-diff",
        description="Compare a native runner trace with a canonical Pemsa trace.",
    )
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--source-map", type=Path)
    parser.add_argument("--source", type=Path, default=CARTRIDGE_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-pixels",
        action="store_true",
        help="compare logical slice fields only",
    )
    arguments = parser.parse_args(argv)
    try:
        native = _load_json(arguments.native, "native trace")
        oracle = _load_json(arguments.oracle, "oracle trace")
        source_map = load_source_map(arguments.source_map, source=arguments.source)
        report = compare_native_to_oracle(
            native,
            oracle,
            source_map=source_map,
            compare_pixels=not arguments.no_pixels,
        )
        payload = (
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_bytes(payload)
        else:
            sys.stdout.buffer.write(payload)
    except (NativeDifferentialError, OSError, ValueError) as error:
        print(f"dodge-native-diff: {error}", file=sys.stderr)
        return 1
    return 0 if report["status"] == "match" else 2


def _compare_frame(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    source_map: Mapping[str, object] | None,
    compare_pixels: bool,
) -> DifferentialMismatch | None:
    frame_number = _int_value(expected, "frame", 0)
    actual_frame = _int_value(actual, "frame", frame_number)
    if actual_frame != frame_number:
        return _mismatch(
            frame_number,
            "frame",
            frame_number,
            actual_frame,
            "lifecycle",
            source_map,
        )

    expected_input = _mapping_value(expected.get("input"), "oracle input")
    comparisons = (
        (
            "input.mask",
            _int_value(expected_input, "mask", frame_number),
            _int_value(actual, "input_mask", frame_number),
            "input",
        ),
        (
            "input.previous_mask",
            _int_value(expected_input, "previous_mask", frame_number),
            _int_value(actual, "previous_input_mask", frame_number),
            "input",
        ),
        (
            "lifecycle.mode",
            ORACLE_MODES.get(
                _int_value(expected_input, "mode", frame_number), "invalid"
            ),
            actual.get("mode"),
            "lifecycle",
        ),
        (
            "lifecycle.dead",
            bool(expected_input.get("dead")),
            bool(actual.get("dead")),
            "lifecycle",
        ),
        (
            "terminal.done",
            bool(expected.get("done")),
            bool(actual.get("done")),
            "lifecycle",
        ),
    )
    for path, expected_value, actual_value, source_name in comparisons:
        if expected_value != actual_value:
            return _mismatch(
                frame_number,
                path,
                expected_value,
                actual_value,
                source_name,
                source_map,
            )

    expected_reward = expected.get("reward", 0.0)
    actual_reward_raw = actual.get("reward_raw", 0)
    if not isinstance(expected_reward, int | float) or isinstance(
        expected_reward, bool
    ):
        raise NativeDifferentialError(f"oracle frame {frame_number} reward is invalid")
    if not isinstance(actual_reward_raw, int) or isinstance(actual_reward_raw, bool):
        raise NativeDifferentialError(
            f"native frame {frame_number} reward_raw is invalid"
        )
    actual_reward = actual_reward_raw / FIXED_SCALE
    if not math.isclose(float(expected_reward), actual_reward, abs_tol=0.0002):
        return _mismatch(
            frame_number,
            "reward",
            expected_reward,
            actual_reward,
            "updategame",
            source_map,
        )

    expected_events = expected.get("events", [])
    actual_events = actual.get("events", [])
    if not isinstance(expected_events, list) or not all(
        isinstance(event, str) for event in expected_events
    ):
        raise NativeDifferentialError(f"oracle frame {frame_number} events are invalid")
    if not isinstance(actual_events, list) or not all(
        isinstance(event, str) for event in actual_events
    ):
        raise NativeDifferentialError(f"native frame {frame_number} events are invalid")
    if expected_events != actual_events:
        return _mismatch(
            frame_number,
            "events",
            expected_events,
            actual_events,
            "updategame",
            source_map,
        )

    snapshot = decode_native_snapshot(
        _string_value(actual, "snapshot_hex", frame_number)
    )
    if snapshot.frame != frame_number:
        return _mismatch(
            frame_number,
            "snapshot.lifecycle.frame",
            frame_number,
            snapshot.frame,
            "lifecycle",
            source_map,
        )
    if snapshot.mode != actual.get("mode"):
        return _mismatch(
            frame_number,
            "snapshot.lifecycle.mode",
            actual.get("mode"),
            snapshot.mode,
            "lifecycle",
            source_map,
        )
    state = _mapping_value(expected.get("state"), "oracle state")
    expected_player = _mapping_value(state.get("player"), "oracle player")
    for index, name in enumerate(("x", "y", "vx", "vy", "size")):
        expected_value = _number_value(expected_player, name, frame_number)
        actual_value = _fixed_float(snapshot.player[index])
        if not math.isclose(expected_value, actual_value, abs_tol=0.0002):
            return _mismatch(
                frame_number,
                f"player.{name}",
                expected_value,
                actual_value,
                "player",
                source_map,
            )

    expected_enemies = _list_value(state, "enemies")
    if len(expected_enemies) != len(snapshot.enemies):
        source_name = "updatefyou" if len(expected_enemies) > 0 else "updateenemies"
        return _mismatch(
            frame_number,
            "enemies.count",
            len(expected_enemies),
            len(snapshot.enemies),
            source_name,
            source_map,
        )
    for index, expected_enemy_value in enumerate(expected_enemies):
        expected_enemy = _mapping_value(
            expected_enemy_value,
            f"oracle enemy {index}",
        )
        actual_enemy = snapshot.enemies[index]
        fields = (
            ("x", expected_enemy.get("x"), actual_enemy.x),
            ("y", expected_enemy.get("y"), actual_enemy.y),
            ("vx", expected_enemy.get("vx"), actual_enemy.vx),
            ("vy", expected_enemy.get("vy"), actual_enemy.vy),
            (
                "width",
                expected_enemy.get("width"),
                8 * FIXED_SCALE if actual_enemy.personality >= 2 else actual_enemy.size,
            ),
            (
                "height",
                expected_enemy.get("height"),
                8 * FIXED_SCALE if actual_enemy.personality >= 2 else actual_enemy.size,
            ),
        )
        for name, expected_value, actual_raw in fields:
            if not isinstance(expected_value, int | float):
                raise NativeDifferentialError(
                    f"oracle enemy {index} field {name} is not numeric"
                )
            actual_value = _fixed_float(actual_raw)
            if not math.isclose(float(expected_value), actual_value, abs_tol=0.0002):
                return _mismatch(
                    frame_number,
                    f"enemies[{index}].{name}",
                    expected_value,
                    actual_value,
                    "updateenemies",
                    source_map,
                )

    if compare_pixels:
        expected_pixels = _oracle_pixels(expected, frame_number)
        actual_pixels = snapshot.pixels
        if len(expected_pixels) != len(actual_pixels):
            raise NativeDifferentialError(
                "native and oracle framebuffer lengths differ"
            )
        for pixel_index, (expected_value, actual_value) in enumerate(
            zip(expected_pixels, actual_pixels, strict=True)
        ):
            if expected_value != actual_value:
                y, x = divmod(pixel_index, FRAME_WIDTH)
                mode = _int_value(expected_input, "mode", frame_number)
                source_name = "drawtransition" if mode == 4 else "drawgame"
                return _mismatch(
                    frame_number,
                    f"pixels[{y},{x}]",
                    expected_value,
                    actual_value,
                    source_name,
                    source_map,
                )
    return None


def _compare_results(
    native: Mapping[str, object],
    oracle: Mapping[str, object],
    native_frames: Sequence[object],
    *,
    source_map: Mapping[str, object] | None,
) -> DifferentialMismatch | None:
    if not native_frames:
        raise NativeDifferentialError("trace contains no frames")
    native_result = _mapping_value(native.get("result"), "native result")
    oracle_result = _mapping_value(oracle.get("result"), "oracle result")
    frame = _int_value(native_frames[-1], "frame", len(native_frames))
    for name in ("frames",):
        if name in oracle_result and name in native_result:
            expected = oracle_result[name]
            actual = native_result[name]
            if expected != actual:
                return _mismatch(
                    frame,
                    f"result.{name}",
                    expected,
                    actual,
                    "lifecycle",
                    source_map,
                )
    last_snapshot = decode_native_snapshot(
        _string_value(native_frames[-1], "snapshot_hex", frame)
    )
    if "score" in oracle_result:
        expected_score = oracle_result["score"]
        actual_score = _fixed_float(last_snapshot.score)
        if isinstance(expected_score, int | float) and not math.isclose(
            float(expected_score), actual_score, abs_tol=0.0002
        ):
            return _mismatch(
                frame,
                "result.score",
                expected_score,
                actual_score,
                "collide",
                source_map,
            )
    if "survival_frames" in oracle_result:
        expected_survival = oracle_result["survival_frames"]
        if expected_survival != last_snapshot.survival_frames:
            return _mismatch(
                frame,
                "result.survival_frames",
                expected_survival,
                last_snapshot.survival_frames,
                "updategame",
                source_map,
            )
    return None


def _report(
    native: Mapping[str, object],
    oracle: Mapping[str, object],
    frames_compared: int,
    mismatch: DifferentialMismatch | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "match" if mismatch is None else "mismatch",
        "frames_compared": frames_compared,
        "first_mismatch": None if mismatch is None else mismatch.to_json(),
        "native_result": native.get("result"),
        "oracle_result": oracle.get("result"),
    }


def _mismatch(
    frame: int,
    path: str,
    expected: object,
    actual: object,
    source_name: str,
    source_map: Mapping[str, object] | None,
) -> DifferentialMismatch:
    return DifferentialMismatch(
        frame=frame,
        path=path,
        expected=expected,
        actual=actual,
        source=_source_for(source_name, source_map),
    )


def _source_for(
    function_name: str,
    source_map: Mapping[str, object] | None,
) -> dict[str, object]:
    function_name = {
        "input": "_update60",
        "lifecycle": "_update60",
        "player": "updategame",
        "updatefyou": "updatefyou",
        "updateenemies": "updateenemies",
        "collide": "collide",
    }.get(function_name, function_name)
    if source_map is not None:
        functions = source_map.get("functions")
        if isinstance(functions, list):
            for function in functions:
                if not isinstance(function, dict):
                    continue
                if function.get("pico8_name") != function_name:
                    continue
                source = function.get("source")
                if isinstance(source, dict):
                    result: dict[str, object] = {
                        "pico8_name": function_name,
                        "section": source.get("section"),
                        "span": source.get("span"),
                    }
                    if "rust_target" in function:
                        result["rust_target"] = function["rust_target"]
                    return result
    return {
        "pico8_name": function_name,
        "section": "lua",
        "span": [1, 2_257],
    }


def _oracle_pixels(frame: Mapping[str, object], frame_number: int) -> bytes:
    pixels = _mapping_value(frame.get("pixels"), f"oracle pixels {frame_number}")
    if pixels.get("encoding") != "palette_index_u8_row_major":
        raise NativeDifferentialError("oracle framebuffer encoding is not canonical")
    if pixels.get("width") != FRAME_WIDTH or pixels.get("height") != FRAME_HEIGHT:
        raise NativeDifferentialError("oracle framebuffer dimensions are invalid")
    data_hex = pixels.get("data_hex")
    if not isinstance(data_hex, str):
        raise NativeDifferentialError("oracle framebuffer data_hex is invalid")
    try:
        data = bytes.fromhex(data_hex)
    except ValueError as error:
        raise NativeDifferentialError(
            "oracle framebuffer is not hexadecimal"
        ) from error
    if len(data) != FRAME_SIZE:
        raise NativeDifferentialError("oracle framebuffer length is invalid")
    return data


def _read_enemy(reader: _Reader) -> NativeEnemy:
    return NativeEnemy(
        x=reader.i32(),
        y=reader.i32(),
        vx=reader.i32(),
        vy=reader.i32(),
        size=reader.i32(),
        max_size=reader.i32(),
        personality=reader.i8(),
        speed=reader.i32(),
        inside=reader.boolean(),
        is_dying=reader.boolean(),
    )


def _fixed_float(raw: int) -> float:
    return raw / FIXED_SCALE


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NativeDifferentialError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise NativeDifferentialError(f"{label} must be a JSON object")
    return value


def _mapping_value(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NativeDifferentialError(f"{label} must be a JSON object")
    return value


def _list_value(value: Mapping[str, object], name: str) -> list[object]:
    result = value.get(name)
    if not isinstance(result, list):
        raise NativeDifferentialError(f"{name} must be a JSON list")
    return result


def _int_value(value: Mapping[str, object], name: str, frame: int) -> int:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, int):
        raise NativeDifferentialError(f"frame {frame} {name} must be an integer")
    return result


def _number_value(value: Mapping[str, object], name: str, frame: int) -> float:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise NativeDifferentialError(f"frame {frame} {name} must be numeric")
    return float(result)


def _string_value(value: Mapping[str, object], name: str, frame: int) -> str:
    result = value.get(name)
    if not isinstance(result, str):
        raise NativeDifferentialError(f"frame {frame} {name} must be a string")
    return result


class _Reader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def take(self, length: int) -> bytes:
        end = self._offset + length
        if length < 0 or end > len(self._data):
            raise NativeDifferentialError("native snapshot is truncated")
        value = self._data[self._offset : end]
        self._offset = end
        return value

    def skip(self, length: int) -> None:
        self.take(length)

    def u8(self) -> int:
        return self.take(1)[0]

    def i8(self) -> int:
        value = self.u8()
        return value - 256 if value >= 128 else value

    def boolean(self) -> bool:
        value = self.u8()
        if value not in {0, 1}:
            raise NativeDifferentialError("native snapshot boolean is invalid")
        return bool(value)

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def ensure_finished(self) -> None:
        if self._offset != len(self._data):
            raise NativeDifferentialError("native snapshot has trailing bytes")
