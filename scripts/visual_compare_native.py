from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from dodge.native.differential import decode_native_snapshot

ROOT = Path(__file__).resolve().parents[1]
FRAME_WIDTH = 128
FRAME_HEIGHT = 128
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT
ORACLE_MODES = {
    1: "menu",
    2: "game",
    3: "settings",
    4: "transition_to_game",
}
PALETTE = (
    (0, 0, 0),
    (29, 43, 83),
    (126, 37, 83),
    (0, 135, 81),
    (171, 82, 54),
    (95, 87, 79),
    (194, 195, 199),
    (255, 241, 232),
    (255, 0, 77),
    (255, 163, 0),
    (255, 236, 39),
    (0, 228, 54),
    (41, 173, 255),
    (131, 118, 156),
    (255, 119, 168),
    (255, 204, 170),
)


@dataclass(frozen=True, slots=True)
class Selection:
    name: str
    oracle: str
    native: str
    frames: tuple[tuple[int, str], ...]


SELECTIONS = (
    Selection(
        name="menu",
        oracle="src/dodge/runtime/.native-p7-menu-oracle-20260902.json",
        native="src/dodge/runtime/.native-p7-menu-native-20260902.json",
        frames=((1, "menu"),),
    ),
    Selection(
        name="gameplay",
        oracle="src/dodge/runtime/.native-p4-current-oracle-20260902.json",
        native="src/dodge/runtime/.native-p4-full-native-f64-20260902.json",
        frames=(
            (1, "transition_to_game"),
            (7, "transition_draw"),
            (13, "game_ready"),
            (57, "enemy_spawn"),
            (128, "gameplay"),
            (249, "pattern_or_spawn"),
            (369, "pattern"),
            (426, "particles"),
            (448, "particles"),
            (501, "death"),
        ),
    ),
    Selection(
        name="settings",
        oracle="src/dodge/runtime/.native-p4-settings-oracle3-20260902.json",
        native="src/dodge/runtime/.native-p4-settings-native-f64-20260902.json",
        frames=(
            (44, "transition_to_settings"),
            (56, "settings"),
            (67, "settings_cursor"),
            (76, "settings"),
            (77, "transition_to_game"),
            (88, "transition_to_game"),
            (89, "game_after_settings"),
            (140, "game_after_settings"),
            (191, "game_after_settings"),
            (233, "death"),
        ),
    ),
)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)
    )


def _write_png(path: Path, width: int, height: int, rgb: bytes) -> None:
    row_size = width * 3
    if len(rgb) != row_size * height:
        raise ValueError("RGB payload has the wrong dimensions")
    scanlines = bytearray()
    for row in range(height):
        scanlines.append(0)
        start = row * row_size
        scanlines.extend(rgb[start : start + row_size])
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
    )
    payload += _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _palette_rgb(pixels: bytes) -> bytes:
    if len(pixels) != FRAME_SIZE:
        raise ValueError(f"expected {FRAME_SIZE} pixels, got {len(pixels)}")
    rgb = bytearray()
    for pixel in pixels:
        if pixel >= len(PALETTE):
            raise ValueError(f"invalid palette index {pixel}")
        rgb.extend(PALETTE[pixel])
    return bytes(rgb)


def _diff_rgb(left: bytes, right: bytes) -> bytes:
    if len(left) != FRAME_SIZE or len(right) != FRAME_SIZE:
        raise ValueError("diff inputs have the wrong dimensions")
    match = (29, 43, 83)
    mismatch = (255, 0, 77)
    rgb = bytearray()
    for expected, actual in zip(left, right, strict=True):
        rgb.extend(match if expected == actual else mismatch)
    return bytes(rgb)


def _scale_rgb(rgb: bytes, width: int, height: int, scale: int) -> bytes:
    row_size = width * 3
    scaled = bytearray()
    for row in range(height):
        start = row * row_size
        source_row = rgb[start : start + row_size]
        expanded_row = b"".join(
            source_row[offset : offset + 3] * scale for offset in range(0, row_size, 3)
        )
        for _ in range(scale):
            scaled.extend(expanded_row)
    return bytes(scaled)


def _triptych(
    oracle_rgb: bytes,
    native_rgb: bytes,
    diff_rgb: bytes,
    *,
    scale: int,
) -> tuple[int, int, bytes]:
    gap = bytes((16, 18, 32))
    columns = (oracle_rgb, native_rgb, diff_rgb)
    base_width = FRAME_WIDTH * len(columns) + 2
    base = bytearray()
    row_size = FRAME_WIDTH * 3
    for row in range(FRAME_HEIGHT):
        row_start = row * row_size
        for index, column in enumerate(columns):
            base.extend(column[row_start : row_start + row_size])
            if index != len(columns) - 1:
                base.extend(gap)
    return (
        base_width * scale,
        FRAME_HEIGHT * scale,
        _scale_rgb(bytes(base), base_width, FRAME_HEIGHT, scale),
    )


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"trace root is not an object: {path}")
    return value


def _frame_map(trace: dict[str, object]) -> dict[int, dict[str, object]]:
    frames = trace.get("frames")
    if not isinstance(frames, list):
        raise ValueError("trace frames are not a list")
    result: dict[int, dict[str, object]] = {}
    for value in frames:
        if not isinstance(value, dict) or not isinstance(value.get("frame"), int):
            raise ValueError("trace contains an invalid frame")
        result[value["frame"]] = value
    return result


def _frame_pixels(
    oracle_frame: dict[str, object], native_frame: dict[str, object]
) -> tuple[bytes, bytes]:
    oracle_pixels = oracle_frame.get("pixels")
    if not isinstance(oracle_pixels, dict):
        raise ValueError("oracle frame has no pixel object")
    data_hex = oracle_pixels.get("data_hex")
    if not isinstance(data_hex, str):
        raise ValueError("oracle frame has no pixel data")
    oracle = bytes.fromhex(data_hex)
    snapshot_hex = native_frame.get("snapshot_hex")
    if not isinstance(snapshot_hex, str):
        raise ValueError("native frame has no snapshot")
    native = decode_native_snapshot(snapshot_hex).pixels
    if len(oracle) != FRAME_SIZE or len(native) != FRAME_SIZE:
        raise ValueError("trace frame does not contain a 128x128 framebuffer")
    return oracle, native


def _frame_report(
    selection: Selection,
    label: str,
    frame_number: int,
    oracle_frame: dict[str, object],
    native_frame: dict[str, object],
    oracle_pixels: bytes,
    native_pixels: bytes,
    image_path: Path,
) -> dict[str, object]:
    native_snapshot_hex = native_frame.get("snapshot_hex")
    if not isinstance(native_snapshot_hex, str):
        raise ValueError("native frame has no snapshot")
    native_snapshot = decode_native_snapshot(native_snapshot_hex)
    differing = [
        index
        for index, (expected, actual) in enumerate(
            zip(oracle_pixels, native_pixels, strict=True)
        )
        if expected != actual
    ]
    oracle_input = oracle_frame.get("input")
    oracle_state = oracle_frame.get("state")
    if not isinstance(oracle_input, dict) or not isinstance(oracle_state, dict):
        raise ValueError("oracle frame is missing state metadata")
    oracle_mode = ORACLE_MODES.get(oracle_input.get("mode"), "unknown")
    return {
        "selection": selection.name,
        "label": label,
        "frame": frame_number,
        "oracle_mode": oracle_mode,
        "native_mode": native_snapshot.mode,
        "oracle_state_counts": {
            "enemies": len(oracle_state.get("enemies", [])),
            "particles": len(oracle_state.get("particles", [])),
            "aoes": len(oracle_state.get("aoes", [])),
        },
        "native_state_counts": {
            "enemies": len(native_snapshot.enemies),
            "particles": len(native_snapshot.particles),
            "patterns": len(native_snapshot.patterns),
            "active_pattern": native_snapshot.active_pattern is not None,
        },
        "oracle_events": oracle_frame.get("events", []),
        "native_events": native_frame.get("events", []),
        "pixels_match": not differing,
        "different_pixel_count": len(differing),
        "first_difference": (
            None
            if not differing
            else {
                "x": differing[0] % FRAME_WIDTH,
                "y": differing[0] // FRAME_WIDTH,
            }
        ),
        "oracle_pixel_sha256": hashlib.sha256(oracle_pixels).hexdigest(),
        "native_pixel_sha256": hashlib.sha256(native_pixels).hexdigest(),
        "image": str(image_path),
    }


def compare_selection(
    selection: Selection, output_dir: Path, scale: int
) -> list[dict[str, object]]:
    oracle_trace = _load_json(ROOT / selection.oracle)
    native_trace = _load_json(ROOT / selection.native)
    oracle_frames = _frame_map(oracle_trace)
    native_frames = _frame_map(native_trace)
    reports = []
    for frame_number, label in selection.frames:
        oracle_frame = oracle_frames.get(frame_number)
        native_frame = native_frames.get(frame_number)
        if oracle_frame is None or native_frame is None:
            raise ValueError(f"frame {frame_number} missing from {selection.name}")
        oracle_pixels, native_pixels = _frame_pixels(oracle_frame, native_frame)
        oracle_rgb = _palette_rgb(oracle_pixels)
        native_rgb = _palette_rgb(native_pixels)
        diff_rgb = _diff_rgb(oracle_pixels, native_pixels)
        width, height, triptych = _triptych(
            oracle_rgb,
            native_rgb,
            diff_rgb,
            scale=scale,
        )
        image_path = output_dir / f"{selection.name}-frame-{frame_number:04d}.png"
        _write_png(image_path, width, height, triptych)
        reports.append(
            _frame_report(
                selection,
                label,
                frame_number,
                oracle_frame,
                native_frame,
                oracle_pixels,
                native_pixels,
                image_path,
            )
        )
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create exact original/native Dodge visual comparison strips."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "src/dodge/runtime/.native-visual-review-20260902",
    )
    parser.add_argument("--scale", type=int, default=4)
    arguments = parser.parse_args(argv)
    if arguments.scale < 1:
        parser.error("--scale must be at least one")

    try:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        frames = [
            report
            for selection in SELECTIONS
            for report in compare_selection(
                selection, arguments.output_dir, arguments.scale
            )
        ]
        mismatches = [report for report in frames if not report["pixels_match"]]
        manifest = {
            "phase": "P7",
            "status": "pixel_match" if not mismatches else "mismatch",
            "comparison": "oracle indexed pixels vs native canonical snapshot pixels",
            "columns": ["original_oracle", "native", "diff_match_blue_mismatch_red"],
            "frame_count": len(frames),
            "scale": arguments.scale,
            "frames": frames,
            "claims_not_made": [
                "owner visual approval",
                "full-game mathematical equivalence",
                "Macroquad window scaling equivalence beyond the lossless capture",
            ],
        }
        manifest_path = arguments.output_dir / "visual-review-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"visual comparison failed: {error}\n")

    print(
        json.dumps(
            {
                "output_dir": str(arguments.output_dir),
                "manifest": str(manifest_path),
                "status": manifest["status"],
                "frames": len(frames),
                "mismatches": len(mismatches),
            },
            separators=(",", ":"),
        )
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
