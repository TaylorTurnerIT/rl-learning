from __future__ import annotations

import struct

import pytest

from dodge.native.differential import (
    NativeDifferentialError,
    compare_native_to_oracle,
    decode_native_snapshot,
)

SOURCE_HASH = bytes.fromhex(
    "7453a9658fd32577385ad72672a54ad84ff70567fadbde75ba6634aa5cc684a3"
)


def _snapshot_hex(
    *,
    frame: int = 1,
    player_x: float = 64.0,
    pixels: bytes = bytes(128 * 128),
) -> str:
    output = bytearray(b"DGSN")
    output.extend(struct.pack("<II", 7, 1))
    output.extend(SOURCE_HASH)
    output.extend(struct.pack("<II", 42, frame))
    output.extend(struct.pack("<Bh", 1, -108))
    output.extend(bytes((1, 0, 0, 32, 32, 1)))
    output.extend(struct.pack("<I", 42))
    output.extend(bytes(31 * 4))
    output.extend(bytes((3, 0)))
    player = (player_x, 64.0, 0.0, 0.0, 4.0)
    output.extend(
        b"".join(struct.pack("<i", int(value * (1 << 16))) for value in player)
    )
    output.extend(struct.pack("<I", 0))
    output.extend(struct.pack("<I", 0))
    output.extend(struct.pack("<I", 0))
    output.extend(struct.pack("<B", 0))
    output.extend(struct.pack("<I", 0))
    output.extend(
        struct.pack(
            "<ii5iIBiiiBIiBBIIBBBBBBiiiiIiiih",
            *([0] * 33),
            -108,
        )
    )
    output.extend(struct.pack("<B", 0))
    output.extend(bytes((1, 12, 1, 2, 1, 1, 1, 0, 0)))
    output.extend(struct.pack("<hh", 0, 0))
    output.extend(bytes(12 * 4))
    output.extend(bytes(128 * 128))
    output.extend(struct.pack("<BH", 6, 0))
    output.extend(bytes(range(16)))
    output.extend(bytes(range(16)))
    output.extend(bytes((1, *([0] * 15))))
    output.extend(struct.pack("<iihhHHh", 0, 0, 0, 0, 128, 128, -108))
    assert len(pixels) == 128 * 128
    output.extend(pixels)
    return output.hex()


def _native_trace(*, player_x: float = 64.0, pixels: bytes = bytes(128 * 128)) -> dict:
    snapshot = _snapshot_hex(player_x=player_x, pixels=pixels)
    return {
        "frames": [
            {
                "frame": 1,
                "input_mask": 32,
                "previous_input_mask": 32,
                "mode": "transition_to_game",
                "game_ready": False,
                "started": True,
                "dead": False,
                "done": False,
                "reward_raw": 0,
                "events": [],
                "state_hash": 0,
                "pixel_hash": 0,
                "snapshot_hex": snapshot,
            }
        ],
        "result": {"frames": 1, "done": False},
    }


def _oracle_trace(*, player_x: float = 64.0, pixels: bytes = bytes(128 * 128)) -> dict:
    return {
        "frames": [
            {
                "frame": 1,
                "state": {
                    "frame": 1,
                    "player": {
                        "x": player_x,
                        "y": 64.0,
                        "vx": 0.0,
                        "vy": 0.0,
                        "size": 4.0,
                    },
                    "enemies": [],
                    "aoes": [],
                },
                "input": {
                    "mask": 32,
                    "previous_mask": 32,
                    "mode": 4,
                    "dead": False,
                },
                "pixels": {
                    "encoding": "palette_index_u8_row_major",
                    "width": 128,
                    "height": 128,
                    "data_hex": pixels.hex(),
                },
                "done": False,
            }
        ],
        "result": {"frames": 1},
    }


def _source_map() -> dict[str, object]:
    return {
        "functions": [
            {
                "pico8_name": "updategame",
                "rust_target": "dodge_core::game::updategame",
                "source": {"section": "lua", "span": [287, 355]},
            },
            {
                "pico8_name": "drawtransition",
                "rust_target": "dodge_core::raster::drawtransition",
                "source": {"section": "lua", "span": [502, 527]},
            },
            {
                "pico8_name": "_update60",
                "rust_target": "dodge_core::compat::_update60",
                "source": {"section": "lua", "span": [136, 183]},
            },
        ]
    }


def test_native_snapshot_decoder_exposes_full_fixed_state_and_pixels() -> None:
    snapshot = decode_native_snapshot(_snapshot_hex())

    assert snapshot.frame == 1
    assert snapshot.mode == "transition_to_game"
    assert snapshot.input_mask == 32
    assert snapshot.previous_input_mask == 32
    assert snapshot.player == (4_194_304, 4_194_304, 0, 0, 262_144)
    assert len(snapshot.pixels) == 128 * 128


def test_field_mismatch_reports_first_path_and_source_span() -> None:
    report = compare_native_to_oracle(
        _native_trace(),
        _oracle_trace(player_x=65.0),
        source_map=_source_map(),
    )

    assert report["status"] == "mismatch"
    assert report["frames_compared"] == 1
    mismatch = report["first_mismatch"]
    assert mismatch["frame"] == 1
    assert mismatch["path"] == "player.x"
    assert mismatch["expected"] == 65.0
    assert mismatch["actual"] == pytest.approx(64.0)
    assert mismatch["source"]["span"] == [287, 355]


def test_pixel_mismatch_reports_row_major_coordinate_and_draw_span() -> None:
    pixels = bytearray(128 * 128)
    pixels[5] = 7
    report = compare_native_to_oracle(
        _native_trace(),
        _oracle_trace(pixels=bytes(pixels)),
        source_map=_source_map(),
    )

    mismatch = report["first_mismatch"]
    assert mismatch["path"] == "pixels[0,5]"
    assert mismatch["expected"] == 7
    assert mismatch["actual"] == 0
    assert mismatch["source"]["span"] == [502, 527]


def test_equal_trace_is_accepted_without_aggregate_hash_only_comparison() -> None:
    report = compare_native_to_oracle(
        _native_trace(),
        _oracle_trace(),
        source_map=_source_map(),
    )

    assert report["status"] == "match"
    assert report["first_mismatch"] is None


def test_reward_mismatch_is_reported_before_state_or_pixel_comparison() -> None:
    oracle = _oracle_trace()
    oracle["frames"][0]["reward"] = 1.0
    report = compare_native_to_oracle(
        _native_trace(),
        oracle,
        source_map=_source_map(),
        compare_pixels=False,
    )

    mismatch = report["first_mismatch"]
    assert mismatch["path"] == "reward"
    assert mismatch["expected"] == 1.0
    assert mismatch["actual"] == 0.0


def test_event_mismatch_is_reported_as_ordered_frame_data() -> None:
    oracle = _oracle_trace()
    oracle["frames"][0]["events"] = ["enemy_spawn"]
    report = compare_native_to_oracle(
        _native_trace(),
        oracle,
        source_map=_source_map(),
        compare_pixels=False,
    )

    mismatch = report["first_mismatch"]
    assert mismatch["path"] == "events"
    assert mismatch["expected"] == ["enemy_spawn"]
    assert mismatch["actual"] == []


def test_snapshot_decoder_rejects_trailing_bytes() -> None:
    with pytest.raises(NativeDifferentialError, match="trailing"):
        decode_native_snapshot(_snapshot_hex() + "00")
