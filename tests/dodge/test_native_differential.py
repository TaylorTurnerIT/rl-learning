from __future__ import annotations

import struct

import pytest

from dodge.control import CARTRIDGE_PATH
from dodge.native.differential import (
    NativeDifferentialError,
    compare_native_to_oracle,
    decode_native_snapshot,
    load_source_map,
)
from dodge.native.manifest import manifest_for_path

SOURCE_HASH = bytes.fromhex(
    "7453a9658fd32577385ad72672a54ad84ff70567fadbde75ba6634aa5cc684a3"
)


def _minimal_pattern_bytes() -> bytes:
    output = bytearray()
    output.extend(struct.pack("<Biii", 7, 0, 0, 72_090))
    output.extend(struct.pack("<I", 2))
    output.extend(bytes((1, 3)))
    output.extend(bytes((1, 2, 1, 0)))
    output.extend(bytes((1, 9)))
    output.extend(struct.pack("<iIiI", 72_090, 4, 12_288, 1))
    output.extend(
        struct.pack("<7i", 1 << 16, 2 << 16, 8 << 16, 4 << 16, 1 << 16, 0, 1 << 16)
    )
    output.extend(struct.pack("<I", 4))
    output.extend(bytes((0,)))
    output.extend(struct.pack("<4i", 3 << 16, 4 << 16, 5 << 16, 6 << 16))
    output.extend(bytes((1,)))
    output.extend(struct.pack("<i", 7 << 16))
    output.extend(bytes((2, 1)))
    output.extend(bytes((3,)))
    output.extend(struct.pack("<Iii", 1, 10 << 16, 11 << 16))
    output.extend(struct.pack("<IiBiI", 4, 8 << 16, 1, 9 << 16, 1))
    output.extend(struct.pack("<4i", 12 << 16, 13 << 16, 14 << 16, 15 << 16))
    output.extend(bytes((1, 0)))
    return bytes(output)


def _snapshot_hex(
    *,
    frame: int = 1,
    player_x: float = 64.0,
    pixels: bytes = bytes(128 * 128),
    with_pattern: bool = False,
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
    output.extend(struct.pack("<I", 1 if with_pattern else 0))
    if with_pattern:
        output.extend(_minimal_pattern_bytes())
    output.extend(struct.pack("<B", 1 if with_pattern else 0))
    if with_pattern:
        output.extend(struct.pack("<I", 0))
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

    assert snapshot.source_sha256 == SOURCE_HASH.hex()
    assert snapshot.core_schema_version == 1
    assert snapshot.seed == 42
    assert snapshot.frame == 1
    assert snapshot.mode == "transition_to_game"
    assert snapshot.transition_y == -108
    assert snapshot.started is True
    assert snapshot.game_ready is False
    assert snapshot.dead is False
    assert snapshot.input_mask == 32
    assert snapshot.previous_input_mask == 32
    assert snapshot.input_source_mode is True
    assert snapshot.rng == snapshot.rng.__class__(42, (0,) * 31, 3, 0)
    assert snapshot.player == (4_194_304, 4_194_304, 0, 0, 262_144)
    assert snapshot.enemies == ()
    assert snapshot.particles == ()
    assert snapshot.patterns == ()
    assert snapshot.active_pattern is None
    assert snapshot.spawns == ()
    assert snapshot.physical_screen == bytes(128 * 128)
    assert snapshot.enemy_timer == 0
    assert snapshot.enemy_est == 0
    assert snapshot.enemy_stats == (0,) * 5
    assert snapshot.friendly_timer == 0
    assert snapshot.friendly_enabled is False
    assert snapshot.enemy_max_size == 0
    assert snapshot.speed == 0
    assert snapshot.freeze_rate == 0
    assert snapshot.freeze_active is False
    assert snapshot.freeze_timer == 0
    assert snapshot.size_timer == 0
    assert snapshot.patterns_enabled is False
    assert snapshot.powerups_enabled is False
    assert snapshot.pattern_timer == 0
    assert snapshot.pattern_delay_frames == 0
    assert snapshot.pattern_active is False
    assert snapshot.new_highscore is False
    assert snapshot.can_click is False
    assert snapshot.has_played is False
    assert snapshot.should_collide is False
    assert snapshot.enemy_should_collide is False
    assert snapshot.bounce_cap_static == 0
    assert snapshot.bounce_cap_moving == 0
    assert snapshot.bounce_cap == 0
    assert snapshot.score == 0
    assert snapshot.survival_frames == 0
    assert snapshot.shake == 0
    assert snapshot.camera_x == 0
    assert snapshot.camera_y == 0
    assert snapshot.transition_render_y == -108
    assert snapshot.transition_from == "menu"
    assert snapshot.settings.theme_index == 1
    assert snapshot.settings.theme_background == 12
    assert snapshot.settings.theme_shadow == 1
    assert snapshot.settings.difficulty == 2
    assert snapshot.settings.patterns_enabled is True
    assert snapshot.settings.powerups_enabled is True
    assert snapshot.settings.cursor == 1
    assert snapshot.settings.message_timer == 0
    assert snapshot.settings.message_sprite == 0
    assert snapshot.settings.message_x == 0
    assert snapshot.settings.message_y == 0
    assert snapshot.highscores == (0,) * 12
    assert snapshot.render_state.draw_color == 6
    assert snapshot.render_state.fill_pattern == 0
    assert snapshot.render_state.draw_palette == bytes(range(16))
    assert snapshot.render_state.screen_palette == bytes(range(16))
    assert snapshot.render_state.transparent == bytes((1, *([0] * 15)))
    assert snapshot.render_state.camera_x == 0
    assert snapshot.render_state.camera_y == 0
    assert snapshot.render_state.clip_x == 0
    assert snapshot.render_state.clip_y == 0
    assert snapshot.render_state.clip_width == 128
    assert snapshot.render_state.clip_height == 128
    assert snapshot.render_state.transition_y == -108
    assert len(snapshot.pixels) == 128 * 128


def test_native_snapshot_decoder_exposes_pattern_targets_and_warnings() -> None:
    snapshot = decode_native_snapshot(_snapshot_hex(with_pattern=True))

    assert snapshot.active_pattern == 0
    assert len(snapshot.patterns) == 1
    pattern = snapshot.patterns[0]
    assert pattern.id == 7
    assert pattern.variants == (1, 3)
    assert pattern.smooth is True
    assert pattern.pattern_type == 2
    assert pattern.bounce_cap is True
    assert pattern.spawn_enabled is False
    assert pattern.automatic_variant == 9
    assert pattern.special == 72_090
    assert pattern.counter == 4
    assert pattern.timer == 12_288
    assert len(pattern.rects) == 1
    rect = pattern.rects[0]
    assert rect.x == 1 << 16
    assert rect.target_index == 4
    assert rect.wait == 8 << 16
    assert rect.shown is True
    assert rect.sh == 9 << 16
    assert rect.collision_done is True
    assert rect.finished is False
    assert [target.kind for target in rect.targets] == [
        "move",
        "wait",
        "set_fyou",
        "set_spawns",
    ]
    assert rect.targets[0].move == (3 << 16, 4 << 16, 5 << 16, 6 << 16)
    assert rect.targets[1].wait == 7 << 16
    assert rect.targets[2].set_fyou is True
    assert rect.targets[3].spawns == ((10 << 16, 11 << 16),)
    assert rect.warnings == ((12 << 16, 13 << 16, 14 << 16, 15 << 16),)


def test_source_map_covers_every_cartridge_function() -> None:
    source_map = load_source_map()
    manifest = manifest_for_path(CARTRIDGE_PATH)
    functions = source_map.get("functions")
    assert isinstance(functions, list)
    entries = {
        function["pico8_name"]: function
        for function in functions
        if isinstance(function, dict) and isinstance(function.get("pico8_name"), str)
    }
    assert set(entries) == {function.name for function in manifest.functions}
    assert source_map.get("unresolved", []) == []
    if "source_sha256" in source_map:
        assert source_map["source_sha256"] == manifest.sha256
    for function in manifest.functions:
        entry = entries[function.name]
        source = entry.get("source")
        assert isinstance(source, dict)
        span = source.get("span")
        assert isinstance(span, list)
        assert len(span) == 2
        assert all(isinstance(line, int) for line in span)
        assert isinstance(entry.get("rust_target"), str)
        assert entry["rust_target"]


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
