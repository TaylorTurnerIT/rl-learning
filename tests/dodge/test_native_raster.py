from __future__ import annotations

import json
from pathlib import Path

from dodge.native.manifest import file_identity, manifest_for_path
from dodge.native.raster import FULL_FILL_PATTERN, IndexedRaster


def test_v90_camera_and_palette_index_match_p1_pixel_probe() -> None:
    raster = IndexedRaster()
    raster.camera(3, 4)
    raster.pset(3, 4, 5)
    raster.camera()

    assert raster.pget(0, 0) == 5
    assert len(raster.indexed_pixels()) == 128 * 128
    assert max(raster.indexed_pixels()) <= 15


def test_v90_draw_and_screen_palette_remaps_are_separate() -> None:
    raster = IndexedRaster()
    raster.pal(1, 2)
    raster.pset(1, 1, 1)
    raster.pal(2, 3, 1)

    assert raster.pget(1, 1) == 2
    assert raster.display_pixels()[1 + 128] == 3
    assert raster.state_json()["draw_palette"][1] == 2
    assert raster.state_json()["screen_palette"][2] == 3


def test_v90_fill_pattern_and_clip_limit_indexed_writes() -> None:
    raster = IndexedRaster()
    raster.fillp(0b0000_0000_0000_0001)
    raster.clip(0, 0, 2, 2)
    raster.rectfill(0, 0, 3, 3, 5)

    assert raster.pget(0, 0) == 5
    assert raster.pget(1, 0) == 5
    assert raster.pget(0, 1) == 5
    assert raster.pget(2, 2) == 0
    assert raster.pget(3, 3) == 0
    assert raster.fill_pattern != FULL_FILL_PATTERN


def test_v90_primitives_are_inclusive_and_ordered() -> None:
    raster = IndexedRaster()
    raster.line(0, 0, 4, 4, 1)
    assert [raster.pget(index, index) for index in range(5)] == [1] * 5

    raster.rect(0, 0, 3, 3, 2)
    assert sum(value == 2 for value in raster.indexed_pixels()) == 12
    assert raster.operations[:2] == ["line", "line"]
    assert raster.operations[-1] == "rect"


def test_v90_sprite_sampling_preserves_palette_indexes_and_transparency() -> None:
    sheet = bytearray(128 * 128)
    sheet[0] = 4
    sheet[1] = 0
    raster = IndexedRaster()
    raster.sprite(sheet, 0, 10, 10)

    assert raster.pget(10, 10) == 4
    assert raster.pget(11, 10) == 0


def test_v90_clear_and_circle_keep_indexed_framebuffer_bounded() -> None:
    raster = IndexedRaster()
    raster.cls(7)
    raster.circfill(64, 64, 1, 8)

    assert raster.pget(64, 64) == 8
    assert raster.pget(63, 64) == 8
    assert raster.pget(64, 63) == 8
    assert all(0 <= value <= 15 for value in raster.indexed_pixels())


def test_v90_p2_raster_fixture_matches_live_pemsa_probe() -> None:
    fixture = json.loads(
        Path("context/kits/dodge-native/p2-raster-fixture.json").read_text()
    )
    source = manifest_for_path(Path(fixture["source"]["path"]))
    pemsa = file_identity(Path(fixture["pemsa"]["path"]))
    assert source.sha256 == fixture["source"]["sha256"]
    assert pemsa.sha256 == fixture["pemsa"]["sha256"]
    camera_raster = IndexedRaster()
    camera_raster.camera(3, 4)
    camera_raster.pset(3, 4, 5)
    camera_raster.camera()
    raster = IndexedRaster()
    raster.fillp(fixture["fill_pattern"]["pattern"])
    raster.rectfill(0, 0, 3, 3, fixture["fill_pattern"]["color"])

    assert camera_raster.pget(0, 0) == fixture["camera_pixel"]
    assert [
        [raster.pget(x, y) for x in range(4)] for y in range(4)
    ] == fixture["fill_pattern"]["pixels"]
