from __future__ import annotations

import json
from pathlib import Path

import pytest

from dodge.control import ControlRuntimeError
from dodge.native.assets import (
    GENERATOR_VERSION,
    GFX_SIZE,
    AssetExtractionError,
    extract_asset_bundle,
    validate_asset_bundle,
    validate_source_map,
)
from dodge.native.manifest import (
    manifest_for_path,
    section_for_bytes,
    sections_for_bytes,
)


def test_p8_sections_retain_exact_marker_payload_and_line_span() -> None:
    source = b"pico-8 cartridge\nversion 42\n__lua__\nanswer=1\n__gfx__\n0123\n"

    sections = sections_for_bytes(source)

    assert [section.name for section in sections] == ["lua", "gfx"]
    assert sections[0].raw == b"__lua__\nanswer=1\n"
    assert sections[0].payload == b"answer=1\n"
    assert sections[0].payload_lines == (b"answer=1\n",)
    assert (sections[0].start_line, sections[0].end_line) == (3, 4)
    assert section_for_bytes(source, "gfx").payload == b"0123\n"


def test_extract_real_cartridge_writes_indexed_assets_and_hash_manifest(
    tmp_path: Path,
) -> None:
    source = Path("src/dodge/game/dodge.p8")
    output = tmp_path / "assets"

    manifest = extract_asset_bundle(source, output)
    loaded = validate_asset_bundle(output, source)

    assert loaded == manifest
    assert manifest["generator_version"] == GENERATOR_VERSION
    assert manifest["source"]["sha256"]
    assert manifest["graphics"] == {
        "width": 128,
        "height": 128,
        "source_rows": 105,
        "implicit_zero_rows": 23,
        "encoding": "palette_index_u8_row_major",
        "sha256": manifest["graphics"]["sha256"],
    }
    assert len((output / "gfx_indices.bin").read_bytes()) == GFX_SIZE
    assert manifest["sfx"]["count"] == 64
    assert manifest["music"]["count"] == 32
    assert manifest["sprites"]["count"] >= 7
    assert json.loads((output / "palette.json").read_text())["entries"][15] == {
        "index": 15,
        "rgb": [255, 204, 170],
    }


def test_v85_indexed_graphics_reassemble_source_rows_and_zero_tail(
    tmp_path: Path,
) -> None:
    source = Path("src/dodge/game/dodge.p8")
    output = tmp_path / "assets"
    extract_asset_bundle(source, output)

    source_rows = [
        line.rstrip("\r\n")
        for line in (output / "sections/gfx.p8").read_text().splitlines(keepends=True)
    ]
    pixels = (output / "gfx_indices.bin").read_bytes()
    expected = bytes(int(value, 16) for row in source_rows for value in row)

    assert pixels[: len(expected)] == expected
    assert pixels[len(expected) :] == b"\0" * (GFX_SIZE - len(expected))


def test_v86_audio_records_preserve_source_order_and_payloads(tmp_path: Path) -> None:
    source = Path("src/dodge/game/dodge.p8")
    output = tmp_path / "assets"
    extract_asset_bundle(source, output)
    sfx = json.loads((output / "sfx.json").read_text())["records"]
    music = json.loads((output / "music.json").read_text())["records"]

    source_sfx = [
        line.strip() for line in (output / "sections/sfx.p8").read_text().splitlines()
    ]
    source_music = [
        line.strip()
        for line in (output / "sections/music.p8").read_text().splitlines()
    ]
    assert [record["id"] for record in sfx] == list(range(64))
    assert [record["hex"] for record in sfx] == source_sfx
    assert [record["id"] for record in music] == list(range(32))
    assert [record["text"] for record in music] == source_music


def test_v47_sprite_records_keep_sheet_rectangles_and_source_spans(
    tmp_path: Path,
) -> None:
    source = Path("src/dodge/game/dodge.p8")
    output = tmp_path / "assets"
    extract_asset_bundle(source, output)
    sprites = json.loads((output / "sprites.json").read_text())["sprites"]
    by_id = {sprite["id"]: sprite for sprite in sprites}

    assert by_id[17]["sheet_x"] == 8
    assert by_id[17]["sheet_y"] == 8
    assert (by_id[17]["width"], by_id[17]["height"]) == (112, 56)
    assert by_id[17]["source_spans"] == [[363, 363]]
    assert by_id[1]["source_spans"] == [[404, 404]]


def test_v88_source_map_classifies_every_cartridge_function(tmp_path: Path) -> None:
    source = Path("src/dodge/game/dodge.p8")
    output = tmp_path / "assets"
    extract_asset_bundle(source, output)
    source_map = json.loads((output / "source_map.json").read_text())
    function_names = {
        function["pico8_name"] for function in source_map["functions"]
    }

    assert len(function_names) == 51
    assert "_init" in function_names
    assert "updatepatterns" in function_names
    assert source_map["unresolved"] == []
    assert all(
        function["source"]["section"] == "lua"
        and function["source"]["span"][0] <= function["source"]["span"][1]
        and function["rust_target"].startswith("dodge_core::")
        for function in source_map["functions"]
    )


def test_v88_static_table_inventory_excludes_dynamic_empty_reset(
    tmp_path: Path,
) -> None:
    source = Path("src/dodge/game/dodge.p8")
    output = tmp_path / "assets"
    extract_asset_bundle(source, output)
    tables = json.loads((output / "static_tables.json").read_text())["tables"]
    names = [table["name"] for table in tables]

    assert "smallsettings" in names
    assert "startbs" in names
    assert "spawns" in names
    assert all(table["status"] == "source_preserved" for table in tables)
    assert all(table["source"].rstrip().endswith("}") for table in tables)
    assert all(
        table["source"][table["source"].find("=") + 1 :].strip() != "{}"
        for table in tables
    )


def test_same_cartridge_bytes_produce_identical_asset_trees(tmp_path: Path) -> None:
    source = Path("src/dodge/game/dodge.p8")
    first = tmp_path / "first"
    second = tmp_path / "second"

    extract_asset_bundle(source, first)
    extract_asset_bundle(source, second)

    first_files = sorted(
        path.relative_to(first) for path in first.rglob("*") if path.is_file()
    )
    second_files = sorted(
        path.relative_to(second) for path in second.rglob("*") if path.is_file()
    )
    assert first_files == second_files
    assert all(
        (first / relative).read_bytes() == (second / relative).read_bytes()
        for relative in first_files
    )


def test_validation_rejects_changed_source_hash(tmp_path: Path) -> None:
    original = Path("src/dodge/game/dodge.p8").read_bytes()
    source = tmp_path / "source.p8"
    output = tmp_path / "assets"
    source.write_bytes(original)
    extract_asset_bundle(source, output)
    source.write_bytes(original.replace(b"000000007000", b"100000007000", 1))

    with pytest.raises(ControlRuntimeError, match="source hash is stale"):
        validate_asset_bundle(output, source)


def test_extraction_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    source = Path("src/dodge/game/dodge.p8")
    output = tmp_path / "assets"
    output.mkdir()

    with pytest.raises(ControlRuntimeError, match="already exists"):
        extract_asset_bundle(source, output)


def test_v91_validation_rejects_stale_generator_and_modified_file(
    tmp_path: Path,
) -> None:
    source = Path("src/dodge/game/dodge.p8")
    output = tmp_path / "assets"
    extract_asset_bundle(source, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["generator_version"] = "dodge-native-assets/old"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AssetExtractionError, match="generator version mismatch"):
        validate_asset_bundle(output, source)

    extract_asset_bundle(source, tmp_path / "fresh")
    (tmp_path / "fresh" / "gfx_indices.bin").write_bytes(b"stale")
    with pytest.raises(AssetExtractionError, match="file hash mismatch"):
        validate_asset_bundle(tmp_path / "fresh", source)


def test_v92_validation_rejects_unresolved_source_map_entries(tmp_path: Path) -> None:
    source = Path("src/dodge/game/dodge.p8")
    output = tmp_path / "assets"
    extract_asset_bundle(source, output)
    source_map = json.loads((output / "source_map.json").read_text())
    source_map["unresolved"] = ["unknown_symbol"]

    with pytest.raises(AssetExtractionError, match="unresolved symbols"):
        validate_source_map(source_map, manifest_for_path(source))
