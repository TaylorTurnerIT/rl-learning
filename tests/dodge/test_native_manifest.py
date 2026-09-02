from __future__ import annotations

from pathlib import Path

from dodge.native.manifest import manifest_for_bytes, manifest_for_path


def test_manifest_indexes_sections_functions_and_assignments() -> None:
    source = (
        b"pico-8 cartridge\n"
        b"version 42\n"
        b"__lua__\n"
        b"local answer=41\n"
        b"function _init()\n"
        b" answer+=1\n"
        b"end\n"
        b"__gfx__\n"
        b"0123\n"
    )

    manifest = manifest_for_bytes(source, path="fixture.p8")

    assert manifest.schema_version == 1
    assert manifest.path == "fixture.p8"
    assert [section.name for section in manifest.sections] == ["lua", "gfx"]
    assert [function.name for function in manifest.functions] == ["_init"]
    assert manifest.functions[0].line == 5
    assert [(item.name, item.operator) for item in manifest.assignments] == [
        ("answer", "="),
        ("answer", "+="),
    ]


def test_manifest_hash_changes_are_localized_to_changed_section() -> None:
    original = (
        b"pico-8 cartridge\nversion 42\n__lua__\nfunction _init()\nend\n__gfx__\n0123\n"
    )
    changed = original.replace(b"0123", b"4567")

    first = manifest_for_bytes(original, path="fixture.p8")
    second = manifest_for_bytes(changed, path="fixture.p8")

    assert first.sha256 != second.sha256
    assert first.sections[0].sha256 == second.sections[0].sha256
    assert first.sections[1].sha256 != second.sections[1].sha256


def test_checked_in_cartridge_manifest_contains_all_runtime_sections() -> None:
    manifest = manifest_for_path(Path("src/dodge/game/dodge.p8"))

    assert [section.name for section in manifest.sections] == [
        "lua",
        "gfx",
        "sfx",
        "music",
    ]
    assert {function.name for function in manifest.functions} >= {
        "_init",
        "_update60",
        "_draw",
        "updategame",
        "drawgame",
    }
    assert manifest.byte_length > 0
    assert len(manifest.sha256) == 64
