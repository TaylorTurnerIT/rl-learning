from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dodge.control import CARTRIDGE_PATH, ControlRuntimeError
from dodge.native.manifest import (
    CartridgeManifest,
    P8Section,
    canonical_json,
    manifest_for_bytes,
    sections_for_bytes,
)

ASSET_SCHEMA_VERSION = 1
GENERATOR_VERSION = "dodge-native-assets/1"
GFX_WIDTH = 128
GFX_HEIGHT = 128
GFX_SIZE = GFX_WIDTH * GFX_HEIGHT

PICO8_PALETTE: tuple[tuple[int, int, int], ...] = (
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

_HEX_LINE_RE = re.compile(rb"^[0-9a-fA-F]+$")
_TABLE_ASSIGNMENT_RE = re.compile(
    r"^\s*(?P<names>[A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)"
    r"\s*=\s*\{"
)
_SPRITE_RE = re.compile(
    r"\bspr\(\s*(?P<sprite>\d+)\s*,(?P<args>[^)]*)\)"
)


class AssetExtractionError(ControlRuntimeError):
    """The source or generated asset bundle violates its contract."""


@dataclass(frozen=True, slots=True)
class GeneratedAsset:
    path: str
    data: bytes
    source: dict[str, object]

    def manifest(self) -> dict[str, object]:
        return {
            "path": self.path,
            "byte_length": len(self.data),
            "sha256": _sha256(self.data),
            "source": self.source,
        }


def extract_asset_bundle(
    source: Path,
    output: Path,
    *,
    generator_version: str = GENERATOR_VERSION,
) -> dict[str, object]:
    """Extract one immutable cartridge into a deterministic asset directory."""
    if output.exists():
        raise AssetExtractionError(
            f"asset output already exists: {output}; choose a new directory"
        )
    try:
        source_bytes = source.read_bytes()
    except OSError as error:
        raise AssetExtractionError(f"could not read asset source: {error}") from error

    bundle = _build_bundle(
        source_bytes,
        source_name=source.name,
        generator_version=generator_version,
    )
    try:
        if source.read_bytes() != source_bytes:
            raise AssetExtractionError("cartridge changed during asset extraction")
    except OSError as error:
        raise AssetExtractionError(
            f"could not re-read asset source after extraction: {error}"
        ) from error

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        for asset in bundle["_assets"]:
            assert isinstance(asset, GeneratedAsset)
            destination = temporary / asset.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(asset.data)
        manifest = dict(bundle["manifest"])
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
        temporary.replace(output)
    except OSError as error:
        raise AssetExtractionError(f"could not write asset output: {error}") from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return manifest


def validate_asset_bundle(
    output: Path,
    source: Path,
    *,
    generator_version: str = GENERATOR_VERSION,
) -> dict[str, object]:
    """Validate generated files before a native consumer loads any asset."""
    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_bytes = source.read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise AssetExtractionError(f"could not read asset bundle: {error}") from error
    if not isinstance(manifest, dict):
        raise AssetExtractionError("asset manifest must be a JSON object")
    if manifest.get("schema_version") != ASSET_SCHEMA_VERSION:
        raise AssetExtractionError("asset manifest schema version mismatch")
    if manifest.get("generator_version") != generator_version:
        raise AssetExtractionError("asset generator version mismatch")

    expected_source = manifest.get("source")
    if not isinstance(expected_source, dict):
        raise AssetExtractionError("asset manifest has no source identity")
    actual_source = manifest_for_bytes(source_bytes, path=source.name).to_json()
    if expected_source.get("sha256") != actual_source["sha256"]:
        raise AssetExtractionError("asset source hash is stale")
    if expected_source.get("byte_length") != actual_source["byte_length"]:
        raise AssetExtractionError("asset source length is stale")
    expected_sections = expected_source.get("sections")
    actual_sections = actual_source.get("sections")
    if expected_sections != actual_sections:
        raise AssetExtractionError("asset section identity is stale")

    files = manifest.get("files")
    if not isinstance(files, list):
        raise AssetExtractionError("asset manifest has no generated files")
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise AssetExtractionError("asset manifest contains an invalid file record")
        path = output / item["path"]
        try:
            data = path.read_bytes()
        except OSError as error:
            raise AssetExtractionError(
                f"asset file is missing: {item['path']}"
            ) from error
        if len(data) != item.get("byte_length") or _sha256(data) != item.get("sha256"):
            raise AssetExtractionError(f"asset file hash mismatch: {item['path']}")
    try:
        source_map = json.loads((output / "source_map.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AssetExtractionError("asset source map is missing or invalid") from error
    validate_source_map(source_map, manifest_for_bytes(source_bytes, path=source.name))
    try:
        if source.read_bytes() != source_bytes:
            raise AssetExtractionError("cartridge changed during asset validation")
    except OSError as error:
        raise AssetExtractionError(
            f"could not re-read asset source during validation: {error}"
        ) from error
    return manifest


def install_compatibility_report(
    output: Path, report: dict[str, object]
) -> dict[str, object]:
    """Install a verified probe report and refresh its generated-file hash."""
    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AssetExtractionError("could not read asset manifest") from error
    if not isinstance(manifest, dict):
        raise AssetExtractionError("asset manifest must be a JSON object")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise AssetExtractionError("asset manifest has no generated files")
    try:
        record = next(
            item for item in files if item.get("path") == "compatibility.json"
        )
    except StopIteration as error:
        raise AssetExtractionError(
            "asset manifest has no compatibility report"
        ) from error
    data = _json_bytes(report)
    record["byte_length"] = len(data)
    record["sha256"] = _sha256(data)
    manifest["compatibility"] = {
        "path": "compatibility.json",
        "status": report.get("status"),
        "sha256": _sha256(data),
    }
    _replace_file(output / "compatibility.json", data)
    _replace_file(manifest_path, _json_bytes(manifest))
    return manifest


def validate_source_map(
    source_map: object, source_manifest: CartridgeManifest
) -> None:
    if not isinstance(source_map, dict):
        raise AssetExtractionError("source map must be a JSON object")
    if source_map.get("source_sha256") != source_manifest.sha256:
        raise AssetExtractionError("source map source hash is stale")
    unresolved = source_map.get("unresolved")
    if not isinstance(unresolved, list):
        raise AssetExtractionError("source map unresolved inventory is invalid")
    if unresolved:
        raise AssetExtractionError("source map contains unresolved symbols")
    functions = source_map.get("functions")
    if not isinstance(functions, list):
        raise AssetExtractionError("source map function inventory is invalid")
    expected = {function.name for function in source_manifest.functions}
    actual: set[str] = set()
    lua_end = next(
        section.end_line
        for section in source_manifest.sections
        if section.name == "lua"
    )
    for function in functions:
        if not isinstance(function, dict):
            raise AssetExtractionError("source map contains an invalid function")
        name = function.get("pico8_name")
        source = function.get("source")
        if not isinstance(name, str) or name in actual:
            raise AssetExtractionError("source map has duplicate or invalid function")
        if not isinstance(source, dict) or source.get("section") != "lua":
            raise AssetExtractionError("source map function section is invalid")
        span = source.get("span")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(value, int) for value in span)
            or not 1 <= span[0] <= span[1] <= lua_end
        ):
            raise AssetExtractionError("source map function span is invalid")
        if function.get("parity_status") == "unresolved":
            raise AssetExtractionError("source map contains unresolved function")
        actual.add(name)
    if actual != expected:
        raise AssetExtractionError("source map function inventory is stale")


def _build_bundle(
    source_bytes: bytes,
    *,
    source_name: str,
    generator_version: str,
) -> dict[str, object]:
    source_manifest = manifest_for_bytes(source_bytes, path=source_name)
    sections = sections_for_bytes(source_bytes)
    by_name = {section.name: section for section in sections}
    required = {"lua", "gfx", "sfx", "music"}
    missing = sorted(required - by_name.keys())
    if missing:
        raise AssetExtractionError(
            f"cartridge is missing required sections: {', '.join(missing)}"
        )

    gfx, gfx_meta = _decode_gfx(by_name["gfx"])
    sprites = _extract_sprites(by_name["lua"])
    sfx = _decode_sfx(by_name["sfx"])
    music = _decode_music(by_name["music"])
    static_tables = _extract_static_tables(source_bytes, by_name["lua"])
    source_map = _build_source_map(
        source_manifest,
        lua_end_line=by_name["lua"].end_line,
        sprites=sprites,
        static_tables=static_tables,
    )
    compatibility = _initial_compatibility_report(
        source_manifest, generator_version=generator_version
    )

    assets = [
        GeneratedAsset(
            "sections/lua.p8",
            by_name["lua"].payload,
            _section_source("lua", by_name["lua"]),
        ),
        GeneratedAsset(
            "sections/gfx.p8",
            by_name["gfx"].payload,
            _section_source("gfx", by_name["gfx"]),
        ),
        GeneratedAsset(
            "sections/sfx.p8",
            by_name["sfx"].payload,
            _section_source("sfx", by_name["sfx"]),
        ),
        GeneratedAsset(
            "sections/music.p8",
            by_name["music"].payload,
            _section_source("music", by_name["music"]),
        ),
        GeneratedAsset(
            "gfx_indices.bin",
            gfx,
            {
                "section": "gfx",
                "span": [by_name["gfx"].start_line, by_name["gfx"].end_line],
                "conversion": "hex palette indexes; implicit rows padded with 0",
            },
        ),
        GeneratedAsset(
            "palette.json",
            _json_bytes(_palette_json()),
            {"section": "gfx", "span": [by_name["gfx"].start_line] * 2},
        ),
        GeneratedAsset(
            "sprites.json",
            _json_bytes({"schema_version": 1, "sprites": sprites}),
            {"section": "lua", "conversion": "numeric spr calls to sheet rectangles"},
        ),
        GeneratedAsset(
            "sfx.json",
            _json_bytes({"schema_version": 1, "records": sfx}),
            _section_source("sfx", by_name["sfx"]),
        ),
        GeneratedAsset(
            "music.json",
            _json_bytes({"schema_version": 1, "records": music}),
            _section_source("music", by_name["music"]),
        ),
        GeneratedAsset(
            "static_tables.json",
            _json_bytes({"schema_version": 1, "tables": static_tables}),
            {"section": "lua", "conversion": "source-preserved literal tables"},
        ),
        GeneratedAsset(
            "source_map.json",
            _json_bytes(source_map),
            {"section": "lua", "conversion": "symbol inventory"},
        ),
        GeneratedAsset(
            "compatibility.json",
            _json_bytes(compatibility),
            {"section": "lua", "conversion": "P2 accepted compatibility probes"},
        ),
    ]
    manifest = {
        "schema_version": ASSET_SCHEMA_VERSION,
        "generator_version": generator_version,
        "source": source_manifest.to_json(),
        "graphics": gfx_meta,
        "palette": {
            "name": "pico8_default_16",
            "entries": len(PICO8_PALETTE),
            "index_type": "u8",
        },
        "sprites": {"count": len(sprites)},
        "sfx": {"count": len(sfx), "encoding": "source_hex_168"},
        "music": {"count": len(music), "encoding": "source_line"},
        "static_tables": {"count": len(static_tables)},
        "source_map": {
            "path": "source_map.json",
            "unresolved": source_map["unresolved"],
        },
        "compatibility": {
            "path": "compatibility.json",
            "status": compatibility["status"],
        },
        "files": [asset.manifest() for asset in assets],
        "_assets": assets,
    }
    return {"manifest": _without_private(manifest), "_assets": assets}


def _decode_gfx(section: P8Section) -> tuple[bytes, dict[str, object]]:
    lines = [line.rstrip(b"\r\n") for line in section.payload_lines]
    if len(lines) > GFX_HEIGHT:
        raise AssetExtractionError("gfx section has more than 128 rows")
    rows: list[bytes] = []
    for row, line in enumerate(lines):
        if len(line) != GFX_WIDTH or _HEX_LINE_RE.fullmatch(line) is None:
            raise AssetExtractionError(f"gfx row {row} is not 128 hex indexes")
        rows.append(bytes(int(value, 16) for value in line.decode("ascii")))
    rows.extend(b"\0" * GFX_WIDTH for _ in range(GFX_HEIGHT - len(rows)))
    pixels = b"".join(rows)
    if len(pixels) != GFX_SIZE or any(value > 15 for value in pixels):
        raise AssetExtractionError("gfx decode did not produce 128x128 indexes")
    return pixels, {
        "width": GFX_WIDTH,
        "height": GFX_HEIGHT,
        "source_rows": len(lines),
        "implicit_zero_rows": GFX_HEIGHT - len(lines),
        "encoding": "palette_index_u8_row_major",
        "sha256": _sha256(pixels),
    }


def _extract_sprites(section: P8Section) -> list[dict[str, object]]:
    source_lines = [line.decode("utf-8") for line in section.payload_lines]
    found: dict[int, dict[str, object]] = {}
    for offset, line in enumerate(source_lines):
        for match in _SPRITE_RE.finditer(line):
            sprite = int(match.group("sprite"))
            args = [value.strip() for value in match.group("args").split(",")]
            width = _numeric_arg(args, 2, 1)
            height = _numeric_arg(args, 3, 1)
            found.setdefault(
                sprite,
                {
                    "id": sprite,
                    "sheet_x": (sprite % 16) * 8,
                    "sheet_y": (sprite // 16) * 8,
                    "width": width * 8,
                    "height": height * 8,
                    "source_spans": [],
                },
            )
            spans = found[sprite]["source_spans"]
            assert isinstance(spans, list)
            line_number = section.start_line + offset + 1
            spans.append([line_number, line_number])
    return list(found.values())


def _decode_sfx(section: P8Section) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, line in enumerate(section.payload_lines):
        value = line.rstrip(b"\r\n")
        if len(value) != 168 or _HEX_LINE_RE.fullmatch(value) is None:
            raise AssetExtractionError(f"sfx record {index} is not 168 hex characters")
        records.append(
            {
                "id": index,
                "source_line": section.start_line + index + 1,
                "hex": value.decode("ascii").lower(),
                "sha256": _sha256(value.lower()),
            }
        )
    if len(records) != 64:
        raise AssetExtractionError("sfx section must contain 64 records")
    return records


def _decode_music(section: P8Section) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, line in enumerate(section.payload_lines):
        value = line.rstrip(b"\r\n")
        if not value:
            continue
        try:
            text = value.decode("ascii")
        except UnicodeDecodeError as error:
            raise AssetExtractionError(f"music record {index} is not ASCII") from error
        records.append(
            {
                "id": index,
                "source_line": section.start_line + index + 1,
                "text": text,
                "sha256": _sha256(value),
            }
        )
    if len(records) != 32:
        raise AssetExtractionError("music section must contain 32 records")
    return records


def _extract_static_tables(
    source_bytes: bytes, section: P8Section
) -> list[dict[str, object]]:
    source_lines = source_bytes.decode("utf-8").splitlines()
    allowed = {
        "smallsettings",
        "spawns",
        "dirx",
        "diry",
        "difspd",
        "difest",
        "startspd",
        "startest",
        "startbs",
        "incbs",
        "tarbs",
        "startbm",
        "incbm",
        "tarbm",
    }
    tables: list[dict[str, object]] = []
    for index in range(section.start_line, section.end_line + 1):
        line = source_lines[index - 1]
        match = _TABLE_ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        names = [name.strip() for name in match.group("names").split(",")]
        if not any(name in allowed for name in names):
            continue
        end = _table_end(source_lines, index - 1)
        text = "\n".join(source_lines[index - 1 : end])
        if re.search(r"=\s*\{\s*\}\s*$", text):
            continue
        for name in names:
            if name in allowed:
                tables.append(
                    {
                        "name": name,
                        "source_span": [index, end],
                        "source": text,
                        "source_sha256": _sha256(text.encode("utf-8")),
                        "status": "source_preserved",
                    }
                )
    return tables


def _build_source_map(
    manifest: CartridgeManifest,
    *,
    lua_end_line: int,
    sprites: list[dict[str, object]],
    static_tables: list[dict[str, object]],
) -> dict[str, object]:
    functions = list(manifest.functions)
    mapped_functions: list[dict[str, object]] = []
    for index, function in enumerate(functions):
        next_line = (
            functions[index + 1].line
            if index + 1 < len(functions)
            else lua_end_line + 1
        )
        mapped_functions.append(
            {
                "kind": "function",
                "pico8_name": function.name,
                "source": {"section": "lua", "span": [function.line, next_line - 1]},
                "rust_target": (
                    f"dodge_core::{_function_module(function.name)}::{function.name}"
                ),
                "conversion_note": "translate explicit PICO-8 state and side effects",
                "parity_status": "inventory_only",
            }
        )
    return {
        "schema_version": 1,
        "source_sha256": manifest.sha256,
        "functions": mapped_functions,
        "sprites": [
            {
                "id": sprite["id"],
                "source": {"section": "lua", "spans": sprite["source_spans"]},
                "rust_target": "dodge_assets::SpriteSheet",
            }
            for sprite in sprites
        ],
        "static_tables": [
            {
                "name": table["name"],
                "source": {"section": "lua", "span": table["source_span"]},
                "rust_target": "dodge_assets::StaticTable",
                "parity_status": table["status"],
            }
            for table in static_tables
        ],
        "compatibility": [
            {
                "name": name,
                "source": {"section": "lua", "span": [1, lua_end_line]},
                "rust_target": f"dodge_core::compat::{name}",
                "parity_status": "probe_required",
            }
            for name in (
                "numeric",
                "rng",
                "input",
                "persistent_state",
                "palette",
                "camera",
                "fill_pattern",
                "raster",
                "sound_events",
            )
        ],
        "unresolved": [],
    }


def _initial_compatibility_report(
    manifest: CartridgeManifest, *, generator_version: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "pending_p2_probes",
        "source": {"path": manifest.path, "sha256": manifest.sha256},
        "generator_version": generator_version,
        "primitives": [
            {"name": name, "status": "pending"}
            for name in (
                "numeric",
                "rng",
                "input",
                "stat",
                "palette",
                "camera",
                "fill_pattern",
                "raster",
                "sound_events",
            )
        ],
    }


def _palette_json() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "pico8_default_16",
        "encoding": "rgb8",
        "entries": [
            {"index": index, "rgb": list(rgb)}
            for index, rgb in enumerate(PICO8_PALETTE)
        ],
    }


def _section_source(name: str, section: P8Section) -> dict[str, object]:
    return {
        "section": name,
        "span": [section.start_line, section.end_line],
        "source_sha256": _sha256(section.raw),
    }


def _function_module(name: str) -> str:
    if name.startswith("draw") or name in {"_draw", "print2", "fillp_dot"}:
        return "raster"
    if name in {"collisioncheck", "collide", "kamikaze"}:
        return "collision"
    if name in {"addpart", "updateparts", "shatter", "spawntrail"}:
        return "particles"
    if name in {"spawnenemies", "addenemy", "updateenemies", "difficultycurve"}:
        return "enemies"
    if name in {"initpatterns", "updatepatterns", "currentrange", "weightedtbl"}:
        return "patterns"
    if name in {"updatefyou", "fyou2", "fyou"}:
        return "enemies"
    if name.startswith("update") or name in {"_init", "reset", "backtomenu"}:
        return "lifecycle"
    if name in {"iniths", "initspawns", "checkhs", "geths", "iscurrenths"}:
        return "state"
    return "compat"


def _numeric_arg(args: list[str], index: int, default: int) -> int:
    if index >= len(args) or not args[index].isdigit():
        return default
    return int(args[index])


def _table_end(lines: list[str], start: int) -> int:
    depth = 0
    for index in range(start, len(lines)):
        depth += lines[index].count("{") - lines[index].count("}")
        if depth == 0:
            return index + 1
    raise AssetExtractionError("unterminated static table")


def _without_private(value: dict[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if not key.startswith("_")}


def _json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _replace_file(path: Path, data: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
        temporary.replace(path)
    except OSError as error:
        raise AssetExtractionError(
            f"could not replace generated file: {path}"
        ) from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-native-extract-assets",
        description="Extract deterministic PICO-8 assets and source mappings.",
    )
    parser.add_argument("--source", type=Path, default=CARTRIDGE_PATH)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generator-version", default=GENERATOR_VERSION)
    arguments = parser.parse_args(argv)
    try:
        manifest = extract_asset_bundle(
            arguments.source,
            arguments.output,
            generator_version=arguments.generator_version,
        )
    except (AssetExtractionError, OSError, ValueError) as error:
        print(f"dodge-native-extract-assets: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "source_sha256": manifest["source"]["sha256"],
                "files": len(manifest["files"]),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
