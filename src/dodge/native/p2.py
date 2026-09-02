from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from dodge.control import CARTRIDGE_PATH, ControlRuntimeError
from dodge.native.assets import (
    AssetExtractionError,
    install_compatibility_report,
    validate_asset_bundle,
)
from dodge.native.compatibility import run_compatibility_report

P2_REPORT_SCHEMA_VERSION = 1


def accept_p2_bundle(
    *,
    source: Path,
    assets: Path,
    output: Path,
    seed: int,
    raster_fixture: Path | None = None,
) -> dict[str, object]:
    manifest = validate_asset_bundle(assets, source)
    compatibility = run_compatibility_report(seed=seed, source=source)
    if compatibility["status"] != "accepted":
        raise AssetExtractionError("P2 compatibility probes did not pass")
    install_compatibility_report(assets, compatibility)
    manifest = validate_asset_bundle(assets, source)
    report = build_p2_acceptance_report(
        source=source,
        manifest=manifest,
        compatibility=compatibility,
        raster_fixture=raster_fixture,
    )
    write_p2_report(output, report)
    return report


def build_p2_acceptance_report(
    *,
    source: Path,
    manifest: dict[str, object],
    compatibility: dict[str, object],
    raster_fixture: Path | None = None,
) -> dict[str, object]:
    source_record = manifest.get("source")
    if not isinstance(source_record, dict):
        raise AssetExtractionError("P2 report has no source identity")
    if compatibility.get("status") != "accepted":
        raise AssetExtractionError("P2 report requires accepted compatibility")
    unresolved = _source_map_unresolved(manifest)
    if unresolved:
        raise AssetExtractionError("P2 report cannot accept unresolved symbols")

    report: dict[str, object] = {
        "schema_version": P2_REPORT_SCHEMA_VERSION,
        "phase": "P2",
        "status": "accepted_primitive_boundary",
        "handoff": "p3_primitive_boundary_ready",
        "source": {"path": source.name, "sha256": source_record["sha256"]},
        "asset_bundle": {
            "generator_version": manifest.get("generator_version"),
            "manifest_sha256": _sha256_json(manifest),
            "files": len(manifest.get("files", [])),
        },
        "compatibility": {
            "status": compatibility["status"],
            "seed": compatibility.get("seed"),
            "record_count": len(compatibility.get("records", [])),
            "report_sha256": _sha256_json(compatibility),
        },
        "accepted": [
            "source_and_section_identity",
            "indexed_128x128_graphics",
            "palette_and_sprite_metadata",
            "lossless_sfx_and_music_records",
            "source_span_and_static_table_inventory",
            "q16_16_numeric_boundaries",
            "pemsa_libc_rand_stream_and_checkpoint",
            "input_btn_btnp_stat_and_persistent_slots",
            "indexed_camera_palette_fill_and_sprite_raster",
            "stale_output_and_unresolved_symbol_rejection",
        ],
        "deferred": [
            {
                "scope": "full_gameplay_function_parity",
                "owner": "P3/P4",
                "reason": "source map is inventory-only until native frame parity",
            },
            {
                "scope": "audio_waveform_parity",
                "owner": "P4",
                "reason": "P2 preserves record identity, not rendered waveform output",
            },
        ],
        "unresolved": unresolved,
    }
    if raster_fixture is not None:
        try:
            fixture_data = raster_fixture.read_bytes()
        except OSError as error:
            raise AssetExtractionError(
                f"could not read raster fixture: {error}"
            ) from error
        report["raster_fixture"] = {
            "path": raster_fixture.name,
            "sha256": hashlib.sha256(fixture_data).hexdigest(),
        }
    return report


def write_p2_report(path: Path, report: dict[str, object]) -> None:
    data = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise AssetExtractionError(f"could not write P2 report: {error}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _source_map_unresolved(manifest: dict[str, object]) -> list[object]:
    source_map = manifest.get("source_map")
    if not isinstance(source_map, dict):
        raise AssetExtractionError("P2 asset manifest has no source map")
    unresolved = source_map.get("unresolved")
    if not isinstance(unresolved, list):
        raise AssetExtractionError("P2 asset manifest unresolved inventory is invalid")
    return unresolved


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-native-p2-report",
        description="Run P2 compatibility probes and write an acceptance report.",
    )
    parser.add_argument("--source", type=Path, default=CARTRIDGE_PATH)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raster-fixture", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = accept_p2_bundle(
            source=arguments.source,
            assets=arguments.assets,
            output=arguments.output,
            seed=arguments.seed,
            raster_fixture=arguments.raster_fixture,
        )
    except (AssetExtractionError, ControlRuntimeError, OSError, ValueError) as error:
        print(f"dodge-native-p2-report: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "status": report["status"],
                "handoff": report["handoff"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
