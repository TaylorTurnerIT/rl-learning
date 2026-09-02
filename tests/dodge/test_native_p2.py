from __future__ import annotations

import json
from pathlib import Path

import pytest

from dodge.native.assets import (
    AssetExtractionError,
    extract_asset_bundle,
    install_compatibility_report,
    validate_asset_bundle,
)
from dodge.native.compatibility import build_compatibility_report
from dodge.native.p2 import build_p2_acceptance_report


def _observed_probe() -> dict[str, object]:
    return {
        "draw": ["5"],
        "input": {
            "input_btn0": ["0", "1", "0", "0", "0"],
            "input_btn1": ["0", "0", "0", "1", "0"],
            "input_btn2": ["0", "0", "0", "1", "0"],
            "input_btn3": ["0", "0", "0", "0", "1"],
            "input_btnp0": ["0", "1", "0", "0", "0"],
            "input_btnp1": ["0", "0", "0", "1", "0"],
            "input_btnp2": ["0", "0", "0", "1", "0"],
            "input_btnp3": ["0", "0", "0", "0", "1"],
            "input_frame": ["1", "2", "3", "4", "5"],
        },
        "list_1": ["1"],
        "list_2": ["3"],
        "list_3": ["4"],
        "list_len": ["3"],
        "numeric_ceil": ["-1"],
        "numeric_floor": ["-2"],
        "numeric_mid": ["4"],
        "numeric_mod": ["3"],
        "rng_first": ["0.0334"],
        "rng_limit": ["3.2996"],
    }


def test_v92_p2_report_installs_compatibility_hash_and_defers_only_later_scopes(
    tmp_path: Path,
) -> None:
    source = Path("src/dodge/game/dodge.p8")
    pemsa = Path("src/dodge/runtime/pemsa")
    assets = tmp_path / "assets"
    extract_asset_bundle(source, assets)
    compatibility = build_compatibility_report(
        seed=42,
        observed=_observed_probe(),  # type: ignore[arg-type]
        source=source,
        pemsa=pemsa,
    )
    assert compatibility["status"] == "accepted"
    manifest = install_compatibility_report(assets, compatibility)
    validate_asset_bundle(assets, source)
    report = build_p2_acceptance_report(
        source=source,
        manifest=manifest,
        compatibility=compatibility,
        raster_fixture=Path("context/kits/dodge-native/p2-raster-fixture.json"),
    )

    assert report["status"] == "accepted_primitive_boundary"
    assert report["handoff"] == "p3_primitive_boundary_ready"
    assert report["unresolved"] == []
    assert len(report["accepted"]) == 10
    assert len(report["deferred"]) == 2
    assert json.loads((assets / "manifest.json").read_text())["compatibility"][
        "status"
    ] == "accepted"


def test_v92_p2_report_rejects_mismatched_compatibility(tmp_path: Path) -> None:
    source = Path("src/dodge/game/dodge.p8")
    assets = tmp_path / "assets"
    extract_asset_bundle(source, assets)

    with pytest.raises(AssetExtractionError, match="accepted compatibility"):
        build_p2_acceptance_report(
            source=source,
            manifest=validate_asset_bundle(assets, source),
            compatibility={"status": "mismatch", "records": []},
        )
