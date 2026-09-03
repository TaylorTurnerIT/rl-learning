"""Verify repeated native observations and record their build provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Final

import numpy as np

from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import NativeBatchEnvironment, NativeBatchResult
from dodge.native.manifest import canonical_json, file_identity, manifest_for_path

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE_PATH: Final = PROJECT_ROOT / "src/dodge/game/dodge.p8"
PEMSA_PATH: Final = PROJECT_ROOT / "src/dodge/runtime/pemsa"
ASSET_ROOT: Final = PROJECT_ROOT / "src/dodge/runtime/.native-assets-check-final"
ASSET_MANIFEST_PATH: Final = ASSET_ROOT / "manifest.json"
FUZZ_REPORT_PATH: Final = PROJECT_ROOT / "context/kits/dodge-native/p7-fuzz-report.json"
DEFAULT_OUTPUT: Final = PROJECT_ROOT / "context/kits/dodge-native/p7-provenance.json"
DEFAULT_LANE_COUNT: Final = 8
DEFAULT_DECISION_STEPS: Final = 128
STEP_FRAMES: Final = 4
SEED_LIMIT: Final = 32_768
OBSERVATION_FIELDS: Final = (
    "lane_ids",
    "frames",
    "frames_advanced",
    "rewards",
    "done",
    "seeds",
    "state_hashes",
    "pixel_hashes",
    "modes",
    "event_flags",
    "pixels",
    "board",
)
REQUIRED_CONFIG_PATHS: Final = (
    Path("native/rust-toolchain.toml"),
    Path("native/.cargo/config.toml"),
    Path("native/rustfmt.toml"),
    Path("native/clippy.toml"),
    Path("native/.config/nextest.toml"),
)
IMPLEMENTATION_PATHS: Final = (
    Path("native/crates/dodge-core/src/game.rs"),
    Path("native/crates/dodge-batch/src/lib.rs"),
    Path("native/crates/dodge-python/src/lib.rs"),
    Path("src/dodge/native/batch.py"),
    Path("scripts/record_native_provenance.py"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="record-native-provenance")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lane-count", type=positive_int, default=DEFAULT_LANE_COUNT)
    parser.add_argument(
        "--decision-steps",
        type=positive_int,
        default=DEFAULT_DECISION_STEPS,
    )
    arguments = parser.parse_args(argv)

    try:
        provenance = _provenance(arguments.lane_count, arguments.decision_steps)
        first = _run_once(arguments.lane_count, arguments.decision_steps)
        second = _run_once(arguments.lane_count, arguments.decision_steps)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"record-native-provenance: {error}", file=sys.stderr)
        return 1

    records_match = first["records"] == second["records"]
    report = {
        "phase": "P7",
        "schema_version": 1,
        "status": "reproducible" if records_match else "mismatch",
        "claim": (
            "same native seeds, actions, configuration, and observation flags "
            "produce byte-identical batch results"
        ),
        "workload": {
            "lane_count": arguments.lane_count,
            "decision_steps": arguments.decision_steps,
            "step_frames": STEP_FRAMES,
            "action_schedule": "(decision_index * 7 + lane_index * 3) modulo nine",
            "reset_schedule": (
                "done lanes reset immediately with deterministic replacement seeds"
            ),
            "observation_flags": {
                "full_state": True,
                "pixels": True,
                "board": True,
            },
        },
        "provenance": provenance,
        "runs": {
            "first": _run_report(first),
            "second": _run_report(second),
            "record_count_equal": len(first["records"]) == len(second["records"]),
            "record_bytes_equal": records_match,
        },
        "claims_not_made": [
            "mathematical proof of full-game equivalence",
            "cross-platform floating-point identity",
            "owner visual approval",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if records_match else 2


def _run_once(lane_count: int, decision_steps: int) -> dict[str, object]:
    seeds = np.asarray(
        [_seed_for(0, lane) for lane in range(lane_count)], dtype=np.uint32
    )
    records: list[bytes] = []
    reset_count = 0
    done_count = 0
    with NativeBatchEnvironment(
        step_frames=STEP_FRAMES,
        execution="parallel",
        full_state=True,
        pixels=True,
        board=True,
    ) as environment:
        result = environment.reset_batch(seeds)
        records.append(_canonical_result_bytes(result))
        reset_count += 1
        for decision in range(decision_steps):
            actions = np.asarray(
                [
                    (decision * 7 + lane * 3) % len(ACTION_CHOICES)
                    for lane in range(lane_count)
                ],
                dtype=np.uint8,
            )
            result = environment.step_batch(actions)
            records.append(_canonical_result_bytes(result))
            done_lanes = np.flatnonzero(result.done).astype(np.uint32)
            done_count += int(done_lanes.size)
            if done_lanes.size:
                replacement_seeds = np.asarray(
                    [_replacement_seed(decision, int(lane)) for lane in done_lanes],
                    dtype=np.uint32,
                )
                reset = environment.reset_lanes(done_lanes, replacement_seeds)
                records.append(_canonical_result_bytes(reset))
                reset_count += 1
    return {
        "records": tuple(records),
        "record_hashes": tuple(_sha256(record) for record in records),
        "record_count": len(records),
        "reset_count": reset_count,
        "done_count": done_count,
        "final_record_sha256": _sha256(records[-1]) if records else None,
        "run_sha256": _sha256(b"".join(_frame(record) for record in records)),
    }


def _provenance(lane_count: int, decision_steps: int) -> dict[str, object]:
    source = manifest_for_path(SOURCE_PATH)
    pemsa = file_identity(PEMSA_PATH).to_json()
    pemsa["path"] = _relative_path(PEMSA_PATH)
    assets = _asset_provenance(source.sha256)
    extension = _native_extension_identity()
    return {
        "source": {
            "path": _relative_path(SOURCE_PATH),
            "sha256": source.sha256,
            "byte_length": source.byte_length,
            "section_sha256": {
                section.name: section.sha256 for section in source.sections
            },
        },
        "pemsa": pemsa,
        "generated_assets": assets,
        "native_binding": extension,
        "tracked_inputs": [
            _file_identity(PROJECT_ROOT / relative_path)
            for relative_path in REQUIRED_CONFIG_PATHS
        ]
        + [
            _file_identity(PROJECT_ROOT / relative_path)
            for relative_path in IMPLEMENTATION_PATHS
        ]
        + [
            _file_identity(PROJECT_ROOT / "native/Cargo.lock"),
            _file_identity(PROJECT_ROOT / "uv.lock"),
            _file_identity(FUZZ_REPORT_PATH),
        ],
        "toolchain": {
            "rust_channel": _rust_channel(),
            "rustc": _command_version(("rustc", "-Vv")),
            "cargo": _command_version(("cargo", "-V")),
            "python": sys.version,
            "uv": _command_version(("uv", "--version")),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "revision": {
            "git_head": _command_version(("git", "rev-parse", "HEAD")),
            "lane_count": lane_count,
            "decision_steps": decision_steps,
        },
    }


def _asset_provenance(source_sha256: str) -> dict[str, object]:
    if not ASSET_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"generated asset manifest does not exist: {ASSET_MANIFEST_PATH}; "
            "run dodge-native-extract-assets first"
        )
    manifest = json.loads(ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("generated asset manifest must be a JSON object")
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("sha256") != source_sha256:
        raise ValueError(
            "generated asset manifest source hash does not match cartridge"
        )
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("generated asset manifest files must be a list")
    identities = []
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("generated asset manifest contains an invalid file")
        path = ASSET_ROOT / entry["path"]
        identity = _file_identity(path)
        if identity["sha256"] != entry.get("sha256"):
            raise ValueError(f"generated asset hash mismatch: {path}")
        identities.append(identity)
    return {
        "generator_version": manifest.get("generator_version"),
        "manifest": _file_identity(ASSET_MANIFEST_PATH),
        "source_sha256": source_sha256,
        "files": identities,
        "source_map": _file_identity(ASSET_ROOT / "source_map.json"),
        "compatibility": _file_identity(ASSET_ROOT / "compatibility.json"),
    }


def _native_extension_identity() -> dict[str, object]:
    import dodge_native

    package_root = Path(dodge_native.__file__).resolve().parent
    candidates = sorted(package_root.glob("dodge_native*.so"))
    candidate = next(iter(candidates), None)
    if candidate is None:
        raise FileNotFoundError(
            f"dodge_native extension was not found in {package_root}"
        )
    return _file_identity(candidate)


def _run_report(run: dict[str, object]) -> dict[str, object]:
    return {
        key: run[key]
        for key in (
            "record_count",
            "reset_count",
            "done_count",
            "final_record_sha256",
            "run_sha256",
            "record_hashes",
        )
    }


def _canonical_result_bytes(result: NativeBatchResult) -> bytes:
    parts = [b"native-batch-result/v1", b"\x00"]
    for name in OBSERVATION_FIELDS:
        value = getattr(result, name)
        parts.append(_frame(name.encode("utf-8")))
        if value is None:
            parts.append(b"\x00")
            continue
        parts.append(b"\x01")
        array = np.ascontiguousarray(value)
        parts.append(_frame(array.dtype.str.encode("ascii")))
        parts.append(_frame(canonical_json(list(array.shape)).encode("ascii")))
        parts.append(_frame(array.tobytes(order="C")))
    parts.append(_frame(b"snapshot_bytes"))
    for snapshot in result.snapshot_bytes:
        if snapshot is None:
            parts.append(b"\x00")
        else:
            parts.append(b"\x01")
            parts.append(_frame(snapshot))
    return b"".join(parts)


def _file_identity(path: Path) -> dict[str, object]:
    identity = file_identity(path).to_json()
    identity["path"] = _relative_path(path)
    return identity


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _rust_channel() -> str:
    toolchain = (PROJECT_ROOT / "native/rust-toolchain.toml").read_text(
        encoding="utf-8"
    )
    match = re.search(r'^channel\s*=\s*"([^"]+)"', toolchain, re.MULTILINE)
    if match is None:
        raise ValueError("native rust-toolchain.toml has no channel")
    return match.group(1)


def _command_version(command: tuple[str, ...]) -> str:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return completed.stdout.strip() or completed.stderr.strip()


def _seed_for(repetition: int, lane: int) -> int:
    return (42 + repetition * 1_003 + lane * 97) % SEED_LIMIT


def _replacement_seed(decision: int, lane: int) -> int:
    return (13 + decision * 31 + lane * 17) % SEED_LIMIT


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "little") + value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
