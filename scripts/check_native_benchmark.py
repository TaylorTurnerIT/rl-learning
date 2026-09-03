"""Run the accepted native workload and enforce its P6 regression budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np

from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import NativeBatchEnvironment, NativeBatchResult
from dodge.native.manifest import file_identity, manifest_for_path

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
P6_RAW_PATH: Final = PROJECT_ROOT / "context/kits/dodge-native/p6-benchmark-raw.json"
SOURCE_PATH: Final = PROJECT_ROOT / "src/dodge/game/dodge.p8"
PEMSA_PATH: Final = PROJECT_ROOT / "src/dodge/runtime/pemsa"
CRITERION_ESTIMATES_PATH: Final = (
    PROJECT_ROOT / "native/target/criterion/batch_full_state_pixels/new/estimates.json"
)
DEFAULT_OUTPUT: Final = (
    PROJECT_ROOT / "context/kits/dodge-native/p7-benchmark-report.json"
)
DEFAULT_REPETITIONS: Final = 5
DEFAULT_LANE_COUNT: Final = 32
DEFAULT_LANE_STEPS: Final = 1_024
STEP_FRAMES: Final = 4
SEED_LIMIT: Final = 32_768
MEDIAN_BUDGET_MULTIPLIER: Final = 1.25
STDDEV_BUDGET_MULTIPLIER: Final = 2.0
STDDEV_BUDGET_FLOOR_SECONDS: Final = 0.05


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-native-benchmark")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=positive_int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--lane-count", type=positive_int, default=DEFAULT_LANE_COUNT)
    parser.add_argument("--lane-steps", type=positive_int, default=DEFAULT_LANE_STEPS)
    arguments = parser.parse_args(argv)
    if arguments.lane_steps % arguments.lane_count:
        parser.error("lane-steps must be divisible by lane-count")

    try:
        baseline = _load_baseline()
        durations, digests = _run_benchmark(
            repetitions=arguments.repetitions,
            lane_count=arguments.lane_count,
            lane_steps=arguments.lane_steps,
        )
        report = _build_report(
            arguments,
            baseline=baseline,
            durations=durations,
            digests=digests,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"check-native-benchmark: {error}", file=sys.stderr)
        return 1

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "within_budget" else 2


def _load_baseline() -> dict[str, object]:
    payload = json.loads(P6_RAW_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("P6 benchmark raw output must be an object")
    workload = payload.get("workload")
    cases = payload.get("cases")
    if not isinstance(workload, dict) or not isinstance(cases, dict):
        raise ValueError("P6 benchmark raw output is missing workload or cases")
    case = cases.get("native_full_state_pixels")
    if not isinstance(case, dict):
        raise ValueError("P6 benchmark raw output has no full-observation case")
    expected_workload = {
        "lane_count": DEFAULT_LANE_COUNT,
        "lane_steps": DEFAULT_LANE_STEPS,
        "batch_steps": DEFAULT_LANE_STEPS // DEFAULT_LANE_COUNT,
        "step_frames": STEP_FRAMES,
    }
    for key, expected in expected_workload.items():
        if workload.get(key) != expected:
            raise ValueError(f"P6 workload field {key!r} is not the accepted value")
    if case.get("observation_payload") != {
        "full_state": True,
        "pixels": True,
        "board": False,
    }:
        raise ValueError("P6 full-observation payload is not the accepted workload")
    median = case.get("median_seconds")
    durations = case.get("durations_seconds")
    if not isinstance(median, int | float) or isinstance(median, bool):
        raise ValueError("P6 baseline median is invalid")
    if not isinstance(durations, list) or len(durations) < 2:
        raise ValueError("P6 baseline needs at least two duration samples")
    numeric_durations = [
        value
        for value in durations
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    if len(numeric_durations) != len(durations):
        raise ValueError("P6 baseline durations are invalid")
    return {
        "path": _relative_identity(P6_RAW_PATH),
        "median_seconds": float(median),
        "population_stddev_seconds": statistics.pstdev(numeric_durations),
        "durations_seconds": [float(value) for value in numeric_durations],
    }


def _run_benchmark(
    *, repetitions: int, lane_count: int, lane_steps: int
) -> tuple[list[float], list[int]]:
    batch_steps = lane_steps // lane_count
    durations: list[float] = []
    digests: list[int] = []
    for repetition in range(repetitions):
        environment = NativeBatchEnvironment(
            step_frames=STEP_FRAMES,
            execution="parallel",
            full_state=True,
            pixels=True,
            board=False,
        )
        digest = 0
        start = time.perf_counter()
        try:
            seeds = np.asarray(
                [_seed_for(repetition, lane) for lane in range(lane_count)],
                dtype=np.uint32,
            )
            result = environment.reset_batch(seeds)
            digest ^= _consume_result(result)
            for batch_index in range(batch_steps):
                actions = np.asarray(
                    [
                        (batch_index * lane_count + lane) % len(ACTION_CHOICES)
                        for lane in range(lane_count)
                    ],
                    dtype=np.uint8,
                )
                result = environment.step_batch(actions)
                digest ^= _consume_result(result)
                done_lanes = np.flatnonzero(result.done).astype(np.uint32)
                if done_lanes.size:
                    replacement_seeds = np.asarray(
                        [
                            _replacement_seed(repetition, batch_index, int(lane))
                            for lane in done_lanes
                        ],
                        dtype=np.uint32,
                    )
                    reset = environment.reset_lanes(done_lanes, replacement_seeds)
                    digest ^= _consume_result(reset)
        finally:
            environment.close()
        durations.append(time.perf_counter() - start)
        digests.append(digest)
    return durations, digests


def _build_report(
    arguments: argparse.Namespace,
    *,
    baseline: dict[str, object],
    durations: list[float],
    digests: list[int],
) -> dict[str, object]:
    median = statistics.median(durations)
    stddev = statistics.pstdev(durations) if len(durations) > 1 else 0.0
    median_baseline = float(baseline["median_seconds"])
    stddev_baseline = float(baseline["population_stddev_seconds"])
    median_limit = median_baseline * MEDIAN_BUDGET_MULTIPLIER
    stddev_limit = max(
        STDDEV_BUDGET_FLOOR_SECONDS,
        stddev_baseline * STDDEV_BUDGET_MULTIPLIER,
    )
    median_pass = median <= median_limit
    stddev_pass = stddev <= stddev_limit
    source = manifest_for_path(SOURCE_PATH)
    return {
        "phase": "P7",
        "schema_version": 1,
        "status": "within_budget" if median_pass and stddev_pass else "regression",
        "workload": {
            "lane_count": arguments.lane_count,
            "lane_steps": arguments.lane_steps,
            "batch_steps": arguments.lane_steps // arguments.lane_count,
            "step_frames": STEP_FRAMES,
            "action_schedule": "global_decision_index modulo nine",
            "observation_payload": {
                "full_state": True,
                "pixels": True,
                "board": False,
            },
        },
        "repetitions": arguments.repetitions,
        "raw": {
            "durations_seconds": durations,
            "digest_per_repetition": digests,
        },
        "statistics": {
            "median_seconds": median,
            "population_stddev_seconds": stddev,
            "throughput_lane_steps_per_second": arguments.lane_steps / median,
        },
        "budget": {
            "baseline": baseline,
            "median_multiplier": MEDIAN_BUDGET_MULTIPLIER,
            "median_limit_seconds": median_limit,
            "median_within_budget": median_pass,
            "stddev_multiplier": STDDEV_BUDGET_MULTIPLIER,
            "stddev_floor_seconds": STDDEV_BUDGET_FLOOR_SECONDS,
            "stddev_limit_seconds": stddev_limit,
            "stddev_within_budget": stddev_pass,
        },
        "provenance": {
            "source": {
                "path": _relative_path(SOURCE_PATH),
                "sha256": source.sha256,
                "section_sha256": {
                    section.name: section.sha256 for section in source.sections
                },
            },
            "pemsa": _relative_identity(PEMSA_PATH),
            "native_cargo_lock": _relative_identity(PROJECT_ROOT / "native/Cargo.lock"),
            "uv_lock": _relative_identity(PROJECT_ROOT / "uv.lock"),
            "python": sys.version,
            "rustc": _command_version(("rustc", "-Vv")),
            "cargo": _command_version(("cargo", "-V")),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "criterion": _criterion_evidence(),
        },
        "claims_not_made": [
            "benchmark equivalence across different hosts",
            "mathematical proof of full-game equivalence",
        ],
    }


def _criterion_evidence() -> dict[str, object]:
    try:
        estimates = json.loads(CRITERION_ESTIMATES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Criterion estimates are missing; run `just dodge-native-bench --noplot`"
        ) from error
    if not isinstance(estimates, dict) or not isinstance(estimates.get("median"), dict):
        raise ValueError("Criterion estimates must contain a median object")
    return {
        "path": _relative_path(CRITERION_ESTIMATES_PATH),
        "sha256": _sha256_file(CRITERION_ESTIMATES_PATH),
        "estimates": estimates,
    }


def _consume_result(result: NativeBatchResult) -> int:
    digest = int(result.state_hashes.sum(dtype=np.uint64))
    digest ^= int(result.pixel_hashes.sum(dtype=np.uint64))
    digest ^= int(result.frames.sum(dtype=np.uint64))
    digest ^= int(result.rewards.sum(dtype=np.float32))
    if result.pixels is not None:
        digest ^= int(result.pixels.sum(dtype=np.uint64))
    if result.board is not None:
        digest ^= int(result.board.sum(dtype=np.float64))
    digest ^= sum(len(value) for value in result.snapshot_bytes if value is not None)
    return digest


def _relative_identity(path: Path) -> dict[str, object]:
    identity = file_identity(path).to_json()
    identity["path"] = _relative_path(path)
    return identity


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def _replacement_seed(repetition: int, batch_index: int, lane: int) -> int:
    return (13 + repetition * 1_009 + batch_index * 31 + lane * 17) % SEED_LIMIT


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
