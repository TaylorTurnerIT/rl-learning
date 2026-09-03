from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import NativeBatchEnvironment, NativeBatchResult
from dodge.neat.environment import DodgeEnv

DEFAULT_OUTPUT = Path("context/kits/dodge-native/p6-benchmark-raw.json")
SEED_LIMIT = 32_768


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark-dodge-native-batch")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=positive_int, default=5)
    parser.add_argument("--lane-count", type=positive_int, default=32)
    parser.add_argument("--lane-steps", type=positive_int, default=1_024)
    arguments = parser.parse_args(argv)
    if arguments.lane_steps % arguments.lane_count:
        parser.error("lane-steps must be divisible by lane-count")

    workload = {
        "lane_count": arguments.lane_count,
        "lane_steps": arguments.lane_steps,
        "batch_steps": arguments.lane_steps // arguments.lane_count,
        "step_frames": 4,
        "action_schedule": "global_decision_index modulo nine",
    }
    cases = {
        "native_full_state_pixels": _benchmark_native(
            workload, repetitions=arguments.repetitions, full_state=True, pixels=True
        ),
        "native_pixels_off_board": _benchmark_native(
            workload, repetitions=arguments.repetitions, full_state=False, pixels=False
        ),
        "python_pemsa_interactive": _benchmark_pemsa(
            workload, repetitions=arguments.repetitions
        ),
    }
    baseline = cases["python_pemsa_interactive"]["median_seconds"]
    for name in ("native_full_state_pixels", "native_pixels_off_board"):
        cases[name]["speedup_vs_python_pemsa"] = baseline / cases[name][
            "median_seconds"
        ]
    report = {
        "phase": "P6",
        "schema_version": 1,
        "status": "measured",
        "workload": workload,
        "repetitions": arguments.repetitions,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "cases": cases,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


def _benchmark_native(
    workload: dict[str, int | str],
    *,
    repetitions: int,
    full_state: bool,
    pixels: bool,
) -> dict[str, object]:
    lane_count = int(workload["lane_count"])
    batch_steps = int(workload["batch_steps"])
    durations: list[float] = []
    digests: list[int] = []
    for repetition in range(repetitions):
        environment = NativeBatchEnvironment(
            step_frames=int(workload["step_frames"]),
            execution="parallel",
            full_state=full_state,
            pixels=pixels,
            board=not (full_state or pixels),
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
    median_seconds = statistics.median(durations)
    return {
        "observation_payload": {
            "full_state": full_state,
            "pixels": pixels,
            "board": not (full_state or pixels),
        },
        "durations_seconds": durations,
        "median_seconds": median_seconds,
        "throughput_lane_steps_per_second": int(workload["lane_steps"])
        / median_seconds,
        "digest_per_repetition": digests,
    }


def _benchmark_pemsa(
    workload: dict[str, int | str], *, repetitions: int
) -> dict[str, object]:
    total_steps = int(workload["lane_steps"])
    durations: list[float] = []
    digests: list[int] = []
    for repetition in range(repetitions):
        environment = DodgeEnv(step_frames=int(workload["step_frames"]))
        digest = 0
        start = time.perf_counter()
        try:
            observation = environment.reset(seed=_seed_for(repetition, 0))
            digest ^= observation.raw_state.frame
            for decision in range(total_steps):
                action = ACTION_CHOICES[decision % len(ACTION_CHOICES)]
                transition = environment.step(action)
                digest ^= transition.observation.raw_state.frame
                digest ^= int(round(transition.reward))
                if transition.done and decision + 1 < total_steps:
                    observation = environment.reset(
                        seed=_replacement_seed(repetition, decision, 0)
                    )
                    digest ^= observation.raw_state.frame
        finally:
            environment.close()
        durations.append(time.perf_counter() - start)
        digests.append(digest)
    median_seconds = statistics.median(durations)
    return {
        "observation_payload": {
            "full_state": False,
            "pixels": False,
            "board": False,
            "legacy_raw_state": True,
        },
        "durations_seconds": durations,
        "median_seconds": median_seconds,
        "throughput_lane_steps_per_second": total_steps / median_seconds,
        "digest_per_repetition": digests,
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
