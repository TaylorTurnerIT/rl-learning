"""Compare the longest recorded GA episode against Pemsa frame by frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Final

from dodge.control import MovementCommand
from dodge.headless import run_headless
from dodge.native.differential import compare_native_to_oracle, load_source_map
from dodge.native.ga import DEFAULT_DATABASE, DatasetEpisode, load_longest_episode
from dodge.native.manifest import canonical_json, file_identity
from dodge.native.oracle import OracleTrace, run_oracle_trace, write_trace

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER: Final = PROJECT_ROOT / "native/target/release/dodge-native-runner"
DEFAULT_OUTPUT: Final = (
    PROJECT_ROOT / "context/kits/dodge-native/p7-ga-full-run-report.json"
)
DEFAULT_MAX_FRAMES: Final = 4_096
DEFAULT_TIMEOUT: Final = 180.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="native-ga-differential")
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_DATABASE,
        help="GA SQLite database (opened read-only)",
    )
    parser.add_argument(
        "--episode-id",
        type=positive_int,
        default=None,
        help="replay one episode instead of selecting the longest",
    )
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="retain native and oracle traces when comparison fails",
    )
    parser.add_argument("--max-frames", type=positive_int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--timeout", type=positive_float, default=DEFAULT_TIMEOUT)
    arguments = parser.parse_args(argv)

    try:
        report = run_validation(arguments)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"native-ga-differential: {error}", file=sys.stderr)
        return 1

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "match" else 2


def run_validation(arguments: argparse.Namespace) -> dict[str, object]:
    episode = load_longest_episode(
        arguments.database,
        episode_id=arguments.episode_id,
    )
    if arguments.max_frames < episode.total_frames:
        raise ValueError(
            f"--max-frames {arguments.max_frames} is below the recorded "
            f"episode length {episode.total_frames}"
        )
    if not arguments.runner.is_file():
        raise FileNotFoundError(
            f"native runner does not exist: {arguments.runner}; build it with "
            "cargo build --release --manifest-path native/Cargo.toml "
            "-p dodge-native-runner"
        )

    oracle = run_oracle_trace(
        list(episode.commands),
        seed=episode.seed,
        timeout=arguments.timeout,
    )
    headless_result = run_headless(
        list(episode.commands),
        seed=episode.seed,
        wait_for_game_start=True,
        timeout=arguments.timeout,
    )
    if headless_result != episode.stored_result:
        raise RuntimeError(
            "database episode result differs from its headless replay: "
            f"stored={episode.stored_result!r} replay={headless_result!r}"
        )

    native = _run_native(
        arguments.runner,
        episode.seed,
        episode.commands,
        max_frames=arguments.max_frames,
        timeout=arguments.timeout,
    )
    oracle_value = json.loads(oracle.canonical_bytes())
    comparison = compare_native_to_oracle(
        native,
        oracle_value,
        source_map=load_source_map(),
        compare_pixels=True,
    )
    if arguments.trace_dir is not None and comparison["status"] != "match":
        _write_mismatch_traces(arguments.trace_dir, native, oracle)

    return _report(
        arguments,
        episode,
        headless_result,
        native,
        oracle,
        comparison,
    )


def _run_native(
    runner: Path,
    seed: int,
    commands: tuple[MovementCommand, ...],
    *,
    max_frames: int,
    timeout: float,
) -> dict[str, object]:
    payload = canonical_json(
        [
            {"move": command.move, "duration_ms": command.duration_ms}
            for command in commands
        ]
    )
    completed = subprocess.run(
        [
            str(runner),
            "--commands",
            "-",
            "--seed",
            str(seed),
            "--max-frames",
            str(max_frames),
        ],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"native runner exited {completed.returncode}: {detail}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("native runner output must be a JSON object")
    return value


def _write_mismatch_traces(
    trace_dir: Path,
    native: dict[str, object],
    oracle: OracleTrace,
) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "native.json").write_text(
        json.dumps(native, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_trace(trace_dir / "oracle.json", oracle)


def _report(
    arguments: argparse.Namespace,
    episode: DatasetEpisode,
    headless_result: dict[str, int | float | bool],
    native: dict[str, object],
    oracle: OracleTrace,
    comparison: dict[str, object],
) -> dict[str, object]:
    oracle_bytes = oracle.canonical_bytes()
    native_bytes = canonical_json(native).encode("utf-8")
    command_values = [
        {"move": command.move, "duration_ms": command.duration_ms}
        for command in episode.commands
    ]
    return {
        "phase": "P7",
        "schema_version": 1,
        "validation": "sqlite_ga_longest_episode_full_draw",
        "status": comparison["status"],
        "database": file_identity(arguments.database).to_json(),
        "episode": {
            "id": episode.episode_id,
            "seed": episode.seed,
            "action_hash": episode.action_hash,
            "stored_result": episode.stored_result,
            "skipped_incomplete_episode_ids": list(episode.skipped_episode_ids),
        },
        "schedule": {
            "commands": len(episode.commands),
            "recorded_steps": episode.recorded_steps,
            "genome_actions": len(episode.genome_actions),
            "command_sha256": _sha256_json(command_values),
        },
        "dataset_replay": {
            "mode": "headless_fast_forward_no_draw",
            "result": headless_result,
            "matches_stored_result": headless_result == episode.stored_result,
        },
        "oracle": {
            "frames": len(oracle.frames),
            "result": oracle.result,
            "trace_sha256": _sha256_bytes(oracle_bytes),
            "matches_stored_result": oracle.result == episode.stored_result,
        },
        "native": {
            "frames": len(native.get("frames", [])),
            "result": native.get("result"),
            "trace_sha256": _sha256_bytes(native_bytes),
            "runner": file_identity(arguments.runner).to_json(),
        },
        "comparison": {
            "frames_compared": comparison["frames_compared"],
            "full_indexed_pixels": True,
            "first_mismatch": comparison["first_mismatch"],
        },
        "mode_boundary": {
            "status": "different_execution_length"
            if oracle.result != episode.stored_result
            else "same_execution_length",
            "note": (
                "The GA result is from no-draw fast-forward execution; the "
                "full-draw oracle executes the source draw path, including "
                "drawtransition's source-side update."
            ),
        },
        "claims_not_made": [
            "infinite-input proof",
            "full-game mathematical equivalence",
            "owner visual approval",
            "headless GA result equals the rendered oracle result",
        ],
    }


def _sha256_json(value: object) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
