"""Run deterministic full-draw differential cases against Pemsa."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Final

from dodge.control import CARTRIDGE_PATH, PEMSA_PATH, MovementCommand, parse_commands
from dodge.dataset import ACTION_CHOICES
from dodge.native.differential import compare_native_to_oracle, load_source_map
from dodge.native.manifest import canonical_json, file_identity, manifest_for_path
from dodge.native.oracle import run_oracle_trace

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS: Final = PROJECT_ROOT / "context/kits/dodge-native/corpus"
DEFAULT_OUTPUT: Final = PROJECT_ROOT / "context/kits/dodge-native/p7-fuzz-report.json"
DEFAULT_RUNNER: Final = PROJECT_ROOT / "native/target/release/dodge-native-runner"
FUZZ_CASE_COUNT: Final = 4
FUZZ_COMMAND_COUNT: Final = 13
FUZZ_DURATIONS_MS: Final = (50, 75, 100, 125)
FUZZ_SEEDS: Final = (3, 17, 41, 89)
LCG_MULTIPLIER: Final = 1_664_525
LCG_INCREMENT: Final = 1_013_904_223


def generated_case(index: int) -> dict[str, object]:
    """Return one reproducible schedule from the P7 deterministic generator."""
    if not 0 <= index < FUZZ_CASE_COUNT:
        raise ValueError(f"generated case index must be below {FUZZ_CASE_COUNT}")
    state = (0xD06E_0000 + index * 0x9E37_79B9) & 0xFFFF_FFFF
    commands: list[dict[str, int | str]] = [
        {"move": "x", "duration_ms": 50},
    ]
    for _ in range(FUZZ_COMMAND_COUNT - 1):
        state = _next_state(state)
        move = ACTION_CHOICES[state % len(ACTION_CHOICES)]
        state = _next_state(state)
        duration_ms = FUZZ_DURATIONS_MS[state % len(FUZZ_DURATIONS_MS)]
        commands.append({"move": move, "duration_ms": duration_ms})
    return {"seed": FUZZ_SEEDS[index], "commands": commands}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="native-differential-fuzz")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--cases", type=positive_int, default=FUZZ_CASE_COUNT)
    parser.add_argument("--timeout", type=positive_float, default=45.0)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="write full native and oracle traces for mismatching cases",
    )
    arguments = parser.parse_args(argv)
    if arguments.cases > FUZZ_CASE_COUNT:
        parser.error(f"cases must be at most {FUZZ_CASE_COUNT}")

    try:
        cases = _run_cases(arguments)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"native-differential-fuzz: {error}", file=sys.stderr)
        return 1

    report = _report(arguments, cases)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "match" else 2


def _run_cases(arguments: argparse.Namespace) -> list[dict[str, object]]:
    if not arguments.runner.is_file():
        raise FileNotFoundError(
            f"native runner does not exist: {arguments.runner}; build it with "
            "cargo build --release --manifest-path native/Cargo.toml -p dodge-native-runner"
        )
    source_map = load_source_map(source=CARTRIDGE_PATH)
    cases: list[dict[str, object]] = []
    for index in range(arguments.cases):
        expected = generated_case(index)
        path = arguments.corpus_dir / f"p7-fuzz-{index:02d}.json"
        actual = _load_case(path)
        if actual != expected:
            raise ValueError(f"{path} does not match generated case {index}")
        seed = int(expected["seed"])
        commands_value = expected["commands"]
        if not isinstance(commands_value, list):
            raise ValueError(f"{path} commands are not a list")
        commands = parse_commands(commands_value)
        native = _run_native(arguments.runner, seed, commands, arguments.timeout)
        oracle = run_oracle_trace(
            commands,
            seed=seed,
            timeout=arguments.timeout,
        )
        oracle_value = json.loads(oracle.canonical_bytes())
        comparison = compare_native_to_oracle(
            native,
            oracle_value,
            source_map=source_map,
            compare_pixels=True,
        )
        if arguments.trace_dir is not None and comparison["status"] != "match":
            _write_mismatch_traces(
                arguments.trace_dir,
                index=index,
                native=native,
                oracle_bytes=oracle.canonical_bytes(),
            )
        cases.append(
            {
                "index": index,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "seed": seed,
                "commands": len(commands),
                "scenario_sha256": _sha256_json(expected),
                "native_trace_sha256": _sha256_json(native),
                "oracle_trace_sha256": _sha256_bytes(oracle.canonical_bytes()),
                "frames_compared": comparison["frames_compared"],
                "status": comparison["status"],
                "first_mismatch": comparison["first_mismatch"],
            }
        )
    return cases


def _write_mismatch_traces(
    trace_dir: Path,
    *,
    index: int,
    native: dict[str, object],
    oracle_bytes: bytes,
) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"p7-fuzz-{index:02d}-native.json").write_text(
        json.dumps(native, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (trace_dir / f"p7-fuzz-{index:02d}-oracle.json").write_bytes(oracle_bytes)


def _load_case(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict) or set(value) != {"seed", "commands"}:
        raise ValueError(f"{path} must contain exactly seed and commands")
    return value


def _run_native(
    runner: Path,
    seed: int,
    commands: list[MovementCommand],
    timeout: float,
) -> dict[str, object]:
    payload = json.dumps(
        [{"move": command.move, "duration_ms": command.duration_ms} for command in commands],
        separators=(",", ":"),
    )
    completed = subprocess.run(
        [
            str(runner),
            "--commands",
            "-",
            "--seed",
            str(seed),
            "--max-frames",
            "2048",
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


def _report(
    arguments: argparse.Namespace,
    cases: list[dict[str, object]],
) -> dict[str, object]:
    source = manifest_for_path(CARTRIDGE_PATH)
    return {
        "phase": "P7",
        "schema_version": 1,
        "status": "match" if all(case["status"] == "match" for case in cases) else "mismatch",
        "generator": {
            "algorithm": "uint32 LCG: state = state * 1664525 + 1013904223 modulo 2^32",
            "case_count": len(cases),
            "command_count": FUZZ_COMMAND_COUNT,
            "durations_ms": list(FUZZ_DURATIONS_MS),
            "seeds": list(FUZZ_SEEDS[: len(cases)]),
        },
        "source": {
            "path": str(CARTRIDGE_PATH.relative_to(PROJECT_ROOT)),
            "sha256": source.sha256,
        },
        "pemsa": file_identity(PEMSA_PATH).to_json(),
        "runner": str(arguments.runner.relative_to(PROJECT_ROOT)),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "cases": cases,
        "claims_not_made": [
            "infinite-input proof",
            "full-game mathematical equivalence",
            "owner visual approval",
        ],
    }


def _next_state(state: int) -> int:
    return (state * LCG_MULTIPLIER + LCG_INCREMENT) & 0xFFFF_FFFF


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
