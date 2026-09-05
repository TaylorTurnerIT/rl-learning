from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from dodge.control import (
    CARTRIDGE_PATH,
    PEMSA_PATH,
    ControlInputError,
    ControlRuntimeError,
    MovementCommand,
    load_commands,
    parse_seed,
)
from dodge.headless import instrument_cartridge
from dodge.native.manifest import (
    CartridgeManifest,
    FileIdentity,
    file_identity,
    manifest_for_path,
)
from dodge.native.trace import (
    CAPTURE_MODE,
    OracleTrace,
    command_schedule,
    parse_full_draw_stdout,
)

DEFAULT_TIMEOUT = 60.0


class XvfbProcess:
    def __init__(self, process: subprocess.Popen[str], display: str) -> None:
        self.process = process
        self.display = display


XvfbStarter = Callable[[float], XvfbProcess]
Runner = Callable[..., subprocess.CompletedProcess[str]]


def run_oracle_trace(
    commands: list[MovementCommand],
    *,
    seed: int,
    source: Path = CARTRIDGE_PATH,
    pemsa: Path = PEMSA_PATH,
    runner: Runner = subprocess.run,
    start_xvfb: XvfbStarter | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,
    capture_frame_limit: int | None = None,
    native_startup_grid_spacing: int | None = None,
    capture_frame_indices: Sequence[int] | None = None,
) -> OracleTrace:
    try:
        original = source.read_text(encoding="utf-8")
        source_manifest = manifest_for_path(source)
        pemsa_identity = file_identity(pemsa)
        instrumented = instrument_cartridge(
            original,
            commands,
            seed=seed,
            render=True,
            wait_for_game_start=True,
            capture_pixels=True,
            capture_frame_limit=capture_frame_limit,
            native_startup_grid_spacing=native_startup_grid_spacing,
            capture_frame_indices=capture_frame_indices,
        )
    except OSError as error:
        raise ControlRuntimeError(f"could not read oracle input: {error}") from error

    with tempfile.TemporaryDirectory(prefix="dodge-native-oracle-") as directory:
        workspace = Path(directory)
        cartridge = workspace / "dodge-full-draw.p8"
        try:
            cartridge.write_text(instrumented, encoding="utf-8")
            xvfb = (start_xvfb or _start_xvfb)(timeout or DEFAULT_TIMEOUT)
            try:
                completed = _run_pemsa(
                    cartridge,
                    workspace=workspace,
                    display=xvfb.display,
                    pemsa=pemsa,
                    runner=runner,
                    timeout=timeout,
                )
            finally:
                _stop_xvfb(xvfb.process)
        except subprocess.TimeoutExpired as error:
            detail = _timeout_text(error.stderr) or _timeout_text(error.stdout)
            suffix = f": {detail.strip()}" if detail.strip() else ""
            timeout_text = f"{timeout:g}" if timeout is not None else "unbounded"
            raise ControlRuntimeError(
                f"native oracle timed out after {timeout_text}s{suffix}"
            ) from error
        except OSError as error:
            raise ControlRuntimeError(
                f"could not run native oracle: {error}"
            ) from error

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ControlRuntimeError(
            f"native oracle Pemsa exited {completed.returncode}: {detail}"
        )

    frames, result = parse_full_draw_stdout(completed.stdout)
    _assert_source_unchanged(source, source_manifest)
    _assert_file_unchanged(pemsa, pemsa_identity)

    provenance: dict[str, object] = {
        "schema_version": 1,
        "capture_mode": CAPTURE_MODE,
        "source": source_manifest.to_json(),
        "pemsa": pemsa_identity.to_json(),
    }
    scenario: dict[str, object] = {
        "seed": seed,
        "initial_state": "first_post_update_post_draw_frame",
        "action_schedule": command_schedule(commands),
        "render": True,
        "wait_for_game_start": True,
    }
    if capture_frame_limit is not None:
        scenario["capture_frame_limit"] = capture_frame_limit
    if native_startup_grid_spacing is not None:
        scenario["native_startup_grid_spacing"] = native_startup_grid_spacing
    if capture_frame_indices is not None:
        scenario["capture_frame_indices"] = list(capture_frame_indices)
    return OracleTrace(
        provenance=provenance,
        scenario=scenario,
        frames=frames,
        result=result,
    )


def write_trace(path: Path, trace: OracleTrace) -> None:
    payload = trace.canonical_bytes()
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
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as error:
        raise ControlRuntimeError(f"could not write oracle trace: {error}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _run_pemsa(
    cartridge: Path,
    *,
    workspace: Path,
    display: str,
    pemsa: Path,
    runner: Runner,
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DISPLAY"] = display
    environment["SDL_VIDEODRIVER"] = "x11"
    environment["SDL_AUDIODRIVER"] = "dummy"
    environment["SDL_RENDER_DRIVER"] = "software"
    try:
        completed = runner(
            [pemsa, cartridge, "--no-splash", "--no-fullscreen"],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise
    except OSError as error:
        raise ControlRuntimeError(
            f"could not run native oracle Pemsa: {error}"
        ) from error
    return completed


def _start_xvfb(timeout: float) -> XvfbProcess:
    try:
        process = subprocess.Popen(
            [
                "Xvfb",
                "-displayfd",
                "1",
                "-screen",
                "0",
                "128x128x24",
                "-nolisten",
                "tcp",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="ascii",
            errors="replace",
        )
    except OSError as error:
        raise ControlRuntimeError(f"could not start Xvfb: {error}") from error

    try:
        if process.stdout is None:
            raise ControlRuntimeError("Xvfb did not provide a display descriptor")
        ready, _, _ = select.select([process.stdout], [], [], timeout)
        if not ready:
            raise ControlRuntimeError(f"timed out after {timeout:g}s waiting for Xvfb")
        display_number = process.stdout.readline().strip()
        if not display_number.isdigit():
            raise ControlRuntimeError("Xvfb returned an invalid display descriptor")
        return XvfbProcess(process, f":{display_number}")
    except Exception:
        _stop_xvfb(process)
        raise


def _stop_xvfb(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    else:
        process.wait()


def _assert_source_unchanged(path: Path, before: CartridgeManifest) -> None:
    try:
        after = manifest_for_path(path)
    except OSError as error:
        raise ControlRuntimeError(
            f"cartridge changed or disappeared during oracle capture: {error}"
        ) from error
    if after != before:
        raise ControlRuntimeError("cartridge changed during oracle capture")


def _assert_file_unchanged(path: Path, before: FileIdentity) -> None:
    try:
        after = file_identity(path)
    except OSError as error:
        raise ControlRuntimeError(
            f"Pemsa changed or disappeared during oracle capture: {error}"
        ) from error
    if after != before:
        raise ControlRuntimeError("Pemsa changed during oracle capture")


def _timeout_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-native-oracle",
        description="Capture a canonical full-draw Dodge cartridge trace.",
    )
    parser.add_argument(
        "--commands", required=True, help="JSON command file, or - for stdin"
    )
    parser.add_argument(
        "--seed", default="42", help="PICO-8 random seed from 0 to 32767"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="trace JSON output path"
    )
    parser.add_argument(
        "--timeout",
        default=DEFAULT_TIMEOUT,
        type=float,
        help=f"bounded Pemsa/Xvfb runtime in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    arguments = parser.parse_args(argv)

    try:
        if arguments.timeout <= 0:
            raise ControlInputError("timeout must be positive")
        seed = parse_seed(arguments.seed)
        commands = load_commands(arguments.commands)
        trace = run_oracle_trace(commands, seed=seed, timeout=arguments.timeout)
        write_trace(arguments.output, trace)
    except (ControlInputError, ControlRuntimeError) as error:
        print(f"dodge-native-oracle: {error}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "frames": len(trace.frames),
                "result": trace.result,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
