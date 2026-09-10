"""Remote worker used by :mod:`dodge.ng.colab_batch` on Colab."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import signal
import subprocess
import sys
import tarfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path

import numpy as np

from dodge.ng.colab_batch import ColabBatchError, create_artifact_archive
from dodge.ng.manifest import load_manifest


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def create_recovery_archive(
    run_directory: Path,
    archive_path: Path,
    *,
    job_identity_sha256: str,
) -> dict[str, object]:
    """Capture one internally consistent checkpoint while training continues."""
    import torch

    run_directory = Path(run_directory).resolve()
    checkpoint = run_directory / "checkpoint-latest.pt"
    if not checkpoint.is_file():
        return {
            "schema_version": 1,
            "job_identity_sha256": job_identity_sha256,
            "checkpoint_step": None,
            "files": [],
            "skipped": "checkpoint-not-yet-published",
        }
    checkpoint_bytes = checkpoint.read_bytes()
    payload = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
    )
    snapshot = payload.get("replay_snapshot")
    if not isinstance(snapshot, dict):
        raise ColabBatchError("latest checkpoint has no immutable replay snapshot")
    snapshot_name = snapshot.get("path")
    if (
        not isinstance(snapshot_name, str)
        or Path(snapshot_name).name != snapshot_name
        or not snapshot_name.startswith("replay-")
        or not snapshot_name.endswith(".u8.gz")
    ):
        raise ColabBatchError("latest checkpoint replay snapshot name is invalid")
    metrics_path = run_directory / "metrics.jsonl"
    records = [
        {
            "path": "checkpoint-latest.pt",
            "bytes": len(checkpoint_bytes),
            "sha256": _sha256_bytes(checkpoint_bytes),
        },
        {
            "path": "metrics.jsonl",
            "bytes": metrics_path.stat().st_size,
            "sha256": None,
        },
        {
            "path": snapshot_name,
            "bytes": (run_directory / snapshot_name).stat().st_size,
            "sha256": None,
        },
    ]
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_suffix(".tmp")
    try:
        with tarfile.open(temporary, "w:") as archive:
            checkpoint_info = tarfile.TarInfo("checkpoint-latest.pt")
            checkpoint_info.size = len(checkpoint_bytes)
            archive.addfile(checkpoint_info, io.BytesIO(checkpoint_bytes))
            for record in records[1:]:
                path = run_directory / str(record["path"])
                digest = hashlib.sha256()
                size = int(record["bytes"])
                with path.open("rb") as source:
                    remaining = size
                    while remaining:
                        block = source.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ColabBatchError(
                                "recovery source truncated during export"
                            )
                        digest.update(block)
                        remaining -= len(block)
                    source.seek(0)
                    info = tarfile.TarInfo(str(record["path"]))
                    info.size = size
                    archive.addfile(info, source)
                record["bytes"] = info.size
                record["sha256"] = digest.hexdigest()
            manifest = {
                "schema_version": 1,
                "job_identity_sha256": job_identity_sha256,
                "checkpoint_step": payload.get("step"),
                "files": records,
                "created_at": time.time(),
            }
            encoded = json.dumps(manifest, sort_keys=True).encode()
            info = tarfile.TarInfo("recovery-manifest.json")
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))
        temporary.replace(archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _run_status(run_directory: Path) -> dict[str, object]:
    status = _read_json(run_directory / "dashboard" / "status.json")
    return status or {}


def _write_worker_status(
    path: Path,
    *,
    base: Mapping[str, object],
    state: str,
    run_status: Mapping[str, object] | None = None,
    **fields: object,
) -> dict[str, object]:
    run_status = run_status or {}
    value = {
        **base,
        "state": state,
        "run_state": run_status.get("state"),
        "step": run_status.get("step", 0),
        "total_steps": run_status.get("total_steps"),
        "native_steps": run_status.get("native_steps", 0),
        "replay_size": run_status.get("replay_size"),
        "record": run_status.get("record"),
        "best_inner": run_status.get("best_inner"),
        "heartbeat_at": time.time(),
        **fields,
    }
    _atomic_write_json(path, value)
    return value


def _argument_value(arguments: Sequence[str], name: str) -> str | None:
    for index, item in enumerate(arguments):
        if item == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if item.startswith(f"{name}="):
            return item.partition("=")[2]
    return None


def _preflight(
    project_root: Path,
    trainer_arguments: Sequence[str],
    manifest_sha256: str,
    expected_gpu: str,
) -> dict[str, object]:
    """Refuse to start if the remote runtime is not the optimized path."""

    device = _argument_value(trainer_arguments, "--device")
    boundary = _argument_value(trainer_arguments, "--native-pixel-boundary")
    if device not in {"auto", "cuda"}:
        raise ColabBatchError("remote pixel worker requires --device auto or cuda")
    if boundary != "fast":
        raise ColabBatchError(
            "remote pixel worker requires --native-pixel-boundary fast"
        )
    manifest_path = _argument_value(trainer_arguments, "--manifest")
    if not manifest_path:
        raise ColabBatchError("remote trainer arguments do not name a manifest")
    manifest = load_manifest(Path(manifest_path))
    if manifest.sha256 != manifest_sha256:
        raise ColabBatchError("remote manifest hash does not match job manifest")
    try:
        import dodge_native  # noqa: F401
        import torch
    except (ImportError, OSError) as error:
        raise ColabBatchError(f"remote ML/native imports failed: {error}") from error
    if not torch.cuda.is_available():
        raise ColabBatchError("Colab session did not expose a usable CUDA device")
    cuda_name = torch.cuda.get_device_name(0)
    if expected_gpu in {"T4", "L4", "A100", "H100"} and expected_gpu.casefold() not in (
        cuda_name.casefold()
    ):
        raise ColabBatchError(
            f"allocated CUDA device {cuda_name!r} does not match {expected_gpu} request"
        )
    from dodge.native.batch import NativeBatchEnvironment

    environment = NativeBatchEnvironment(
        step_frames=4,
        execution="serial",
        full_state=False,
        pixels=True,
        board=False,
        ml=False,
    )
    try:
        seeds = np.asarray([manifest.training_seeds[0]], dtype=np.uint32)
        environment.reset_batch_with_startup(seeds)
        result = environment.step_pixels(
            np.zeros(environment.lane_count, dtype=np.uint8)
        )
        if result.pixels.shape != (environment.lane_count, 128, 128):
            raise ColabBatchError(
                "remote native pixel boundary returned an invalid shape"
            )
        if result.pixels.dtype != np.uint8 or int(result.pixels.max()) > 15:
            raise ColabBatchError(
                "remote native pixel boundary returned invalid palette data"
            )
    finally:
        environment.close()
    return {
        "cuda": True,
        "cuda_name": cuda_name,
        "cuda_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "native_pixel_boundary": "fast",
        "manifest_sha256": manifest.sha256,
    }


def finalize_artifacts(
    run_directory: Path,
    archive_path: Path,
    *,
    job_id: str,
    manifest_sha256: str,
    job_identity_sha256: str,
    source_bundle_sha256: str,
    native_wheel_sha256: str,
    include_frame_store: bool,
) -> dict[str, object]:
    """Public remote packer entrypoint used by bootstrap and polling."""

    return create_artifact_archive(
        run_directory,
        archive_path,
        job_id=job_id,
        manifest_sha256=manifest_sha256,
        job_identity_sha256=job_identity_sha256,
        source_bundle_sha256=source_bundle_sha256,
        native_wheel_sha256=native_wheel_sha256,
        include_frame_store=include_frame_store,
    )


def run_worker(
    *,
    job_id: str,
    project_root: Path,
    run_directory: Path,
    status_path: Path,
    manifest_sha256: str,
    job_identity_sha256: str,
    source_bundle_sha256: str,
    native_wheel_sha256: str,
    ownership_nonce: str,
    gpu: str,
    heartbeat_seconds: float,
    trainer_arguments: Sequence[str],
    include_frame_store: bool,
) -> int:
    project_root = Path(project_root).resolve()
    run_directory = Path(run_directory).resolve()
    status_path = Path(status_path).resolve()
    base = {
        "schema_version": 1,
        "job_id": job_id,
        "worker_pid": os.getpid(),
        "gpu": gpu,
        "project_root": str(project_root),
        "run_directory": str(run_directory),
        "manifest_sha256": manifest_sha256,
        "job_identity_sha256": job_identity_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "native_wheel_sha256": native_wheel_sha256,
        "ownership_nonce": ownership_nonce,
    }
    try:
        preflight = _preflight(
            project_root,
            trainer_arguments,
            manifest_sha256,
            gpu,
        )
        _write_worker_status(status_path, base=base, state="starting", **preflight)
    except Exception as error:
        _write_worker_status(
            status_path,
            base=base,
            state="failed",
            last_error=f"{type(error).__name__}: {error}",
        )
        return 1

    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(project_root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "dodge.ng.pixel_dqn",
        *trainer_arguments,
    ]
    log_path = project_root.parent / "worker-training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[bytes] | None = None
    prior_handlers: dict[int, object] = {}

    def forward_signal(signum: int, _frame: object) -> None:
        if process is not None and process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        prior_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, forward_signal)
    try:
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _write_worker_status(
                status_path,
                base=base,
                state="running",
                trainer_pid=process.pid,
                **preflight,
            )
            while process.poll() is None:
                current = _run_status(run_directory)
                _write_worker_status(
                    status_path,
                    base=base,
                    state="running",
                    trainer_pid=process.pid,
                    **preflight,
                    run_status=current,
                )
                time.sleep(heartbeat_seconds)
            returncode = process.wait()
    except Exception as error:
        if process is not None and process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        with suppress(Exception):
            _write_worker_status(
                status_path,
                base=base,
                state="failed",
                trainer_pid=process.pid if process is not None else None,
                **preflight,
                run_status=_run_status(run_directory),
                last_error=f"{type(error).__name__}: {error}",
            )
        return 1
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)

    current = _run_status(run_directory)
    assert process is not None
    if returncode != 0:
        _write_worker_status(
            status_path,
            base=base,
            state="failed",
            trainer_pid=process.pid,
            **preflight,
            run_status=current,
            return_code=returncode,
            last_error=f"trainer exited with code {returncode}",
        )
        return returncode
    run_record = _read_json(run_directory / "run.json") or {}
    stopped = current.get("state") == "stopped" or bool(run_record.get("stopped_early"))
    worker_state = "stopped" if stopped else "completed"
    try:
        inventory = finalize_artifacts(
            run_directory,
            project_root.parent / "artifacts.tar.gz",
            job_id=job_id,
            manifest_sha256=manifest_sha256,
            job_identity_sha256=job_identity_sha256,
            source_bundle_sha256=source_bundle_sha256,
            native_wheel_sha256=native_wheel_sha256,
            include_frame_store=include_frame_store,
        )
    except Exception as error:
        _write_worker_status(
            status_path,
            base=base,
            state="failed",
            trainer_pid=process.pid,
            **preflight,
            run_status=current,
            return_code=returncode,
            last_error=f"{type(error).__name__}: {error}",
        )
        return 1
    _write_worker_status(
        status_path,
        base=base,
        state=worker_state,
        trainer_pid=process.pid,
        **preflight,
        run_status=current,
        return_code=returncode,
        artifacts=inventory,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-colab-worker")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--job-identity-sha256", required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--native-wheel-sha256", required=True)
    parser.add_argument("--ownership-nonce", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument(
        "--include-frame-store", dest="include_frame_store", action="store_true"
    )
    parser.add_argument(
        "--no-frame-store", dest="include_frame_store", action="store_false"
    )
    parser.set_defaults(include_frame_store=False)
    parser.add_argument("--trainer-args-json", required=True)
    arguments = parser.parse_args(argv)
    try:
        trainer_arguments = json.loads(arguments.trainer_args_json)
    except json.JSONDecodeError as error:
        print(f"dodge-ng-colab-worker: invalid trainer args: {error}", file=sys.stderr)
        return 2
    if (
        not isinstance(trainer_arguments, list)
        or any(not isinstance(item, str) for item in trainer_arguments)
        or arguments.heartbeat_seconds <= 0
    ):
        print(
            "dodge-ng-colab-worker: invalid trainer/heartbeat arguments",
            file=sys.stderr,
        )
        return 2
    return run_worker(
        job_id=arguments.job_id,
        project_root=arguments.project_root,
        run_directory=arguments.run_dir,
        status_path=arguments.status_path,
        manifest_sha256=arguments.manifest_sha256,
        job_identity_sha256=arguments.job_identity_sha256,
        source_bundle_sha256=arguments.source_bundle_sha256,
        native_wheel_sha256=arguments.native_wheel_sha256,
        ownership_nonce=arguments.ownership_nonce,
        gpu=arguments.gpu,
        heartbeat_seconds=arguments.heartbeat_seconds,
        trainer_arguments=trainer_arguments,
        include_frame_store=arguments.include_frame_store,
    )


if __name__ == "__main__":
    raise SystemExit(main())
