"""Observable, resumable batch execution on Google Colab runtimes.

The Colab MCP project is useful for interactive browser/session access. This
module adds the durable job layer needed by NG experiments: a local manifest,
named-session lifecycle, bounded polling, GPU escalation, and verified
artifact retrieval.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tarfile
import textwrap
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from dodge.control import PROJECT_ROOT
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, load_manifest

JOB_SCHEMA_VERSION: Final[int] = 1
ARTIFACT_SCHEMA_VERSION: Final[int] = 1
DEFAULT_JOB_ROOT: Final[Path] = PROJECT_ROOT / "history" / "dodge" / "ng" / "colab-jobs"
REMOTE_JOB_ROOT: Final[Path] = Path("/content/dodge-ng-jobs")
COLAB_GPU_TIERS: Final[tuple[str, ...]] = ("T4", "L4", "G4", "A100", "H100")
DEFAULT_GPU_TIERS: Final[tuple[str, ...]] = ("T4", "L4", "A100", "H100")
JOB_STATES: Final[tuple[str, ...]] = (
    "planned",
    "provisioning",
    "running",
    "stopping",
    "completed",
    "failed",
    "cancelled",
)
TERMINAL_STATES: Final[frozenset[str]] = frozenset({"completed", "failed", "cancelled"})
REMOTE_STATES: Final[frozenset[str]] = frozenset(
    {"starting", "running", "completed", "failed", "stopped", "cancelled"}
)
REMOTE_TERMINAL_STATES: Final[frozenset[str]] = frozenset(
    {"completed", "failed", "stopped", "cancelled"}
)
MAX_EVENT_BYTES: Final[int] = 2 * 1024 * 1024
MAX_COMMAND_LOG_BYTES: Final[int] = 512 * 1024
STATE_ORDER: Final[dict[str, int]] = {
    "planned": 0,
    "provisioning": 1,
    "running": 2,
    "stopping": 3,
    "completed": 4,
    "failed": 4,
    "cancelled": 4,
}
REQUIRED_ARTIFACTS: Final[tuple[str, ...]] = (
    "checkpoint-latest.pt",
    "metrics.jsonl",
    "run.json",
)
DEFAULT_ARTIFACT_FILES: Final[tuple[str, ...]] = (
    "checkpoint-best.pt",
    "checkpoint-latest.pt",
    "REPORT.md",
    "metrics.jsonl",
    "report.json",
    "run.json",
    "pixel_split_survival.png",
    "pixel_training_curves.png",
    "dashboard/metrics.jsonl",
    "dashboard/status.json",
)
_EXCLUDED_SOURCE_PARTS: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".venv",
        ".devenv",
        ".direnv",
        "target",
        "history",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".torchinductor",
        ".mojo_cache",
    }
)


class ColabBatchError(RuntimeError):
    """Raised when a batch job cannot be safely prepared or completed."""


@dataclass(frozen=True, slots=True)
class JobPaths:
    """Stable local paths for one batch job."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "job.json"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def events(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def source_bundle(self) -> Path:
        return self.root / "source.tar.gz"

    @property
    def resume_bundle(self) -> Path:
        return self.root / "resume.tar.gz"

    @property
    def wheel_directory(self) -> Path:
        return self.root / "wheels"

    @property
    def bootstrap(self) -> Path:
        return self.root / "bootstrap.py"

    @property
    def poller(self) -> Path:
        return self.root / "poll_status.py"

    @property
    def stopper(self) -> Path:
        return self.root / "request_stop.py"

    @property
    def packer(self) -> Path:
        return self.root / "pack_artifacts.py"

    @property
    def recovery_exporter(self) -> Path:
        return self.root / "export_recovery.py"

    @property
    def recovery_archive(self) -> Path:
        return self.root / "recovery.tar"

    @property
    def recovery_directory(self) -> Path:
        return self.root / "recovery" / "latest"

    @property
    def downloaded_archive(self) -> Path:
        return self.root / "artifacts.tar.gz"

    @property
    def artifact_manifest(self) -> Path:
        return self.root / "artifact-manifest.json"

    @property
    def controller_log(self) -> Path:
        return self.logs / "controller.log"

    @property
    def controller_lock(self) -> Path:
        return self.root / "controller.lock"

    @property
    def stop_request(self) -> Path:
        return self.root / "stop-request.json"


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured result of one `colab` CLI operation."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class ColabCLI:
    """Small injectable adapter around the installed `colab` executable."""

    def __init__(
        self,
        executable: str = "colab",
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.executable = executable
        self._runner = runner or subprocess.run

    def call(
        self,
        arguments: Sequence[str],
        *,
        timeout: float = 120.0,
    ) -> CommandResult:
        command = (self.executable, *tuple(str(item) for item in arguments))
        try:
            result = self._runner(
                list(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as error:
            raise ColabBatchError(
                f"Colab CLI not found: {self.executable}; install/authenticate it first"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise ColabBatchError(
                f"Colab CLI command timed out after {timeout:.1f}s: {' '.join(command)}"
            ) from error
        return CommandResult(
            command=command,
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )


def job_paths(job_root: Path, job_id: str) -> JobPaths:
    """Resolve one job directory without permitting path traversal."""

    if not job_id or Path(job_id).name != job_id:
        raise ValueError("job id must be a single safe path component")
    root = Path(job_root).resolve() / job_id
    return JobPaths(root)


def parse_gpu_tiers(value: str | Sequence[str]) -> tuple[str, ...]:
    """Parse and validate an ordered GPU escalation policy."""

    raw_values = value.split(",") if isinstance(value, str) else list(value)
    tiers = tuple(item.strip().upper() for item in raw_values if item.strip())
    if not tiers:
        raise ValueError("GPU tier policy must not be empty")
    unknown = sorted(set(tiers) - set(COLAB_GPU_TIERS))
    if unknown:
        raise ValueError(
            f"unsupported Colab GPU tier(s): {', '.join(unknown)}; "
            f"choose from {', '.join(COLAB_GPU_TIERS)}"
        )
    if len(set(tiers)) != len(tiers):
        raise ValueError("GPU tier policy must not repeat a tier")
    if tiers[0] != "T4":
        raise ValueError("GPU tier policy must start with T4")
    tier_indexes = [COLAB_GPU_TIERS.index(tier) for tier in tiers]
    if tier_indexes != sorted(tier_indexes):
        raise ValueError("GPU tier policy must advance from cheaper to larger tiers")
    return tiers


def _allocation_may_escalate(detail: str) -> bool:
    normalized = detail.casefold()
    capacity_markers = (
        "capacity",
        "unavailable",
        "resource exhausted",
        "could not assign",
        "no backend",
        "temporarily unable",
        "too many assignments",
        "toomanyassignments",
        "precondition failed",
        "http 412",
        "status code 412",
    )
    return any(marker in normalized for marker in capacity_markers)


def _session_exists(result: CommandResult) -> bool:
    if not result.ok:
        return False
    detail = f"{result.stdout}\n{result.stderr}".casefold()
    return "not found" not in detail


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    return slug[:40] or "pixel-dqn"


def _new_job_id(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{_safe_slug(prefix)}-{stamp}-{secrets.token_hex(3)}"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ColabBatchError(f"cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ColabBatchError(f"JSON artifact must contain an object: {path}")
    return value


def load_job(paths: JobPaths) -> dict[str, object]:
    return _read_json(paths.manifest)


def load_status(paths: JobPaths) -> dict[str, object]:
    return _read_json(paths.status)


def append_event(paths: JobPaths, event: str, **fields: object) -> None:
    paths.events.parent.mkdir(parents=True, exist_ok=True)
    payload = {"at": time.time(), "event": event, **fields}
    with paths.events.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        stream.write("\n")
    _trim_text_file(paths.events, MAX_EVENT_BYTES)


def _trim_text_file(path: Path, maximum_bytes: int) -> None:
    """Retain a bounded, line-aligned tail of an append-only operator log."""

    if path.stat().st_size <= maximum_bytes:
        return
    with path.open("rb") as stream:
        stream.seek(-maximum_bytes, os.SEEK_END)
        tail = stream.read()
    newline = tail.find(b"\n")
    if newline >= 0:
        tail = tail[newline + 1 :]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.trim")
    temporary.write_bytes(tail)
    temporary.replace(path)


def transition_status(
    paths: JobPaths,
    state: str,
    *,
    allow_same: bool = True,
    **fields: object,
) -> dict[str, object]:
    """Atomically advance local job status under the monotonic state machine."""

    if state not in JOB_STATES:
        raise ValueError(f"unknown Colab job state: {state}")
    previous = load_status(paths) if paths.status.is_file() else {}
    old_state = previous.get("state")
    if isinstance(old_state, str) and old_state in JOB_STATES:
        if (
            state != old_state
            and state not in {"failed", "cancelled"}
            and STATE_ORDER[state] < STATE_ORDER[old_state]
        ):
            raise ColabBatchError(
                f"job state regression forbidden: {old_state} -> {state}"
            )
        if state == old_state and not allow_same:
            raise ColabBatchError(f"job state already {state}")
        if old_state in TERMINAL_STATES and state != old_state:
            raise ColabBatchError(
                f"terminal job cannot transition: {old_state} -> {state}"
            )
    status = {
        **previous,
        "schema_version": JOB_SCHEMA_VERSION,
        "state": state,
        "updated_at": time.time(),
        **fields,
    }
    _atomic_write_json(paths.status, status)
    if state != old_state:
        append_event(paths, "state", old_state=old_state, state=state)
    return status


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ColabBatchError(f"cannot hash file {path}: {error}") from error
    return digest.hexdigest()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ColabBatchError(f"path is outside project root: {path}") from error


def normalize_trainer_args(
    arguments: Sequence[str],
    *,
    remote_manifest: str,
    remote_run_dir: str,
) -> tuple[str, ...]:
    """Pin remote paths and require the optimized CUDA pixel boundary."""

    result = [str(item) for item in arguments]

    def replace_option(name: str, value: str) -> None:
        replacement: list[str] = []
        index = 0
        while index < len(result):
            token = result[index]
            if token == name:
                if index + 1 >= len(result):
                    raise ValueError(f"trainer option {name} has no value")
                index += 2
                continue
            if token.startswith(f"{name}="):
                index += 1
                continue
            replacement.append(token)
            index += 1
        result[:] = [*replacement, name, value]

    def option_value(name: str) -> str | None:
        values: list[str] = []
        for index, token in enumerate(result):
            if token == name:
                if index + 1 >= len(result):
                    raise ValueError(f"trainer option {name} has no value")
                values.append(result[index + 1])
            elif token.startswith(f"{name}="):
                values.append(token.partition("=")[2])
        if len(values) > 1:
            raise ValueError(f"trainer option {name} must not be repeated")
        return values[0] if values else None

    replace_option("--manifest", remote_manifest)
    replace_option("--run-dir", remote_run_dir)
    device = option_value("--device")
    if device is not None:
        if device == "cpu":
            raise ValueError("Colab pixel jobs require --device auto or --device cuda")
        if device not in {"auto", "cuda"}:
            raise ValueError(f"unsupported Colab trainer device: {device}")
    else:
        result.extend(("--device", "auto"))
    boundary = option_value("--native-pixel-boundary")
    if boundary and boundary != "fast":
        raise ValueError("Colab pixel jobs require --native-pixel-boundary fast")
    if not boundary:
        result.extend(("--native-pixel-boundary", "fast"))
    return tuple(result)


def create_job(
    *,
    project_root: Path = PROJECT_ROOT,
    local_run_dir: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    trainer_args: Sequence[str] = (),
    gpu_tiers: Sequence[str] = DEFAULT_GPU_TIERS,
    job_root: Path = DEFAULT_JOB_ROOT,
    job_id: str | None = None,
    job_prefix: str = "pixel-dqn",
    session_prefix: str = "dodge-ng",
    poll_seconds: float = 15.0,
    heartbeat_timeout_seconds: float = 180.0,
    recovery_sync_seconds: float = 300.0,
    include_frame_store: bool = False,
    resume_from: Path | None = None,
) -> tuple[JobPaths, dict[str, object]]:
    """Create and persist a planned job before any remote allocation."""

    if (
        poll_seconds <= 0
        or heartbeat_timeout_seconds <= 0
        or recovery_sync_seconds <= 0
    ):
        raise ValueError("poll and heartbeat timeouts must be positive")
    tiers = parse_gpu_tiers(gpu_tiers)
    project_root = Path(project_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    manifest_rel = _relative_to_root(manifest_path, project_root)
    resolved_job_id = job_id or _new_job_id(job_prefix)
    paths = job_paths(Path(job_root), resolved_job_id)
    if paths.root.exists():
        raise ValueError(f"Colab job already exists: {paths.root}")
    remote_root = REMOTE_JOB_ROOT / resolved_job_id
    remote_source = remote_root / "source"
    remote_run = remote_root / "run"
    remote_manifest = remote_source / manifest_rel
    normalized_args = normalize_trainer_args(
        trainer_args,
        remote_manifest=str(remote_manifest),
        remote_run_dir=str(remote_run),
    )
    if resume_from is not None and "--resume" not in normalized_args:
        normalized_args = (*normalized_args, "--resume")
    if resume_from is None and "--resume" in normalized_args:
        raise ValueError("--resume requires --resume-from")
    paths.root.mkdir(parents=True, exist_ok=False)
    paths.logs.mkdir()
    session_name = (
        f"{_safe_slug(session_prefix)}-{_safe_slug(resolved_job_id)}"[:54]
        + f"-{secrets.token_hex(4)}"
    )
    ownership_nonce = secrets.token_hex(16)
    payload: dict[str, object] = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": resolved_job_id,
        "created_at": time.time(),
        "project_root": str(project_root),
        "local_run_dir": str(Path(local_run_dir).resolve()),
        "manifest_path": str(manifest_path),
        "manifest_relative_path": manifest_rel,
        "manifest_sha256": manifest.sha256,
        "trainer": {
            "module": "dodge.ng.pixel_dqn",
            "argv": list(normalized_args),
        },
        "gpu_policy": {
            "requested_tiers": list(tiers),
            "selected_tier": None,
            "attempts": [],
        },
        "session": {
            "prefix": _safe_slug(session_prefix),
            "name": session_name,
            "ownership_nonce": ownership_nonce,
            "remote_root": str(remote_root),
        },
        "observability": {
            "poll_seconds": poll_seconds,
            "heartbeat_timeout_seconds": heartbeat_timeout_seconds,
            "recovery_sync_seconds": recovery_sync_seconds,
        },
        "artifacts": {
            "include_frame_store": include_frame_store,
            "checkpoint_capability": (
                "learner-and-replay-resume; native lanes restart; "
                "immutable compressed replay snapshots included"
            ),
            "default_allowlist": list(DEFAULT_ARTIFACT_FILES),
            "required": list(REQUIRED_ARTIFACTS),
        },
        "resume": {
            "source_directory": (
                None if resume_from is None else str(Path(resume_from).resolve())
            ),
            "bundle": None,
            "bundle_sha256": None,
        },
        "source": {},
        "job_identity": {},
        "job_identity_sha256": None,
    }
    _atomic_write_json(paths.manifest, payload)
    transition_status(
        paths,
        "planned",
        job_id=resolved_job_id,
        manifest_sha256=manifest.sha256,
        step=0,
        total_steps=_trainer_total_steps(normalized_args),
    )
    append_event(paths, "planned", gpu_tiers=list(tiers))
    return paths, payload


def _trainer_total_steps(arguments: Sequence[str]) -> int | None:
    for index, item in enumerate(arguments):
        if item == "--total-steps" and index + 1 < len(arguments):
            try:
                return int(arguments[index + 1])
            except ValueError:
                return None
        if item.startswith("--total-steps="):
            try:
                return int(item.partition("=")[2])
            except ValueError:
                return None
    return 200_000


def _source_file_paths(project_root: Path) -> list[Path]:
    roots = (
        project_root / "src" / "dodge",
        project_root / "native",
        project_root / "context" / "kits" / "dodge-ng",
    )
    direct = (
        project_root / "pyproject.toml",
        project_root / "uv.lock",
    )
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            raise ColabBatchError(f"required source path is missing: {root}")
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative_parts = path.relative_to(project_root).parts
            if _EXCLUDED_SOURCE_PARTS.intersection(relative_parts):
                continue
            files.append(path)
    for path in direct:
        if path.is_file():
            files.append(path)
    return sorted(set(files))


def create_source_bundle(
    project_root: Path,
    output_path: Path,
) -> tuple[list[dict[str, object]], str]:
    """Package only reproducible source inputs and return its inventory/hash."""

    project_root = Path(project_root).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = _source_file_paths(project_root)
    inventory: list[dict[str, object]] = []
    with tarfile.open(output_path, "w:gz") as archive:
        for path in files:
            relative = path.relative_to(project_root).as_posix()
            archive.add(path, arcname=f"source/{relative}", recursive=False)
            inventory.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return inventory, sha256_file(output_path)


def build_native_wheel(
    project_root: Path,
    output_directory: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> Path:
    """Build the abi3 native extension once for the remote Python runtime."""

    output_directory.mkdir(parents=True, exist_ok=True)
    command = (
        "maturin",
        "build",
        "--release",
        "--locked",
        "--manifest-path",
        str(Path(project_root) / "native" / "crates" / "dodge-python" / "Cargo.toml"),
        "--out",
        str(output_directory),
    )
    try:
        result = (runner or subprocess.run)(
            list(command),
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise ColabBatchError(
            "maturin is required to build the Colab native wheel; "
            "run this command inside devenv"
        ) from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "maturin failed").strip()
        raise ColabBatchError(f"native wheel build failed: {detail[-2000:]}")
    wheels = sorted(output_directory.glob("dodge_native-*.whl"))
    if not wheels:
        raise ColabBatchError(
            f"maturin produced no dodge_native wheel in {output_directory}"
        )
    return wheels[-1]


def prepare_job(
    paths: JobPaths,
    *,
    build_wheel: Callable[[Path, Path], Path] = build_native_wheel,
) -> dict[str, object]:
    """Create source/wheel payloads and persist their identities before submit."""

    job = load_job(paths)
    project_root = Path(str(job["project_root"]))
    source_inventory, source_sha = create_source_bundle(
        project_root,
        paths.source_bundle,
    )
    wheel = build_wheel(project_root, paths.wheel_directory)
    wheel_record = {
        "path": wheel.name,
        "bytes": wheel.stat().st_size,
        "sha256": sha256_file(wheel),
    }
    resume = job.get("resume")
    if not isinstance(resume, Mapping):
        raise ColabBatchError("job resume metadata is invalid")
    resume_record = dict(resume)
    if resume.get("source_directory") is not None:
        source_run = Path(str(resume["source_directory"]))
        resume_files = create_resume_bundle(source_run, paths.resume_bundle)
        resume_record.update(
            bundle=paths.resume_bundle.name,
            bundle_sha256=sha256_file(paths.resume_bundle),
            bundle_bytes=paths.resume_bundle.stat().st_size,
            files=resume_files,
        )
    job["resume"] = resume_record
    job["source"] = {
        "bundle": paths.source_bundle.name,
        "bundle_bytes": paths.source_bundle.stat().st_size,
        "bundle_sha256": source_sha,
        "files": source_inventory,
        "wheel": wheel_record,
    }
    identity = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job["job_id"],
        "manifest_sha256": job["manifest_sha256"],
        "trainer": job["trainer"],
        "requested_gpu_tiers": job["gpu_policy"]["requested_tiers"],
        "session": {
            "name": job["session"]["name"],
            "ownership_nonce": job["session"]["ownership_nonce"],
            "remote_root": job["session"]["remote_root"],
        },
        "source_bundle_sha256": source_sha,
        "native_wheel_sha256": wheel_record["sha256"],
        "resume_bundle_sha256": resume_record.get("bundle_sha256"),
    }
    job["job_identity"] = identity
    job["job_identity_sha256"] = _canonical_sha256(identity)
    _atomic_write_json(paths.manifest, job)
    append_event(
        paths,
        "prepared",
        source_sha256=source_sha,
        wheel_sha256=wheel_record["sha256"],
    )
    return job


def create_resume_bundle(
    run_directory: Path,
    output_path: Path,
) -> list[dict[str, object]]:
    """Package exactly the checkpoint state required by pixel DQN resume."""
    import torch

    run_directory = Path(run_directory).resolve()
    checkpoint = run_directory / "checkpoint-latest.pt"
    metrics = run_directory / "metrics.jsonl"
    if not checkpoint.is_file() or not metrics.is_file():
        raise ColabBatchError(
            "resume source lacks checkpoint-latest.pt or metrics.jsonl"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    snapshot = payload.get("replay_snapshot")
    name = snapshot.get("path") if isinstance(snapshot, Mapping) else None
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.startswith("replay-")
        or not name.endswith(".u8.gz")
    ):
        raise ColabBatchError("resume checkpoint lacks valid immutable replay snapshot")
    files = [checkpoint, metrics, run_directory / name]
    if not files[-1].is_file():
        raise ColabBatchError("resume checkpoint replay snapshot is missing")
    records = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.name, recursive=False)
    return records


def extract_recovery_archive(
    archive_path: Path,
    destination: Path,
    *,
    expected_job_identity_sha256: str,
) -> dict[str, object]:
    """Verify and atomically install one resumable off-runtime checkpoint."""
    archive_path = Path(archive_path).resolve()
    destination = Path(destination).resolve()
    staging = destination.parent / f".{destination.name}.incoming-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with tarfile.open(archive_path, "r:") as archive:
            members = archive.getmembers()
            if any(not _safe_archive_name(item.name) for item in members):
                raise ColabBatchError("recovery archive contains unsafe path")
            if any(not item.isfile() for item in members):
                raise ColabBatchError("recovery archive contains non-file entry")
            archive.extractall(staging, filter="data")
        inventory = _read_json(staging / "recovery-manifest.json")
        if inventory.get("job_identity_sha256") != expected_job_identity_sha256:
            raise ColabBatchError("recovery archive identity mismatch")
        files = inventory.get("files")
        if not isinstance(files, list):
            raise ColabBatchError("recovery file inventory is invalid")
        expected_names = {"checkpoint-latest.pt", "metrics.jsonl"}
        snapshot_names = set()
        for item in files:
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                raise ColabBatchError("recovery file entry is invalid")
            name = item["path"]
            if not _safe_archive_name(name):
                raise ColabBatchError("recovery file path is invalid")
            path = staging / name
            if not path.is_file() or sha256_file(path) != item.get("sha256"):
                raise ColabBatchError(f"recovery file hash failed: {name}")
            expected_names.discard(name)
            if name.startswith("replay-") and name.endswith(".u8.gz"):
                snapshot_names.add(name)
        member_names = {item.name for item in members}
        declared_names = {str(item["path"]) for item in files}
        if expected_names or len(snapshot_names) != 1:
            raise ColabBatchError("recovery archive lacks resumable files")
        if member_names != declared_names | {"recovery-manifest.json"}:
            raise ColabBatchError("recovery archive membership mismatch")
        backup = destination.parent / f".{destination.name}.previous"
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            destination.replace(backup)
        staging.replace(destination)
        shutil.rmtree(backup, ignore_errors=True)
        return inventory
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def artifact_inventory(
    run_directory: Path,
    *,
    include_frame_store: bool = False,
) -> list[dict[str, object]]:
    """Hash the allowlisted result files without traversing arbitrary history."""

    root = Path(run_directory).resolve()
    records: list[dict[str, object]] = []
    candidates = [root / relative for relative in DEFAULT_ARTIFACT_FILES]
    candidates.extend(root.glob("replay-*.u8.gz"))
    replay_directory = root / "dashboard" / "replays"
    if replay_directory.is_dir():
        candidates.extend(replay_directory.glob("*.json"))
        candidates.extend(replay_directory.glob("*.u8"))
    if include_frame_store:
        candidates.append(root / ".pixel-frames.u8")
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        relative = _relative_to_root(path, root)
        is_frame_store = relative == ".pixel-frames.u8" or relative.startswith(
            "replay-"
        )
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "resume_required": is_frame_store,
            }
        )
    return sorted(records, key=lambda item: str(item["path"]))


def create_artifact_archive(
    run_directory: Path,
    archive_path: Path,
    *,
    job_id: str,
    manifest_sha256: str,
    job_identity_sha256: str,
    source_bundle_sha256: str,
    native_wheel_sha256: str,
    include_frame_store: bool = False,
) -> dict[str, object]:
    """Create a verified, allowlisted result archive."""

    root = Path(run_directory).resolve()
    records = artifact_inventory(root, include_frame_store=include_frame_store)
    if not records:
        raise ColabBatchError(f"run has no retrievable artifacts: {root}")
    record_paths = {str(item["path"]): item for item in records}
    inventory: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "job_id": job_id,
        "manifest_sha256": manifest_sha256,
        "job_identity_sha256": job_identity_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "native_wheel_sha256": native_wheel_sha256,
        "run_directory": str(root),
        "files": records,
        "required": list(REQUIRED_ARTIFACTS),
        "complete": all(item in record_paths for item in REQUIRED_ARTIFACTS),
        "frame_store_included": include_frame_store
        and ".pixel-frames.u8" in record_paths,
        "immutable_replay_snapshots_included": any(
            name.startswith("replay-") for name in record_paths
        ),
        "created_at": time.time(),
    }
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    encoded_manifest = json.dumps(inventory, indent=2, sort_keys=True).encode()
    with tarfile.open(archive_path, "w:gz") as archive:
        for item in records:
            relative = str(item["path"])
            archive.add(root / relative, arcname=relative, recursive=False)
        info = tarfile.TarInfo("artifact-manifest.json")
        info.size = len(encoded_manifest)
        info.mtime = int(time.time())
        archive.addfile(info, io.BytesIO(encoded_manifest))
    inventory["archive"] = {
        "path": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
    }
    return inventory


def _safe_archive_name(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts and name != ""


def extract_artifact_archive(
    archive_path: Path,
    local_run_directory: Path,
    *,
    expected_job_id: str,
    expected_manifest_sha256: str,
    expected_job_identity_sha256: str,
    expected_source_bundle_sha256: str,
    expected_native_wheel_sha256: str,
    expected_archive_sha256: str | None = None,
    manifest_output: Path | None = None,
) -> dict[str, object]:
    """Verify archive identity/file hashes before installing returned artifacts."""

    archive_path = Path(archive_path).resolve()
    if (
        expected_archive_sha256 is not None
        and sha256_file(archive_path) != expected_archive_sha256
    ):
        raise ColabBatchError("downloaded artifact archive hash does not match remote")
    staging = archive_path.parent / f".artifact-extract-{os.getpid()}"
    install: Path | None = None
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if any(not _safe_archive_name(member.name) for member in members):
                raise ColabBatchError("artifact archive contains an unsafe path")
            if any(not (member.isfile() or member.isdir()) for member in members):
                raise ColabBatchError("artifact archive contains a non-regular entry")
            archive.extractall(staging, filter="data")
        inventory = _read_json(staging / "artifact-manifest.json")
        if inventory.get("job_id") != expected_job_id:
            raise ColabBatchError("artifact archive job id does not match local job")
        if inventory.get("manifest_sha256") != expected_manifest_sha256:
            raise ColabBatchError("artifact archive manifest hash does not match job")
        if inventory.get("job_identity_sha256") != expected_job_identity_sha256:
            raise ColabBatchError(
                "artifact archive payload identity does not match job"
            )
        if inventory.get("source_bundle_sha256") != expected_source_bundle_sha256:
            raise ColabBatchError("artifact archive source identity does not match job")
        if inventory.get("native_wheel_sha256") != expected_native_wheel_sha256:
            raise ColabBatchError("artifact archive wheel identity does not match job")
        files = inventory.get("files")
        if not isinstance(files, list):
            raise ColabBatchError("artifact archive file inventory is invalid")
        for item in files:
            if not isinstance(item, Mapping):
                raise ColabBatchError("artifact archive file entry is invalid")
            relative = item.get("path")
            expected_sha = item.get("sha256")
            if not isinstance(relative, str) or not _safe_archive_name(relative):
                raise ColabBatchError("artifact archive file path is invalid")
            if not isinstance(expected_sha, str):
                raise ColabBatchError("artifact archive file hash is invalid")
            source = staging / relative
            if not source.is_file() or sha256_file(source) != expected_sha:
                raise ColabBatchError(f"artifact hash verification failed: {relative}")
        required = inventory.get("required")
        if required != list(REQUIRED_ARTIFACTS) or any(
            not isinstance(item, str)
            or not _safe_archive_name(item)
            or not (staging / item).is_file()
            for item in required
        ):
            raise ColabBatchError("artifact archive is incomplete")
        declared_names = [str(item["path"]) for item in files]
        member_names = [member.name for member in members]
        if len(member_names) != len(set(member_names)):
            raise ColabBatchError("artifact archive contains duplicate members")
        if set(member_names) != {*declared_names, "artifact-manifest.json"}:
            raise ColabBatchError(
                "artifact archive membership does not match inventory"
            )
        output = Path(local_run_directory).resolve()
        if output.exists():
            raise ColabBatchError(f"artifact destination already exists: {output}")
        install = output.parent / f".{output.name}.incoming-{expected_job_id}"
        if install.exists():
            shutil.rmtree(install)
        install.mkdir(parents=True)
        for item in files:
            relative = str(item["path"])
            destination = install / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _relative_to_root(destination, install)
            shutil.copy2(staging / relative, destination)
        # Commit the receipt with the directory, not afterward: recovery must
        # distinguish our verified result from an unrelated existing run.
        _atomic_write_json(install / ".colab-artifact-receipt.json", inventory)
        install.replace(output)
        if manifest_output is not None:
            _atomic_write_json(Path(manifest_output), inventory)
        return inventory
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if install is not None:
            shutil.rmtree(install, ignore_errors=True)


def _command_log(paths: JobPaths, operation: str, result: CommandResult) -> None:
    paths.logs.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_slug(operation)
    log_path = paths.logs / f"{safe_name}.log"
    mode = "w" if operation == "poll" else "a"
    with log_path.open(mode, encoding="utf-8") as stream:
        stream.write(f"$ {' '.join(result.command)}\n")
        stream.write(f"exit={result.returncode}\n")
        if result.stdout:
            stream.write("[stdout]\n")
            stream.write(result.stdout)
            if not result.stdout.endswith("\n"):
                stream.write("\n")
        if result.stderr:
            stream.write("[stderr]\n")
            stream.write(result.stderr)
            if not result.stderr.endswith("\n"):
                stream.write("\n")
        stream.write("\n")
    _trim_text_file(log_path, MAX_COMMAND_LOG_BYTES)


def _json_from_output(output: str) -> dict[str, object] | None:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _recovery_resume_signature(value: object) -> tuple[str, str, str] | None:
    """Identify the immutable checkpoint and replay pair in a recovery manifest."""

    if not isinstance(value, Mapping):
        return None
    files = value.get("files")
    if not isinstance(files, list):
        return None
    checkpoint_sha: str | None = None
    snapshots: list[tuple[str, str]] = []
    for item in files:
        if not isinstance(item, Mapping):
            return None
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            return None
        if path == "checkpoint-latest.pt":
            if checkpoint_sha is not None:
                return None
            checkpoint_sha = digest
        elif path.startswith("replay-") and path.endswith(".u8.gz"):
            snapshots.append((path, digest))
    if checkpoint_sha is None or len(snapshots) != 1:
        return None
    snapshot_name, snapshot_sha = snapshots[0]
    return checkpoint_sha, snapshot_name, snapshot_sha


def _validated_remote_status(
    value: Mapping[str, object],
    *,
    expected_job_id: str,
    expected_manifest_sha256: str,
    expected_job_identity_sha256: str,
    expected_ownership_nonce: str,
) -> dict[str, object]:
    """Validate remote identity and heartbeat fields before trusting progress."""

    state = value.get("state")
    if state not in REMOTE_STATES:
        raise ColabBatchError(f"remote worker reported an invalid state: {state!r}")
    # The poll helper emits an identity-free `starting` record until bootstrap
    # creates its first status. No other identity omission is accepted.
    if value.get("job_id") is None and state == "starting":
        return dict(value)
    if value.get("job_id") != expected_job_id:
        raise ColabBatchError("remote status job id does not match local job")
    if value.get("manifest_sha256") != expected_manifest_sha256:
        raise ColabBatchError("remote status manifest hash does not match local job")
    if value.get("job_identity_sha256") != expected_job_identity_sha256:
        raise ColabBatchError("remote status payload identity does not match local job")
    if value.get("ownership_nonce") != expected_ownership_nonce:
        raise ColabBatchError("remote status ownership nonce does not match local job")
    heartbeat = value.get("heartbeat_at")
    if not isinstance(heartbeat, (int, float)) or not math.isfinite(heartbeat):
        raise ColabBatchError("remote status heartbeat is missing or non-finite")
    if heartbeat > time.time() + 300.0:
        raise ColabBatchError("remote status heartbeat is implausibly in the future")
    worker_pid = value.get("worker_pid")
    if state in {"running", "completed", "stopped"} and (
        not isinstance(worker_pid, int) or worker_pid <= 0
    ):
        raise ColabBatchError("remote status worker pid is invalid")
    for field in ("step", "native_steps", "total_steps"):
        field_value = value.get(field)
        if field_value is not None and (
            not isinstance(field_value, int) or field_value < 0
        ):
            raise ColabBatchError(f"remote status {field} is invalid")
    return dict(value)


def _remote_script(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")


def _write_remote_scripts(paths: JobPaths, job: Mapping[str, object]) -> None:
    remote = job["session"]
    if not isinstance(remote, Mapping):
        raise ColabBatchError("job session metadata is invalid")
    remote_root = str(remote["remote_root"])
    source_root = f"{remote_root}/source"
    remote_run = f"{remote_root}/run"
    source = job.get("source")
    if not isinstance(source, Mapping):
        raise ColabBatchError("job source bundle is not prepared")
    bundle_name = str(source["bundle"])
    wheel = source.get("wheel")
    if not isinstance(wheel, Mapping):
        raise ColabBatchError("job native wheel is not prepared")
    wheel_name = str(wheel["path"])
    resume = job.get("resume")
    if not isinstance(resume, Mapping):
        raise ColabBatchError("job resume metadata is invalid")
    resume_name = resume.get("bundle")
    resume_sha = resume.get("bundle_sha256")
    trainer = job.get("trainer")
    if not isinstance(trainer, Mapping) or not isinstance(trainer.get("argv"), list):
        raise ColabBatchError("job trainer metadata is invalid")
    trainer_json = json.dumps(trainer["argv"])
    artifacts = job.get("artifacts")
    include_frame_store = bool(
        isinstance(artifacts, Mapping) and artifacts.get("include_frame_store")
    )
    job_id = str(job["job_id"])
    manifest_sha = str(job["manifest_sha256"])
    job_identity_sha = str(job["job_identity_sha256"])
    source_sha = str(source["bundle_sha256"])
    wheel_sha = str(wheel["sha256"])
    ownership_nonce = str(remote["ownership_nonce"])
    gpu_policy = job.get("gpu_policy")
    selected_tier = (
        str(gpu_policy["selected_tier"])
        if isinstance(gpu_policy, Mapping) and gpu_policy.get("selected_tier")
        else "unknown"
    )
    heartbeat_seconds = (
        float(job["observability"]["poll_seconds"])
        if isinstance(job.get("observability"), Mapping)
        else 15.0
    )
    _remote_script(
        paths.bootstrap,
        f"""
        import json
        import hashlib
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tarfile
        import time

        remote_root = Path({remote_root!r})
        bundle = Path('/content/{bundle_name}')
        wheel = Path('/content/{wheel_name}')
        resume_bundle = {f"Path('/content/{resume_name}')" if resume_name else "None"}
        source_root = remote_root / 'source'
        status_path = remote_root / 'remote-status.json'
        remote_root.mkdir(parents=True, exist_ok=True)

        def sha256(path):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest()

        def write_status(value):
            status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = status_path.with_name('.remote-status.tmp')
            temporary.write_text(json.dumps(value, sort_keys=True) + '\\n')
            temporary.replace(status_path)

        def safe_extract(archive, destination):
            destination = destination.resolve()
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if destination not in target.parents and target != destination:
                    raise RuntimeError('source archive contains an unsafe path')
                if not (member.isfile() or member.isdir()):
                    raise RuntimeError('source archive contains a non-regular entry')
            archive.extractall(destination, filter='data')

        try:
            if sha256(bundle) != {source_sha!r}:
                raise RuntimeError('source bundle hash does not match job identity')
            if sha256(wheel) != {wheel_sha!r}:
                raise RuntimeError('native wheel hash does not match job identity')
            if resume_bundle is not None and sha256(resume_bundle) != {resume_sha!r}:
                raise RuntimeError('resume bundle hash does not match job identity')
            with tarfile.open(bundle, 'r:gz') as archive:
                safe_extract(archive, remote_root)
            if resume_bundle is not None:
                (remote_root / 'run').mkdir(parents=True, exist_ok=True)
                with tarfile.open(resume_bundle, 'r:gz') as archive:
                    safe_extract(archive, remote_root / 'run')
            install = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                 '--no-deps', str(wheel)],
                capture_output=True,
                text=True,
                check=False,
            )
            if install.returncode != 0:
                raise RuntimeError(
                    install.stderr[-2000:] or 'native wheel install failed'
                )
            argv = json.loads({trainer_json!r})
            environment = os.environ.copy()
            environment['PYTHONPATH'] = (
                str(source_root / 'src') + os.pathsep
                + environment.get('PYTHONPATH', '')
            )
            command = [
                sys.executable,
                '-m',
                'dodge.ng.colab_worker',
                '--job-id',
                {job_id!r},
                '--project-root',
                str(source_root),
                '--run-dir',
                str(remote_root / 'run'),
                '--status-path',
                str(status_path),
                '--manifest-sha256',
                {manifest_sha!r},
                '--job-identity-sha256',
                {job_identity_sha!r},
                '--source-bundle-sha256',
                {source_sha!r},
                '--native-wheel-sha256',
                {wheel_sha!r},
                '--ownership-nonce',
                {ownership_nonce!r},
                '--gpu',
                {selected_tier!r},
                '--heartbeat-seconds',
                str({heartbeat_seconds!r}),
                '--include-frame-store' if {include_frame_store!r}
                else '--no-frame-store',
                '--trainer-args-json',
                json.dumps(argv),
            ]
            log_path = remote_root / 'worker.log'
            log = log_path.open('ab', buffering=0)
            write_status({{
                'schema_version': 1, 'job_id': {job_id!r}, 'state': 'starting',
                'worker_pid': None, 'heartbeat_at': time.time(),
                'manifest_sha256': {manifest_sha!r},
                'job_identity_sha256': {job_identity_sha!r},
                'ownership_nonce': {ownership_nonce!r},
                'source_root': str(source_root),
                'run_directory': str(remote_root / 'run'),
            }})
            process = subprocess.Popen(
                command,
                cwd=source_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(json.dumps({{'started': True, 'pid': process.pid}}))
        except Exception as error:
            write_status({{
                'schema_version': 1, 'job_id': {job_id!r}, 'state': 'failed',
                'heartbeat_at': time.time(), 'manifest_sha256': {manifest_sha!r},
                'job_identity_sha256': {job_identity_sha!r},
                'ownership_nonce': {ownership_nonce!r},
                'last_error': f"{{type(error).__name__}}: {{error}}",
            }})
            raise
        """,
    )
    _remote_script(
        paths.poller,
        f"""
        import json
        from pathlib import Path
        path = Path({remote_root!r}) / 'remote-status.json'
        try:
            print(json.dumps(json.loads(path.read_text())))
        except Exception as error:
            print(json.dumps({{'state': 'starting', 'last_error': str(error)}}))
        """,
    )
    _remote_script(
        paths.stopper,
        f"""
        import json
        import os
        import tempfile
        import time
        from pathlib import Path
        run = Path({remote_run!r})
        target = run / 'dashboard' / 'control.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.control-', dir=target.parent)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(
                {{'version': 1, 'id': str(time.time_ns()), 'command': 'stop'}},
                stream,
            )
            stream.write('\\n')
        os.replace(name, target)
        print(json.dumps({{'stop_requested': True, 'path': str(target)}}))
        """,
    )
    _remote_script(
        paths.packer,
        f"""
        import json
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path({source_root!r}) / 'src'))
        from dodge.ng.colab_worker import finalize_artifacts
        result = finalize_artifacts(
            Path({remote_run!r}), Path({remote_root!r}) / 'artifacts-controller.tar.gz',
            job_id={job_id!r}, manifest_sha256={manifest_sha!r},
            job_identity_sha256={job_identity_sha!r},
            source_bundle_sha256={source_sha!r},
            native_wheel_sha256={wheel_sha!r},
            include_frame_store={include_frame_store!r},
        )
        print(json.dumps(result, sort_keys=True))
        """,
    )
    _remote_script(
        paths.recovery_exporter,
        f"""
        import json
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path({source_root!r}) / 'src'))
        from dodge.ng.colab_worker import create_recovery_archive
        result = create_recovery_archive(
            Path({remote_run!r}), Path({remote_root!r}) / 'recovery.tar',
            job_identity_sha256={job_identity_sha!r},
        )
        print(json.dumps(result, sort_keys=True))
        """,
    )


class ColabBatchController:
    """Provision, run, observe, stop, and retrieve one prepared job."""

    def __init__(
        self,
        paths: JobPaths,
        *,
        cli: ColabCLI | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.paths = paths
        self.job = load_job(paths)
        self.cli = cli or ColabCLI()
        self._sleep = sleep
        self.session_name: str | None = _session_name(self.job)
        self.stop_requested = False
        self._owns_session = bool(load_status(paths).get("allocation_started"))

    def run(self, *, keep_session: bool = False) -> dict[str, object]:
        """Run a prepared job to terminal state and retrieve its artifacts."""

        with (
            _controller_lease(self.paths),
            _controller_stop_signals(lambda: self.request_stop(send_remote=False)),
        ):
            current = load_status(self.paths)
            transition_status(
                self.paths,
                str(current["state"]),
                controller_pid=os.getpid(),
                controller_started_at=time.time(),
            )
            return self._run_locked(keep_session=keep_session)

    def _run_locked(self, *, keep_session: bool) -> dict[str, object]:
        """Execute the lifecycle while holding the per-job controller lease."""

        try:
            if self._cancel_before_launch():
                return load_status(self.paths)
            self._auth_preflight()
            self._provision()
            if self._cancel_before_launch():
                return load_status(self.paths)
            self._upload()
            if self._cancel_before_launch():
                return load_status(self.paths)
            self._launch()
            remote_status = self._watch()
            return self._finish_remote(remote_status, keep_session=keep_session)
        except KeyboardInterrupt:
            append_event(self.paths, "local_interrupt")
            try:
                self.request_stop()
                remote_status = self._watch()
                return self._finish_remote(remote_status, keep_session=False)
            except (ColabBatchError, OSError, ValueError) as error:
                return self._record_failure(error)
        except (ColabBatchError, OSError, ValueError) as error:
            self._record_failure(error)
            raise

    def _record_failure(self, error: Exception) -> dict[str, object]:
        detail = f"{type(error).__name__}: {error}"
        append_event(self.paths, "controller_error", error=detail)
        current = load_status(self.paths)
        if current.get("state") in TERMINAL_STATES:
            return current
        if self._owns_session and current.get("state") in {"running", "stopping"}:
            # The remote runtime may hold the only checkpoint. A network failure
            # is not authorization to destroy it. `watch` can re-adopt this job.
            append_event(self.paths, "session_retained", reason=detail)
            return transition_status(
                self.paths,
                "stopping",
                recovery_required=True,
                error=detail,
                recovery_command=f"just dodge-ng-colab watch {self.paths.root}",
            )
        if self._owns_session:
            self._download_remote_diagnostics()
            self._cleanup(keep_session=False)
        return transition_status(self.paths, "failed", error=detail)

    def _cancel_before_launch(self) -> bool:
        if not self.paths.stop_request.is_file():
            return False
        cleaned = self._cleanup(keep_session=False)
        transition_status(
            self.paths,
            "cancelled" if cleaned else "failed",
            reason="stopped before worker launch",
            recovery_required=not cleaned,
        )
        return True

    def recover(self, *, keep_session: bool = False) -> dict[str, object]:
        """Adopt a launched named session after its local supervisor exits."""

        with (
            _controller_lease(self.paths),
            _controller_stop_signals(lambda: self.request_stop(send_remote=False)),
        ):
            current = load_status(self.paths)
            if current.get("state") in TERMINAL_STATES:
                return current
            if current.get("state") not in {"running", "stopping"}:
                raise ColabBatchError(
                    "automatic recovery requires a launched running/stopping job"
                )
            transition_status(
                self.paths,
                str(current["state"]),
                controller_pid=os.getpid(),
                controller_recovered_at=time.time(),
            )
            self._auth_preflight()
            if not self.session_name:
                raise ColabBatchError("cannot recover without a persisted session name")
            status = self.cli.call(
                ["status", "-s", self.session_name],
                timeout=30.0,
            )
            _command_log(self.paths, "recover-session-status", status)
            if not _session_exists(status):
                raise ColabBatchError("cannot recover because Colab session is absent")
            try:
                remote = self._watch()
                return self._finish_remote(remote, keep_session=keep_session)
            except (ColabBatchError, OSError, ValueError) as error:
                self._record_failure(error)
                raise

    def request_stop(self, *, send_remote: bool = True) -> None:
        """Persist a stop request; only the leased supervisor changes job state."""
        current = load_status(self.paths)
        if current.get("state") in TERMINAL_STATES:
            return
        self.stop_requested = True
        _atomic_write_json(self.paths.stop_request, {"requested_at": time.time()})
        remote = current.get("remote")
        if send_remote and isinstance(remote, Mapping):
            validated = _validated_remote_status(
                remote,
                expected_job_id=str(self.job["job_id"]),
                expected_manifest_sha256=str(self.job["manifest_sha256"]),
                expected_job_identity_sha256=str(self.job["job_identity_sha256"]),
                expected_ownership_nonce=str(self.job["session"]["ownership_nonce"]),
            )
            if validated.get("state") not in REMOTE_TERMINAL_STATES:
                self._send_remote_stop()

    def _send_stop(self) -> None:
        self.stop_requested = True
        transition_status(self.paths, "stopping", reason="stop requested")
        self._send_remote_stop()

    def _send_remote_stop(self) -> None:
        result = self.cli.call(
            ["exec", "-s", self.session_name, "-f", str(self.paths.stopper)],
            timeout=120.0,
        )
        _command_log(self.paths, "request-stop", result)
        append_event(self.paths, "stop_requested", ok=result.ok)
        if not result.ok:
            detail = (result.stderr or result.stdout).strip()[-1000:]
            raise ColabBatchError(f"remote stop request failed: {detail}")

    def _finish_remote(
        self,
        remote_status: Mapping[str, object],
        *,
        keep_session: bool,
    ) -> dict[str, object]:
        self._download_remote_diagnostics()
        current = load_status(self.paths)
        if current.get("state") != "stopping":
            transition_status(
                self.paths,
                "stopping",
                remote_state=remote_status.get("state"),
            )
        if (
            remote_status.get("state") in {"completed", "stopped", "cancelled"}
            or int(remote_status.get("step", 0)) > 0
        ):
            self._retrieve(remote_status)
        else:
            append_event(
                self.paths,
                "retrieve_skipped",
                reason="remote job failed before completion",
            )
        final_state = self._local_terminal_state(remote_status)
        if not self._cleanup(keep_session=keep_session):
            return transition_status(
                self.paths,
                "stopping",
                recovery_required=True,
                error="artifacts verified but session cleanup failed; retry watch",
            )
        return transition_status(
            self.paths,
            final_state,
            remote_state=remote_status.get("state"),
            recovery_required=False,
            error=None,
        )

    def _auth_preflight(self) -> None:
        result = self.cli.call(["whoami"], timeout=30.0)
        _command_log(self.paths, "auth-preflight", result)
        if not result.ok:
            raise ColabBatchError(
                "Colab authentication preflight failed: "
                f"{(result.stderr or result.stdout).strip()[-1000:]}"
            )
        append_event(self.paths, "auth_preflight", ok=True)

    def _provision(self) -> None:
        policy = self.job.get("gpu_policy")
        if not isinstance(policy, Mapping) or not isinstance(
            policy.get("requested_tiers"), list
        ):
            raise ColabBatchError("job GPU policy is invalid")
        self.session_name = _session_name(self.job)
        if not self.session_name:
            raise ColabBatchError("job has no persisted Colab session name")
        transition_status(self.paths, "provisioning", session_name=self.session_name)
        collision = self.cli.call(
            ["status", "-s", self.session_name],
            timeout=30.0,
        )
        _command_log(self.paths, "session-collision-check", collision)
        if _session_exists(collision):
            raise ColabBatchError(
                f"refusing to reuse existing Colab session: {self.session_name}"
            )
        if not collision.ok:
            raise ColabBatchError(
                "cannot establish that requested session name is unused"
            )
        attempts: list[dict[str, object]] = []
        for index, tier_value in enumerate(policy["requested_tiers"], start=1):
            tier = str(tier_value)
            append_event(self.paths, "gpu_attempt", tier=tier, attempt=index)
            transition_status(
                self.paths,
                "provisioning",
                allocation_started=True,
                allocation_tier=tier,
                allocation_attempt=index,
            )
            self._owns_session = True
            try:
                result = self.cli.call(
                    ["new", "-s", self.session_name, "--gpu", tier],
                    timeout=300.0,
                )
            except ColabBatchError as error:
                reconciliation = self.cli.call(
                    ["status", "-s", self.session_name], timeout=30.0
                )
                _command_log(
                    self.paths,
                    f"reconcile-{tier}",
                    reconciliation,
                )
                if _session_exists(reconciliation):
                    result = CommandResult(
                        (self.cli.executable, "new", "-s", self.session_name),
                        0,
                        "allocation adopted after ambiguous local timeout",
                        "",
                    )
                    append_event(
                        self.paths,
                        "gpu_allocation_reconciled",
                        tier=tier,
                        error=str(error),
                    )
                else:
                    raise
            _command_log(self.paths, f"new-{tier}", result)
            attempt = {
                "tier": tier,
                "attempt": index,
                "returncode": result.returncode,
                "ok": result.ok,
                "output": (result.stderr or result.stdout).strip()[-2000:],
            }
            attempts.append(attempt)
            policy = {**policy, "attempts": attempts}
            self.job["gpu_policy"] = policy
            _atomic_write_json(self.paths.manifest, self.job)
            if result.ok:
                policy = {**policy, "selected_tier": tier}
                self.job["gpu_policy"] = policy
                _atomic_write_json(self.paths.manifest, self.job)
                transition_status(
                    self.paths,
                    "provisioning",
                    selected_gpu=tier,
                    session_name=self.session_name,
                    gpu_attempt=index,
                )
                append_event(self.paths, "gpu_selected", tier=tier, attempt=index)
                return
            detail = (result.stderr or result.stdout).strip()
            if not _allocation_may_escalate(detail):
                raise ColabBatchError(
                    "GPU allocation failed without a safe escalation signal: "
                    f"{detail[-1000:]}"
                )
            cleanup = self.cli.call(["stop", "-s", self.session_name], timeout=120.0)
            _command_log(self.paths, f"cleanup-{tier}", cleanup)
            append_event(self.paths, "gpu_attempt_cleanup", tier=tier, ok=cleanup.ok)
        raise ColabBatchError("all configured Colab GPU tiers failed to allocate")

    def _upload(self) -> None:
        if not self.session_name:
            raise ColabBatchError("cannot upload before session allocation")
        source = self.job.get("source")
        if not isinstance(source, Mapping):
            raise ColabBatchError("job has no prepared source payload")
        bundle_name = str(source["bundle"])
        wheel = source.get("wheel")
        if not isinstance(wheel, Mapping):
            raise ColabBatchError("job has no prepared native wheel")
        wheel_name = str(wheel["path"])
        uploads = [
            (self.paths.source_bundle, f"/content/{bundle_name}"),
            (self.paths.wheel_directory / wheel_name, f"/content/{wheel_name}"),
        ]
        resume = self.job.get("resume")
        if isinstance(resume, Mapping) and resume.get("bundle"):
            uploads.append((self.paths.resume_bundle, f"/content/{resume['bundle']}"))
        for local, remote in uploads:
            result = self.cli.call(
                ["upload", "-s", self.session_name, str(local), remote],
                timeout=600.0,
            )
            _command_log(self.paths, f"upload-{local.name}", result)
            if not result.ok:
                raise ColabBatchError(
                    f"failed to upload {local.name}: "
                    f"{(result.stderr or result.stdout).strip()[-1000:]}"
                )
        _write_remote_scripts(self.paths, self.job)
        append_event(self.paths, "uploaded", bundle=bundle_name, wheel=wheel_name)

    def _launch(self) -> None:
        if not self.session_name:
            raise ColabBatchError("cannot launch before session allocation")
        result = self.cli.call(
            [
                "exec",
                "-s",
                self.session_name,
                "-f",
                str(self.paths.bootstrap),
                "--timeout",
                "180",
            ],
            timeout=240.0,
        )
        _command_log(self.paths, "launch", result)
        if not result.ok:
            raise ColabBatchError(
                f"remote worker launch failed: "
                f"{(result.stderr or result.stdout).strip()[-1500:]}"
            )
        launch = _json_from_output(result.stdout)
        transition_status(
            self.paths,
            "running",
            trainer_pid=launch.get("pid") if launch else None,
            selected_gpu=self.job["gpu_policy"]["selected_tier"],
        )
        append_event(self.paths, "launched", launch=launch or {})

    def _watch(self) -> dict[str, object]:
        if not self.session_name:
            raise ColabBatchError("cannot watch before session allocation")
        observability = self.job["observability"]
        poll_seconds = float(observability["poll_seconds"])
        heartbeat_timeout = float(observability["heartbeat_timeout_seconds"])
        recovery_sync_seconds = float(observability.get("recovery_sync_seconds", 300.0))
        last_remote_update = time.monotonic()
        last_recovery_sync = time.monotonic() - recovery_sync_seconds
        last_heartbeat: float | None = None
        last_remote_rank = -1
        last_signature: tuple[object, ...] | None = None
        stop_sent = False
        while True:
            result = self.cli.call(
                [
                    "exec",
                    "-s",
                    self.session_name,
                    "-f",
                    str(self.paths.poller),
                    "--timeout",
                    "60",
                ],
                timeout=120.0,
            )
            _command_log(self.paths, "poll", result)
            parsed = _json_from_output(result.stdout) if result.ok else None
            remote = (
                _validated_remote_status(
                    parsed,
                    expected_job_id=str(self.job["job_id"]),
                    expected_manifest_sha256=str(self.job["manifest_sha256"]),
                    expected_job_identity_sha256=str(self.job["job_identity_sha256"]),
                    expected_ownership_nonce=str(
                        self.job["session"]["ownership_nonce"]
                    ),
                )
                if parsed is not None
                else None
            )
            if remote is not None and remote.get("job_id") is not None:
                remote_state = str(remote["state"])
                remote_rank = (
                    0
                    if remote_state == "starting"
                    else 1
                    if remote_state == "running"
                    else 2
                )
                if remote_rank < last_remote_rank:
                    raise ColabBatchError(
                        "remote state regression: "
                        f"rank {last_remote_rank} -> {remote_state}"
                    )
                last_remote_rank = remote_rank
                if (
                    self.paths.stop_request.is_file()
                    and not stop_sent
                    and remote_state not in REMOTE_TERMINAL_STATES
                ):
                    self._send_stop()
                    stop_sent = True
                heartbeat = float(remote["heartbeat_at"])
                if last_heartbeat is None or heartbeat > last_heartbeat:
                    last_remote_update = time.monotonic()
                    last_heartbeat = heartbeat
                if (
                    remote_state == "running"
                    and int(remote.get("step", 0)) > 0
                    and time.monotonic() - last_recovery_sync >= recovery_sync_seconds
                ):
                    synced = self._sync_recovery()
                    last_recovery_sync = time.monotonic()
                    if not synced:
                        # A pre-checkpoint no-op should be retried promptly so
                        # the first published checkpoint is not delayed by a
                        # full recovery interval.
                        last_recovery_sync -= max(
                            0.0, recovery_sync_seconds - poll_seconds
                        )
                signature = (
                    remote.get("state"),
                    remote.get("run_state"),
                    remote.get("step"),
                    remote.get("native_steps"),
                    remote.get("last_error"),
                )
                if signature != last_signature:
                    append_event(self.paths, "remote_status", **remote)
                    last_signature = signature
                current_state = load_status(self.paths).get("state")
                local_state = "stopping" if current_state == "stopping" else "running"
                transition_status(
                    self.paths,
                    local_state,
                    remote=remote,
                    step=remote.get("step", 0),
                    total_steps=remote.get("total_steps"),
                    native_steps=remote.get("native_steps"),
                    trainer_state=remote.get("run_state"),
                    heartbeat_at=remote.get("heartbeat_at"),
                    last_error=remote.get("last_error"),
                )
                if remote.get("state") in REMOTE_TERMINAL_STATES:
                    return remote
            elif not result.ok:
                append_event(
                    self.paths,
                    "poll_error",
                    returncode=result.returncode,
                    error=(result.stderr or result.stdout).strip()[-1000:],
                )
            if time.monotonic() - last_remote_update > heartbeat_timeout:
                raise ColabBatchError(
                    f"remote heartbeat stale for more than {heartbeat_timeout:.1f}s"
                )
            self._sleep(poll_seconds)

    def _sync_recovery(self) -> bool:
        """Copy the latest published checkpoint outside the Colab runtime."""
        if not self.session_name:
            return False
        export = self.cli.call(
            [
                "exec",
                "-s",
                self.session_name,
                "-f",
                str(self.paths.recovery_exporter),
                "--timeout",
                "300",
            ],
            timeout=360.0,
        )
        _command_log(self.paths, "export-recovery", export)
        if not export.ok:
            append_event(
                self.paths,
                "recovery_sync_skipped",
                error=(export.stderr or export.stdout).strip()[-1000:],
            )
            return False
        remote_record = _json_from_output(export.stdout)
        if remote_record is None or remote_record.get(
            "job_identity_sha256"
        ) != self.job.get("job_identity_sha256"):
            raise ColabBatchError("remote recovery export identity mismatch")
        if remote_record.get("skipped") == "checkpoint-not-yet-published":
            append_event(
                self.paths,
                "recovery_sync_skipped",
                reason="checkpoint-not-yet-published",
            )
            return False
        local_manifest_path = self.paths.recovery_directory / "recovery-manifest.json"
        if local_manifest_path.is_file():
            try:
                local_record = _read_json(local_manifest_path)
            except ColabBatchError:
                local_record = None
            if _recovery_resume_signature(remote_record) == _recovery_resume_signature(
                local_record
            ):
                append_event(
                    self.paths,
                    "recovery_sync_unchanged",
                    checkpoint_step=remote_record.get("checkpoint_step"),
                )
                return True
        temporary = self.paths.root / ".recovery-download.tar"
        temporary.unlink(missing_ok=True)
        remote_archive = str(self.job["session"]["remote_root"]) + "/recovery.tar"
        download = self.cli.call(
            ["download", "-s", self.session_name, remote_archive, str(temporary)],
            timeout=1800.0,
        )
        _command_log(self.paths, "download-recovery", download)
        if not download.ok:
            append_event(
                self.paths,
                "recovery_sync_failed",
                error=(download.stderr or download.stdout).strip()[-1000:],
            )
            temporary.unlink(missing_ok=True)
            return False
        try:
            inventory = extract_recovery_archive(
                temporary,
                self.paths.recovery_directory,
                expected_job_identity_sha256=str(self.job["job_identity_sha256"]),
            )
            temporary.replace(self.paths.recovery_archive)
        finally:
            temporary.unlink(missing_ok=True)
        transition_status(
            self.paths,
            str(load_status(self.paths)["state"]),
            recovery_checkpoint_step=inventory.get("checkpoint_step"),
            recovery_synced_at=time.time(),
        )
        append_event(
            self.paths,
            "recovery_synced",
            checkpoint_step=inventory.get("checkpoint_step"),
            bytes=self.paths.recovery_archive.stat().st_size,
        )
        return True

    def _retrieve(self, remote_status: Mapping[str, object] | None = None) -> None:
        output = Path(str(self.job["local_run_dir"]))
        receipt = output / ".colab-artifact-receipt.json"
        if receipt.is_file():
            inventory = _read_json(receipt)
            expected = {
                "job_id": self.job["job_id"],
                "job_identity_sha256": self.job["job_identity_sha256"],
                "manifest_sha256": self.job["manifest_sha256"],
                "source_bundle_sha256": self.job["source"]["bundle_sha256"],
                "native_wheel_sha256": self.job["source"]["wheel"]["sha256"],
            }
            if any(inventory.get(key) != value for key, value in expected.items()):
                raise ColabBatchError("installed artifact receipt identity mismatch")
            files = inventory.get("files")
            if not isinstance(files, list) or inventory.get("required") != list(
                REQUIRED_ARTIFACTS
            ):
                raise ColabBatchError("installed artifact receipt is incomplete")
            names = set()
            for item in files:
                if not isinstance(item, Mapping) or not isinstance(
                    item.get("path"), str
                ):
                    raise ColabBatchError("installed artifact receipt entry is invalid")
                relative = item["path"]
                if not _safe_archive_name(relative) or sha256_file(
                    output / relative
                ) != item.get("sha256"):
                    raise ColabBatchError(f"installed artifact changed: {relative}")
                names.add(relative)
            if not set(REQUIRED_ARTIFACTS).issubset(names):
                raise ColabBatchError("installed artifact receipt lacks required files")
            _atomic_write_json(self.paths.artifact_manifest, inventory)
            append_event(self.paths, "retrieval_already_verified")
            return
        if not self.session_name:
            raise ColabBatchError("cannot retrieve before session allocation")
        packed = remote_status.get("artifacts") if remote_status is not None else None
        packed_by_worker = isinstance(packed, Mapping)
        if not isinstance(packed, Mapping):
            result = self.cli.call(
                [
                    "exec",
                    "-s",
                    self.session_name,
                    "-f",
                    str(self.paths.packer),
                    "--timeout",
                    "180",
                ],
                timeout=240.0,
            )
            _command_log(self.paths, "pack-artifacts", result)
            if not result.ok:
                raise ColabBatchError(
                    "remote artifact packing failed: "
                    f"{(result.stderr or result.stdout).strip()[-1500:]}"
                )
            packed = _json_from_output(result.stdout)
        if packed is None:
            raise ColabBatchError("remote artifact packer returned no JSON")
        archive_record = packed.get("archive")
        if not isinstance(archive_record, Mapping) or not isinstance(
            archive_record.get("sha256"), str
        ):
            raise ColabBatchError("remote artifact packer returned no archive hash")
        remote_archive = str(self.job["session"]["remote_root"]) + (
            "/artifacts.tar.gz" if packed_by_worker else "/artifacts-controller.tar.gz"
        )
        result = self.cli.call(
            [
                "download",
                "-s",
                self.session_name,
                remote_archive,
                str(self.paths.downloaded_archive),
            ],
            timeout=1800.0,
        )
        _command_log(self.paths, "download-artifacts", result)
        if not result.ok:
            detail = (result.stderr or result.stdout).strip()[-1500:]
            raise ColabBatchError(f"artifact download failed: {detail}")
        inventory = extract_artifact_archive(
            self.paths.downloaded_archive,
            Path(str(self.job["local_run_dir"])),
            expected_job_id=str(self.job["job_id"]),
            expected_manifest_sha256=str(self.job["manifest_sha256"]),
            expected_job_identity_sha256=str(self.job["job_identity_sha256"]),
            expected_source_bundle_sha256=str(self.job["source"]["bundle_sha256"]),
            expected_native_wheel_sha256=str(self.job["source"]["wheel"]["sha256"]),
            expected_archive_sha256=str(archive_record["sha256"]),
            manifest_output=self.paths.artifact_manifest,
        )
        append_event(
            self.paths,
            "retrieved",
            complete=inventory.get("complete"),
            file_count=len(inventory.get("files", [])),
            archive_sha256=sha256_file(self.paths.downloaded_archive),
        )

    def _download_remote_diagnostics(self) -> bool:
        """Keep worker, trainer, and bootstrap diagnostics before release."""
        if not self.session_name or not self._owns_session:
            return False
        downloaded = False
        for name in ("worker.log", "worker-training.log", "remote-status.json"):
            remote_log = str(self.job["session"]["remote_root"]) + "/" + name
            local_log = self.paths.logs / ("remote-" + name)
            try:
                result = self.cli.call(
                    ["download", "-s", self.session_name, remote_log, str(local_log)],
                    timeout=180.0,
                )
            except (ColabBatchError, OSError) as error:
                append_event(
                    self.paths,
                    "remote_diagnostics",
                    file=name,
                    downloaded=False,
                    error=str(error),
                )
                continue
            _command_log(self.paths, "download-remote-" + name, result)
            if result.ok and local_log.is_file():
                _trim_text_file(local_log, MAX_COMMAND_LOG_BYTES)
                downloaded = True
            append_event(
                self.paths,
                "remote_diagnostics",
                file=name,
                downloaded=result.ok,
                bytes=local_log.stat().st_size
                if result.ok and local_log.is_file()
                else 0,
            )
        return downloaded

    def _cleanup(self, *, keep_session: bool) -> bool:
        if keep_session:
            append_event(self.paths, "cleanup_skipped", reason="keep-session")
            return True
        if not self.session_name or not self._owns_session:
            return True
        try:
            result = self.cli.call(["stop", "-s", self.session_name], timeout=180.0)
        except ColabBatchError as error:
            append_event(self.paths, "session_cleanup", ok=False, error=str(error))
            return False
        _command_log(self.paths, "stop-session", result)
        append_event(self.paths, "session_cleanup", ok=result.ok)
        return result.ok

    def _local_terminal_state(self, remote: Mapping[str, object]) -> str:
        state = remote.get("state")
        if self.stop_requested or state in {"stopped", "cancelled"}:
            return "cancelled"
        if state == "completed":
            return "completed"
        return "failed"


def _session_name(job: Mapping[str, object]) -> str | None:
    session = job.get("session")
    if isinstance(session, Mapping) and isinstance(session.get("name"), str):
        return session["name"]
    return None


@contextmanager
def _controller_stop_signals(request_stop: Callable[[], None]):
    """SIGINT/SIGTERM request a checkpoint, including while a CLI call runs."""
    previous = {}

    def handler(_signum: int, _frame: object) -> None:
        request_stop()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, handler)
        yield
    finally:
        for signum, prior in previous.items():
            signal.signal(signum, prior)


@contextmanager
def _controller_lease(paths: JobPaths):
    """Ensure at most one local process supervises a job lifecycle."""

    paths.controller_lock.parent.mkdir(parents=True, exist_ok=True)
    with paths.controller_lock.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ColabBatchError(
                f"another controller already supervises job {paths.root.name}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _resolve_job(argument: str, job_root: Path) -> JobPaths:
    candidate = Path(argument)
    if candidate.is_dir() and (candidate / "job.json").is_file():
        return JobPaths(candidate.resolve())
    return job_paths(job_root, argument)


def _spawn_detached_controller(paths: JobPaths, *, keep_session: bool) -> int:
    """Launch a local supervisor that survives the submitting terminal."""

    command = [
        sys.executable,
        "-u",
        "-m",
        "dodge.ng.colab_batch",
        "run-job",
        str(paths.root),
    ]
    if keep_session:
        command.append("--keep-session")
    paths.logs.mkdir(parents=True, exist_ok=True)
    transition_status(
        paths,
        "planned",
        controller_command=command,
        controller_launching_at=time.time(),
    )
    with paths.controller_log.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    append_event(paths, "controller_detached", pid=process.pid)
    return process.pid


def _common_job_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
    parser.add_argument("--job-id")
    parser.add_argument("--job-prefix", default="pixel-dqn")
    parser.add_argument("--session-prefix", default="dodge-ng")
    parser.add_argument(
        "--gpu-tiers",
        default=",".join(DEFAULT_GPU_TIERS),
        help="ordered escalation tiers, starting with the cheapest viable request",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--heartbeat-timeout", type=float, default=180.0)
    parser.add_argument("--recovery-sync-seconds", type=float, default=300.0)
    parser.add_argument("--include-frame-store", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="local run/recovery directory containing checkpoint, metrics, snapshot",
    )
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help=(
            "pixel DQN arguments after `--`; paths are rewritten for the remote bundle"
        ),
    )


def _create_from_arguments(
    arguments: argparse.Namespace,
) -> tuple[JobPaths, dict[str, object]]:
    trainer_args = list(arguments.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    return create_job(
        local_run_dir=arguments.run_dir,
        manifest_path=arguments.manifest,
        trainer_args=trainer_args,
        gpu_tiers=parse_gpu_tiers(arguments.gpu_tiers),
        job_root=arguments.job_root,
        job_id=arguments.job_id,
        job_prefix=arguments.job_prefix,
        session_prefix=arguments.session_prefix,
        poll_seconds=arguments.poll_seconds,
        heartbeat_timeout_seconds=arguments.heartbeat_timeout,
        recovery_sync_seconds=arguments.recovery_sync_seconds,
        include_frame_store=arguments.include_frame_store,
        resume_from=arguments.resume_from,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-colab")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="write a local job manifest without allocation"
    )
    _common_job_arguments(plan)
    prepare = commands.add_parser(
        "prepare", help="build source and wheel payloads for a planned job"
    )
    prepare.add_argument("job")
    prepare.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
    submit = commands.add_parser(
        "submit", help="prepare, allocate, run, and retrieve a job"
    )
    _common_job_arguments(submit)
    submit.add_argument("--keep-session", action="store_true")
    submit.add_argument(
        "--foreground",
        action="store_true",
        help="keep supervision attached instead of returning after local launch",
    )
    run_job = commands.add_parser(
        "run-job", help="execute one already-prepared job (used by detached submit)"
    )
    run_job.add_argument("job")
    run_job.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
    run_job.add_argument("--keep-session", action="store_true")
    status = commands.add_parser("status", help="print local job status")
    status.add_argument("job")
    status.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
    watch = commands.add_parser(
        "watch", help="observe and finalize an already-launched job"
    )
    watch.add_argument("job")
    watch.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
    watch.add_argument("--keep-session", action="store_true")
    stop = commands.add_parser("stop", help="gracefully stop a running job")
    stop.add_argument("job")
    stop.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
    retrieve = commands.add_parser(
        "retrieve", help="retrieve artifacts from a terminal job"
    )
    retrieve.add_argument("job")
    retrieve.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)

    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "plan":
            paths, _ = _create_from_arguments(arguments)
            print(
                json.dumps(
                    {"job_id": paths.root.name, "job_directory": str(paths.root)}
                )
            )
            return 0
        if arguments.command == "prepare":
            paths = _resolve_job(arguments.job, arguments.job_root)
            prepared = prepare_job(paths)
            print(
                json.dumps(
                    {
                        "job_id": paths.root.name,
                        "source": prepared["source"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "submit":
            paths, _ = _create_from_arguments(arguments)
            prepare_job(paths)
            if not arguments.foreground:
                pid = _spawn_detached_controller(
                    paths,
                    keep_session=arguments.keep_session,
                )
                print(
                    json.dumps(
                        {
                            "job_id": paths.root.name,
                            "job_directory": str(paths.root),
                            "controller_pid": pid,
                            "state": "planned",
                        },
                        sort_keys=True,
                    )
                )
                return 0
            result = ColabBatchController(paths).run(
                keep_session=arguments.keep_session
            )
            print(json.dumps(result, sort_keys=True))
            return 0 if result.get("state") == "completed" else 1
        paths = _resolve_job(arguments.job, arguments.job_root)
        if arguments.command == "run-job":
            result = ColabBatchController(paths).run(
                keep_session=arguments.keep_session
            )
            print(json.dumps(result, sort_keys=True))
            return 0 if result.get("state") == "completed" else 1
        if arguments.command == "status":
            print(
                json.dumps(
                    {"job": load_job(paths), "status": load_status(paths)}, indent=2
                )
            )
            return 0
        controller = ColabBatchController(paths)
        if arguments.command == "stop":
            controller.request_stop()
            print(
                json.dumps(
                    {
                        "status": load_status(paths),
                        "stop_requested": paths.stop_request.is_file(),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "retrieve":
            with _controller_lease(paths):
                if (
                    load_status(paths).get("remote", {}).get("state")
                    not in REMOTE_TERMINAL_STATES
                ):
                    raise ColabBatchError(
                        "retrieve requires an observed terminal worker; use watch"
                    )
                controller._retrieve()
            print(json.dumps({"retrieved": True, "job_id": paths.root.name}))
            return 0
        if arguments.command == "watch":
            result = controller.recover(keep_session=arguments.keep_session)
            print(json.dumps(result))
            return 0 if result.get("state") == "completed" else 1
    except (ColabBatchError, OSError, ValueError) as error:
        print(f"dodge-ng-colab: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
