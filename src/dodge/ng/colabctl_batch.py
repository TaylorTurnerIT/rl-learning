"""Submit NG DDQN jobs through colabctl's durable Colab job API.

The local adapter owns reproducible source packaging and artifact verification.
``colabctl`` owns runtime allocation, detached execution, status, logs, and
cancellation.  The adapter deliberately constructs ``DetachedColabBackend``
over the sanctioned Google ``colab`` CLI transport; it does not opt into
colabctl's reverse-engineered native transport.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from dodge.control import PROJECT_ROOT
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, load_manifest

JOB_SCHEMA_VERSION: Final[int] = 1
ARTIFACT_SCHEMA_VERSION: Final[int] = 1
DEFAULT_JOB_ROOT: Final[Path] = PROJECT_ROOT / "history" / "dodge" / "ng" / "colab-jobs"
DEFAULT_GPU_TIERS: Final[tuple[str, ...]] = ("T4", "L4", "A100", "H100")
COLAB_GPU_TIERS: Final[tuple[str, ...]] = ("T4", "L4", "G4", "A100", "H100")
DEFAULT_TRAINER_MODULE: Final[str] = "dodge.ng.hpo"
DEFAULT_REMOTE_REQUIREMENTS: Final[tuple[str, ...]] = ("optuna>=4.0",)
MAX_LOCAL_LOG_BYTES: Final[int] = 512 * 1024
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
    }
)
_TRAINER_MODULE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)


class ColabCtlBatchError(RuntimeError):
    """Raised when a Colab job cannot be prepared or safely reconciled."""


@dataclass(frozen=True, slots=True)
class JobPaths:
    """Small, stable local record for one remote job."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "job.json"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def logs(self) -> Path:
        return self.root / "colabctl.log"

    @property
    def downloaded_archive(self) -> Path:
        return self.root / "artifacts.tar.gz"

    @property
    def artifact_manifest(self) -> Path:
        return self.root / "artifact-manifest.json"


def job_paths(job_root: Path, job_id: str) -> JobPaths:
    """Resolve one local job directory without permitting path traversal."""

    if not job_id or Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("job id must be a single safe path component")
    return JobPaths(Path(job_root).resolve() / job_id)


def parse_gpu_tiers(value: str | Sequence[str]) -> tuple[str, ...]:
    """Validate an ordered, T4-first GPU escalation policy."""

    raw = value.split(",") if isinstance(value, str) else list(value)
    tiers = tuple(item.strip().upper() for item in raw if item.strip())
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
    indexes = [COLAB_GPU_TIERS.index(tier) for tier in tiers]
    if indexes != sorted(indexes):
        raise ValueError("GPU tier policy must advance from cheaper to larger tiers")
    return tiers


def normalize_trainer_args(
    arguments: Sequence[str],
    *,
    remote_manifest: str,
    remote_run_dir: str,
) -> tuple[str, ...]:
    """Rewrite local path options so the remote trainer is self-contained."""

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

    replace_option("--manifest", remote_manifest)
    replace_option("--run-dir", remote_run_dir)
    return tuple(result)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ColabCtlBatchError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ColabCtlBatchError(f"JSON file must contain an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ColabCtlBatchError(f"cannot hash file {path}: {error}") from error
    return digest.hexdigest()


def _safe_archive_name(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError as error:
        raise ColabCtlBatchError(f"path is outside root: {path}") from error


def _source_file_paths(project_root: Path) -> list[Path]:
    roots = (
        project_root / "src" / "dodge",
        project_root / "native",
        project_root / "context" / "kits" / "dodge-ng",
    )
    direct = (project_root / "pyproject.toml", project_root / "uv.lock")
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            raise ColabCtlBatchError(f"required source path is missing: {root}")
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative_parts = path.relative_to(project_root).parts
            if _EXCLUDED_SOURCE_PARTS.intersection(relative_parts):
                continue
            files.append(path)
    files.extend(path for path in direct if path.is_file())
    return sorted(set(files))


def create_source_bundle(
    project_root: Path,
    output_path: Path,
) -> tuple[int, int, str]:
    """Package reproducible Python/native inputs and return count, bytes, hash."""

    root = Path(project_root).resolve()
    files = _source_file_paths(root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            archive.add(path, arcname=f"source/{relative}", recursive=False)
            total_bytes += path.stat().st_size
    return len(files), total_bytes, sha256_file(output)


def build_native_wheel(
    project_root: Path,
    output_directory: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> Path:
    """Build the native extension once for the remote Python runtime."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    command = (
        "maturin",
        "build",
        "--release",
        "--locked",
        "--manifest-path",
        str(Path(project_root) / "native" / "crates" / "dodge-python" / "Cargo.toml"),
        "--out",
        str(output),
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
        raise ColabCtlBatchError(
            "maturin is required to build the Colab native wheel; "
            "run this command inside devenv"
        ) from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "maturin failed").strip()
        raise ColabCtlBatchError(f"native wheel build failed: {detail[-2000:]}")
    wheels = sorted(output.glob("dodge_native-*.whl"))
    if not wheels:
        raise ColabCtlBatchError(f"maturin produced no dodge_native wheel in {output}")
    return wheels[-1]


def create_resume_bundle(
    run_directory: Path, output_path: Path
) -> list[dict[str, object]]:
    """Package a prior run, excluding only generated Python caches and frame stores."""

    root = Path(run_directory).resolve()
    if not root.is_dir():
        raise ColabCtlBatchError(f"resume directory does not exist: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative_parts = path.relative_to(root).parts
        if "__pycache__" in relative_parts or path.name in {
            ".pixel-frames.u8",
            ".colab-artifact-receipt.json",
            "artifact-manifest.json",
        }:
            continue
        files.append(path)
    if not files:
        raise ColabCtlBatchError(f"resume directory has no files: {root}")
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files)
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(files):
            archive.add(
                path, arcname=path.relative_to(root).as_posix(), recursive=False
            )
    return records


def _create_payload(
    source_bundle: Path,
    wheel: Path,
    output_path: Path,
    *,
    resume_bundle: Path | None = None,
) -> str:
    output = Path(output_path)
    wheel_name = Path(wheel).name
    if not wheel_name.endswith(".whl"):
        raise ColabCtlBatchError(f"native wheel has an invalid filename: {wheel_name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        archive.add(source_bundle, arcname="source.tar.gz", recursive=False)
        archive.add(wheel, arcname=wheel_name, recursive=False)
        if resume_bundle is not None:
            archive.add(resume_bundle, arcname="resume.tar.gz", recursive=False)
    return sha256_file(output)


def _build_remote_script(
    payload: Path,
    *,
    submission_id: str,
    payload_sha256: str,
    source_sha256: str,
    native_wheel_sha256: str,
    resume_sha256: str | None,
    manifest_relative_path: str,
    manifest_sha256: str,
    trainer_module: str,
    trainer_args: Sequence[str],
    job_identity_sha256: str,
    include_frame_store: bool,
    output_path: Path,
) -> None:
    """Create a standalone script whose only remote input is its embedded payload."""

    payload_b64 = base64.b64encode(Path(payload).read_bytes()).decode("ascii")
    script = f"""#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

SUBMISSION_ID = {submission_id!r}
PAYLOAD_SHA256 = {payload_sha256!r}
SOURCE_SHA256 = {source_sha256!r}
NATIVE_WHEEL_SHA256 = {native_wheel_sha256!r}
RESUME_SHA256 = {resume_sha256!r}
MANIFEST_RELATIVE_PATH = {manifest_relative_path!r}
MANIFEST_SHA256 = {manifest_sha256!r}
TRAINER_MODULE = {trainer_module!r}
TRAINER_ARGS = {list(trainer_args)!r}
JOB_IDENTITY_SHA256 = {job_identity_sha256!r}
INCLUDE_FRAME_STORE = {include_frame_store!r}
PAYLOAD_B64 = {payload_b64!r}

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE / "workspace"
SOURCE_ROOT = WORKSPACE / "source"
SOURCE_PACKAGE_ROOT = SOURCE_ROOT / "src"
RUN_DIRECTORY = HERE / "run"
PAYLOAD_PATH = HERE / "payload.tar.gz"
ARTIFACT_ARCHIVE = HERE / "artifacts.tar.gz"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def extract_files(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = set()
        for member in members:
            if not safe_name(member.name) or not member.isfile():
                raise RuntimeError("payload contains an unsafe or non-file member")
            if member.name in names:
                raise RuntimeError("payload contains duplicate members")
            names.add(member.name)
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents:
                raise RuntimeError("payload member escapes its destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("payload member could not be read")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def artifact_files() -> list[tuple[Path, str]]:
    records = []
    for path in sorted(RUN_DIRECTORY.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(RUN_DIRECTORY).as_posix()
        if path.name == ".pixel-frames.u8" and not INCLUDE_FRAME_STORE:
            continue
        if path.name in {{".colab-artifact-receipt.json", "artifact-manifest.json"}}:
            continue
        records.append((path, relative))
    return records


def archive_artifacts(exit_code: int, error: str | None = None) -> dict[str, object]:
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records = []
    for path, relative in artifact_files():
        records.append(
            {{
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }}
        )
    inventory = {{
        "schema_version": 1,
        "submission_id": SUBMISSION_ID,
        "job_identity_sha256": JOB_IDENTITY_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "source_bundle_sha256": SOURCE_SHA256,
        "native_wheel_sha256": NATIVE_WHEEL_SHA256,
        "payload_sha256": PAYLOAD_SHA256,
        "files": records,
        "complete": exit_code == 0,
        "trainer_exit_code": exit_code,
        "frame_store_included": INCLUDE_FRAME_STORE,
    }}
    if error is not None:
        inventory["error"] = error[-4000:]
    encoded = json.dumps(inventory, indent=2, sort_keys=True).encode("utf-8")
    with tarfile.open(ARTIFACT_ARCHIVE, "w:gz") as archive:
        for path, relative in artifact_files():
            archive.add(path, arcname=relative, recursive=False)
        info = tarfile.TarInfo("artifact-manifest.json")
        info.size = len(encoded)
        archive.addfile(info, io.BytesIO(encoded))
    (HERE / "artifact-manifest.json").write_bytes(encoded)
    return inventory


def main() -> int:
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    try:
        payload = base64.b64decode(PAYLOAD_B64, validate=True)
        if sha256_bytes(payload) != PAYLOAD_SHA256:
            raise RuntimeError("embedded payload hash does not match")
        PAYLOAD_PATH.write_bytes(payload)
        payload_root = HERE / "payload"
        if payload_root.exists():
            shutil.rmtree(payload_root)
        extract_files(PAYLOAD_PATH, payload_root)
        source_bundle = payload_root / "source.tar.gz"
        wheel_candidates = sorted(payload_root.glob("*.whl"))
        if len(wheel_candidates) != 1:
            raise RuntimeError("native wheel is missing or ambiguous")
        wheel = wheel_candidates[0]
        if sha256_file(source_bundle) != SOURCE_SHA256:
            raise RuntimeError("source bundle hash does not match")
        if sha256_file(wheel) != NATIVE_WHEEL_SHA256:
            raise RuntimeError("native wheel hash does not match")
        if RESUME_SHA256 is not None:
            resume = payload_root / "resume.tar.gz"
            if sha256_file(resume) != RESUME_SHA256:
                raise RuntimeError("resume bundle hash does not match")
        if WORKSPACE.exists():
            shutil.rmtree(WORKSPACE)
        extract_files(source_bundle, WORKSPACE)
        manifest = SOURCE_ROOT / MANIFEST_RELATIVE_PATH
        if not manifest.is_file():
            raise RuntimeError("source manifest hash does not match")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest_payload, dict):
            raise RuntimeError("source manifest hash does not match")
        manifest_body = {{
            key: manifest_payload[key]
            for key in (
                "manifest_id",
                "schema_version",
                "split_seed",
                "legacy_seed_max",
                "sample_space",
                "training_seeds",
                "holdout_seeds",
            )
        }}
        manifest_hash = hashlib.sha256(
            json.dumps(
                manifest_body,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            manifest_payload.get("manifest_sha256") != manifest_hash
            or manifest_hash != MANIFEST_SHA256
        ):
            raise RuntimeError("source manifest hash does not match")
        if RESUME_SHA256 is not None:
            extract_files(payload_root / "resume.tar.gz", RUN_DIRECTORY)
        install = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                str(wheel),
            ],
            check=False,
            text=True,
        )
        if install.returncode != 0:
            raise RuntimeError("native wheel installation failed")
        environment = os.environ.copy()
        prior_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(SOURCE_PACKAGE_ROOT)
            if not prior_pythonpath
            else str(SOURCE_PACKAGE_ROOT) + os.pathsep + prior_pythonpath
        )
        trainer_args = [
            argument.replace("{{SOURCE_ROOT}}", str(SOURCE_ROOT)).replace(
                "{{RUN_DIRECTORY}}", str(RUN_DIRECTORY)
            )
            for argument in TRAINER_ARGS
        ]
        command = [sys.executable, "-u", "-m", TRAINER_MODULE, *trainer_args]
        print(json.dumps({{"event": "trainer_start", "command": command}}), flush=True)
        trainer = subprocess.run(
            command,
            cwd=SOURCE_ROOT,
            env=environment,
            check=False,
        )
        exit_code = int(trainer.returncode)
        archive_artifacts(exit_code)
        print(
            json.dumps(
                {{
                    "event": "trainer_complete",
                    "exit_code": exit_code,
                    "artifact_count": len(artifact_files()),
                }}
            ),
            flush=True,
        )
        return exit_code
    except Exception as error:
        detail = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        (RUN_DIRECTORY / "failure.txt").write_text(detail[-8000:], encoding="utf-8")
        try:
            archive_artifacts(1, detail)
        except Exception as archive_error:
            print(
                f"artifact archive failed: {{archive_error}}",
                file=sys.stderr,
                flush=True,
            )
        print(detail, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script, encoding="utf-8")
    output_path.chmod(0o700)


def _load_colabctl() -> tuple[Any, ...]:
    """Import colabctl lazily so local CPU work does not require the extra."""

    try:
        from colabctl import Accelerator, AcceleratorUnavailableError, ColabClient
        from colabctl.backends.base import JobSpec
        from colabctl.jobs.backend import DetachedColabBackend
        from colabctl.state import StateStore
    except ImportError as error:
        raise ColabCtlBatchError(
            "colabctl is not installed; run `uv sync --extra colab` "
            "or invoke this command through `just dodge-ng-colab`"
        ) from error
    return (
        Accelerator,
        AcceleratorUnavailableError,
        ColabClient,
        DetachedColabBackend,
        JobSpec,
        StateStore,
    )


def _validate_colab_cli_dependency() -> None:
    """Reject an incompatible Colab CLI client before allocating a runtime."""

    try:
        import jupyter_kernel_client
    except ImportError as error:
        raise ColabCtlBatchError(
            "jupyter-kernel-client is missing; run `uv sync --extra colab`"
        ) from error
    if not all(
        hasattr(jupyter_kernel_client, name)
        for name in ("KernelClient", "JupyterSubprotocol")
    ):
        raise ColabCtlBatchError(
            "google-colab-cli requires KernelClient and JupyterSubprotocol; "
            "the colab extra must pin the compatible Google Colab fork"
        )


@asynccontextmanager
async def _backend(
    *,
    auth_mode: str,
    colab_bin: str,
) -> Any:
    (
        _accelerator,
        _unavailable_error,
        colab_client,
        detached_backend,
        _job_spec,
        _state_store,
    ) = _load_colabctl()
    del _accelerator, _unavailable_error, _job_spec, _state_store
    client = colab_client(
        transport_name="cli", auth_mode=auth_mode, colab_bin=colab_bin
    )
    backend = detached_backend(client.transport)
    try:
        yield client, backend
    finally:
        await backend.aclose()


def _state_value(value: object) -> str:
    state = getattr(value, "value", value)
    return str(state).lower()


def _info_json(info: object) -> dict[str, object]:
    return {
        "job_id": str(getattr(info, "id", "")),
        "backend": str(getattr(info, "backend", "colab")),
        "state": _state_value(getattr(info, "state", "unknown")),
        "accelerator": str(getattr(getattr(info, "accelerator", None), "value", "")),
        "detail": getattr(info, "detail", None),
    }


def _write_status(paths: JobPaths, info: object, **fields: object) -> None:
    _atomic_write_json(
        paths.status,
        {
            "schema_version": JOB_SCHEMA_VERSION,
            **_info_json(info),
            "updated_at": time.time(),
            **fields,
        },
    )


def _write_bounded_log(paths: JobPaths, text: str) -> None:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) > MAX_LOCAL_LOG_BYTES:
        encoded = encoded[-MAX_LOCAL_LOG_BYTES:]
        newline = encoded.find(b"\n")
        if newline >= 0:
            encoded = encoded[newline + 1 :]
    paths.logs.parent.mkdir(parents=True, exist_ok=True)
    paths.logs.write_bytes(encoded)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    return slug[:36] or "dodge-ng"


def _new_submission_id(prefix: str) -> str:
    return f"{_safe_slug(prefix)}-{int(time.time())}-{secrets.token_hex(3)}"


def _validate_trainer_module(value: str) -> str:
    if not _TRAINER_MODULE_RE.fullmatch(value):
        raise ValueError(f"invalid trainer module: {value!r}")
    return value


def _validate_manifest(manifest_path: Path) -> tuple[Path, object, str]:
    path = Path(manifest_path).resolve()
    project_root = PROJECT_ROOT.resolve()
    relative = _relative_to_root(path, project_root)
    manifest = load_manifest(path)
    manifest.validate()
    return path, manifest, relative


def _requirements(trainer_module: str, requested: Sequence[str]) -> list[str]:
    values = list(
        DEFAULT_REMOTE_REQUIREMENTS if trainer_module == DEFAULT_TRAINER_MODULE else ()
    )
    for requirement in requested:
        if requirement not in values:
            values.append(requirement)
    return values


async def _submit_remote(
    script: Path,
    *,
    gpu_tiers: Sequence[str],
    requirements: Sequence[str],
    timeout: int | None,
    session_name: str,
    auth_mode: str,
    colab_bin: str,
) -> tuple[object, list[dict[str, object]], object]:
    _validate_colab_cli_dependency()
    (
        accelerator,
        unavailable_error,
        colab_client,
        detached_backend,
        job_spec,
        state_store,
    ) = _load_colabctl()
    try:
        from colabctl import TooManyAssignmentsError
    except ImportError as error:
        raise ColabCtlBatchError(
            "colabctl does not expose typed capacity errors; upgrade the colab extra"
        ) from error
    client = colab_client(
        transport_name="cli", auth_mode=auth_mode, colab_bin=colab_bin
    )
    backend = detached_backend(client.transport)
    attempts: list[dict[str, object]] = []
    try:
        for tier in gpu_tiers:
            try:
                info = await backend.submit(
                    job_spec(
                        script_path=str(script),
                        requirements=list(requirements),
                        accelerator=accelerator(tier),
                        timeout=timeout,
                        name=session_name,
                    )
                )
            except TooManyAssignmentsError as error:
                attempts.append(
                    {"tier": tier, "state": "capacity", "detail": str(error)}
                )
                continue
            except unavailable_error as error:
                attempts.append(
                    {"tier": tier, "state": "unavailable", "detail": str(error)}
                )
                continue
            except Exception:
                with suppress(Exception):
                    await client.transport.stop(session_name)
                raise
            attempts.append({"tier": tier, "state": "submitted"})
            stored = state_store().get_job(info.id)
            if stored is None:
                raise ColabCtlBatchError(
                    f"colabctl submitted {info.id} but did not persist its job record"
                )
            return info, attempts, stored
    finally:
        await backend.aclose()
    raise ColabCtlBatchError(
        "no requested Colab GPU tier was available: "
        + "; ".join(f"{item['tier']}: {item['detail']}" for item in attempts)
    )


def _job_record(
    *,
    job_id: str,
    submission_id: str,
    run_directory: Path,
    manifest_path: Path,
    manifest: object,
    manifest_relative_path: str,
    trainer_module: str,
    trainer_args: Sequence[str],
    requirements: Sequence[str],
    gpu_tiers: Sequence[str],
    attempts: Sequence[Mapping[str, object]],
    source_file_count: int,
    source_bytes: int,
    source_sha256: str,
    native_wheel_sha256: str,
    payload_sha256: str,
    resume_sha256: str | None,
    job_identity_sha256: str,
    session_name: str | None,
    remote_dir: str | None,
) -> dict[str, object]:
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "submission_id": submission_id,
        "owner": "dodge-ng-colabctl",
        "transport": "cli",
        "project_root": str(PROJECT_ROOT.resolve()),
        "run_directory": str(Path(run_directory).resolve()),
        "manifest": {
            "path": str(manifest_path),
            "relative_path": manifest_relative_path,
            "manifest_id": getattr(manifest, "manifest_id", None),
            "sha256": getattr(manifest, "sha256", None),
        },
        "trainer": {
            "module": trainer_module,
            "args": list(trainer_args),
            "requirements": list(requirements),
        },
        "gpu_policy": {
            "requested_tiers": list(gpu_tiers),
            "attempts": [dict(item) for item in attempts],
        },
        "payload": {
            "source_file_count": source_file_count,
            "source_bytes": source_bytes,
            "source_bundle_sha256": source_sha256,
            "native_wheel_sha256": native_wheel_sha256,
            "payload_sha256": payload_sha256,
            "resume_bundle_sha256": resume_sha256,
        },
        "job_identity_sha256": job_identity_sha256,
        "colab": {
            "remote_job_id": job_id,
            "session_name": session_name,
            "remote_dir": remote_dir,
        },
        "created_at": time.time(),
    }


async def submit_job(
    *,
    run_directory: Path,
    manifest_path: Path,
    job_root: Path,
    gpu_tiers: Sequence[str],
    trainer_module: str,
    trainer_args: Sequence[str],
    requirements: Sequence[str],
    job_name: str | None,
    timeout: int | None,
    resume_from: Path | None,
    include_frame_store: bool,
    auth_mode: str,
    colab_bin: str,
) -> tuple[JobPaths, dict[str, object]]:
    """Prepare, submit, and persist one detached colabctl job."""

    tiers = parse_gpu_tiers(gpu_tiers)
    if timeout is not None and timeout <= 0:
        raise ValueError("job timeout must be positive")
    trainer_module = _validate_trainer_module(trainer_module)
    manifest_path, manifest, manifest_relative_path = _validate_manifest(manifest_path)
    root = Path(job_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    submission_id = _new_submission_id(job_name or trainer_module.rsplit(".", 1)[-1])
    remote_manifest = f"{{SOURCE_ROOT}}/{manifest_relative_path}"
    normalized_args = normalize_trainer_args(
        trainer_args,
        remote_manifest=remote_manifest,
        remote_run_dir="{RUN_DIRECTORY}",
    )
    remote_requirements = _requirements(trainer_module, requirements)
    run_directory = Path(run_directory).resolve()

    with tempfile.TemporaryDirectory(prefix=".submit-", dir=root) as temporary_name:
        temporary = Path(temporary_name)
        source_bundle = temporary / "source.tar.gz"
        source_file_count, source_bytes, source_sha256 = create_source_bundle(
            PROJECT_ROOT, source_bundle
        )
        wheel = build_native_wheel(PROJECT_ROOT, temporary / "wheels")
        native_wheel_sha256 = sha256_file(wheel)
        resume_bundle: Path | None = None
        resume_sha256: str | None = None
        if resume_from is not None:
            resume_bundle = temporary / "resume.tar.gz"
            create_resume_bundle(resume_from, resume_bundle)
            resume_sha256 = sha256_file(resume_bundle)
        payload = temporary / "payload.tar.gz"
        payload_sha256 = _create_payload(
            source_bundle,
            wheel,
            payload,
            resume_bundle=resume_bundle,
        )
        identity = {
            "schema_version": JOB_SCHEMA_VERSION,
            "submission_id": submission_id,
            "manifest_sha256": manifest.sha256,
            "source_bundle_sha256": source_sha256,
            "native_wheel_sha256": native_wheel_sha256,
            "payload_sha256": payload_sha256,
            "resume_bundle_sha256": resume_sha256,
            "trainer_module": trainer_module,
            "trainer_args": list(normalized_args),
            "requirements": remote_requirements,
        }
        job_identity_sha256 = _canonical_sha256(identity)
        script = temporary / "job.py"
        _build_remote_script(
            payload,
            submission_id=submission_id,
            payload_sha256=payload_sha256,
            source_sha256=source_sha256,
            native_wheel_sha256=native_wheel_sha256,
            resume_sha256=resume_sha256,
            manifest_relative_path=manifest_relative_path,
            manifest_sha256=manifest.sha256,
            trainer_module=trainer_module,
            trainer_args=normalized_args,
            job_identity_sha256=job_identity_sha256,
            include_frame_store=include_frame_store,
            output_path=script,
        )
        info, attempts, stored = await _submit_remote(
            script,
            gpu_tiers=tiers,
            requirements=remote_requirements,
            timeout=timeout,
            session_name=f"dodge-ng-{secrets.token_hex(4)}",
            auth_mode=auth_mode,
            colab_bin=colab_bin,
        )

    job_id = str(info.id)
    paths = job_paths(root, job_id)
    if paths.root.exists():
        raise ColabCtlBatchError(f"local job directory already exists: {paths.root}")
    paths.root.mkdir(parents=True)
    session_name = getattr(stored, "session_name", None)
    remote_dir = getattr(stored, "remote_dir", None)
    record = _job_record(
        job_id=job_id,
        submission_id=submission_id,
        run_directory=run_directory,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_relative_path=manifest_relative_path,
        trainer_module=trainer_module,
        trainer_args=normalized_args,
        requirements=remote_requirements,
        gpu_tiers=tiers,
        attempts=attempts,
        source_file_count=source_file_count,
        source_bytes=source_bytes,
        source_sha256=source_sha256,
        native_wheel_sha256=native_wheel_sha256,
        payload_sha256=payload_sha256,
        resume_sha256=resume_sha256,
        job_identity_sha256=job_identity_sha256,
        session_name=session_name,
        remote_dir=remote_dir,
    )
    _atomic_write_json(paths.manifest, record)
    _write_status(paths, info, submission_id=submission_id)
    return paths, record


def _resolve_local_job(argument: str, job_root: Path) -> JobPaths | None:
    candidate = Path(argument)
    if candidate.is_dir() and (candidate / "job.json").is_file():
        return JobPaths(candidate.resolve())
    direct = job_paths(job_root, argument)
    if direct.manifest.is_file():
        return direct
    root = Path(job_root).resolve()
    if not root.is_dir():
        return None
    for directory in root.iterdir():
        manifest = directory / "job.json"
        if not directory.is_dir() or not manifest.is_file():
            continue
        try:
            value = _read_json(manifest)
        except ColabCtlBatchError:
            continue
        if value.get("job_id") == argument or value.get("submission_id") == argument:
            return JobPaths(directory)
        colab = value.get("colab")
        if isinstance(colab, Mapping) and colab.get("remote_job_id") == argument:
            return JobPaths(directory)
    return None


def _record_and_id(
    argument: str, job_root: Path
) -> tuple[JobPaths | None, str, dict[str, object] | None]:
    paths = _resolve_local_job(argument, job_root)
    if paths is None:
        return None, argument, None
    record = _read_json(paths.manifest)
    colab = record.get("colab")
    remote_id = (
        str(colab.get("remote_job_id"))
        if isinstance(colab, Mapping) and colab.get("remote_job_id")
        else str(record.get("job_id", argument))
    )
    return paths, remote_id, record


def _stored_session(
    record: Mapping[str, object], state_store: object
) -> tuple[str, str]:
    colab = record.get("colab")
    stored_id = str(record.get("job_id", ""))
    stored = state_store.get_job(stored_id)
    session_name = getattr(stored, "session_name", None) if stored is not None else None
    remote_dir = getattr(stored, "remote_dir", None) if stored is not None else None
    if isinstance(colab, Mapping):
        session_name = session_name or colab.get("session_name")
        remote_dir = remote_dir or colab.get("remote_dir")
    if not isinstance(session_name, str) or not isinstance(remote_dir, str):
        raise ColabCtlBatchError(
            f"colabctl has no reattachable session record for job {stored_id}"
        )
    return session_name, remote_dir


def _extract_artifacts(
    archive_path: Path,
    destination: Path,
    *,
    expected_submission_id: str,
    expected_job_identity_sha256: str,
    expected_manifest_sha256: str,
    expected_source_sha256: str,
    expected_native_wheel_sha256: str,
    expected_payload_sha256: str,
    manifest_output: Path | None = None,
) -> dict[str, object]:
    """Verify identity, hashes, and membership before installing artifacts."""

    archive = Path(archive_path).resolve()
    output = Path(destination).resolve()
    archive_sha256 = sha256_file(archive)
    if output.exists():
        receipt = output / ".colab-artifact-receipt.json"
        if receipt.is_file():
            previous = _read_json(receipt)
            if previous.get("archive_sha256") == archive_sha256:
                return previous
        raise ColabCtlBatchError(f"artifact destination already exists: {output}")
    staging = archive.parent / f".artifact-extract-{os.getpid()}"
    install = output.parent / f".{output.name}.incoming-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(install, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        with tarfile.open(archive, "r:gz") as stream:
            members = stream.getmembers()
            member_names = [member.name for member in members]
            if len(member_names) != len(set(member_names)):
                raise ColabCtlBatchError("artifact archive contains duplicate members")
            if any(not _safe_archive_name(member.name) for member in members):
                raise ColabCtlBatchError("artifact archive contains an unsafe path")
            if any(not member.isfile() for member in members):
                raise ColabCtlBatchError("artifact archive contains a non-file entry")
            for member in members:
                target = (staging / member.name).resolve()
                if staging.resolve() not in target.parents:
                    raise ColabCtlBatchError(
                        "artifact archive member escapes its destination"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise ColabCtlBatchError(
                        f"artifact archive member could not be read: {member.name}"
                    )
                with source, target.open("wb") as destination_stream:
                    shutil.copyfileobj(source, destination_stream, length=1024 * 1024)
        inventory = _read_json(staging / "artifact-manifest.json")
        expected = {
            "submission_id": expected_submission_id,
            "job_identity_sha256": expected_job_identity_sha256,
            "manifest_sha256": expected_manifest_sha256,
            "source_bundle_sha256": expected_source_sha256,
            "native_wheel_sha256": expected_native_wheel_sha256,
            "payload_sha256": expected_payload_sha256,
        }
        for key, value in expected.items():
            if inventory.get(key) != value:
                raise ColabCtlBatchError(f"artifact archive {key} does not match job")
        files = inventory.get("files")
        if not isinstance(files, list):
            raise ColabCtlBatchError("artifact archive file inventory is invalid")
        declared_names: list[str] = []
        for item in files:
            if not isinstance(item, Mapping):
                raise ColabCtlBatchError("artifact archive file entry is invalid")
            relative = item.get("path")
            expected_sha = item.get("sha256")
            if not isinstance(relative, str) or not _safe_archive_name(relative):
                raise ColabCtlBatchError("artifact archive file path is invalid")
            if not isinstance(expected_sha, str):
                raise ColabCtlBatchError("artifact archive file hash is invalid")
            source = staging / relative
            if not source.is_file() or sha256_file(source) != expected_sha:
                raise ColabCtlBatchError(
                    f"artifact hash verification failed: {relative}"
                )
            declared_names.append(relative)
        if set(member_names) != {*declared_names, "artifact-manifest.json"}:
            raise ColabCtlBatchError(
                "artifact archive membership does not match inventory"
            )
        install.parent.mkdir(parents=True, exist_ok=True)
        install.mkdir(parents=True)
        for relative in declared_names:
            target = install / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.resolve().relative_to(install.resolve())
            shutil.copy2(staging / relative, target)
        inventory = {
            **inventory,
            "archive_sha256": archive_sha256,
            "retrieved_at": time.time(),
        }
        _atomic_write_json(install / ".colab-artifact-receipt.json", inventory)
        output.parent.mkdir(parents=True, exist_ok=True)
        install.replace(output)
        if manifest_output is not None:
            _atomic_write_json(Path(manifest_output), inventory)
        return inventory
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(install, ignore_errors=True)


async def _download_artifacts(
    client: object,
    state_store: object,
    paths: JobPaths,
    record: Mapping[str, object],
) -> dict[str, object]:
    session_name, remote_dir = _stored_session(record, state_store)
    session = client.attach(session_name)
    await session.download(f"{remote_dir}/artifacts.tar.gz", paths.downloaded_archive)
    manifest = record.get("manifest")
    payload = record.get("payload")
    if not isinstance(manifest, Mapping) or not isinstance(payload, Mapping):
        raise ColabCtlBatchError("local job record lacks artifact identity")
    return _extract_artifacts(
        paths.downloaded_archive,
        Path(str(record["run_directory"])),
        expected_submission_id=str(record["submission_id"]),
        expected_job_identity_sha256=str(record["job_identity_sha256"]),
        expected_manifest_sha256=str(manifest["sha256"]),
        expected_source_sha256=str(payload["source_bundle_sha256"]),
        expected_native_wheel_sha256=str(payload["native_wheel_sha256"]),
        expected_payload_sha256=str(payload["payload_sha256"]),
        manifest_output=paths.artifact_manifest,
    )


async def _release_session(
    client: object, state_store: object, record: Mapping[str, object]
) -> None:
    session_name, _remote_dir = _stored_session(record, state_store)
    await client.attach(session_name).stop()


async def _status_command(
    argument: str,
    *,
    job_root: Path,
    auth_mode: str,
    colab_bin: str,
) -> int:
    paths, remote_id, record = _record_and_id(argument, job_root)
    async with _backend(auth_mode=auth_mode, colab_bin=colab_bin) as (_client, backend):
        info = await backend.status(remote_id)
    if paths is not None and record is not None:
        _write_status(paths, info)
        output = {"job": record, "status": _read_json(paths.status)}
    else:
        output = {"status": _info_json(info)}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


async def _logs_command(
    argument: str,
    *,
    job_root: Path,
    tail: int,
    auth_mode: str,
    colab_bin: str,
) -> int:
    paths, remote_id, _record = _record_and_id(argument, job_root)
    async with _backend(auth_mode=auth_mode, colab_bin=colab_bin) as (_client, backend):
        text = await backend.logs(remote_id)
    if paths is not None:
        _write_bounded_log(paths, text)
    lines = text.splitlines()
    if tail >= 0:
        lines = lines[-tail:]
    if lines:
        print("\n".join(lines))
    return 0


async def _watch_command(
    argument: str,
    *,
    job_root: Path,
    poll_seconds: float,
    keep_session: bool,
    auth_mode: str,
    colab_bin: str,
) -> int:
    if poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    paths, remote_id, record = _record_and_id(argument, job_root)
    if paths is None or record is None:
        raise ColabCtlBatchError(
            "watch requires a local job record for artifact retrieval"
        )
    (
        _accelerator,
        _unavailable_error,
        _client_type,
        _backend_type,
        _job_spec,
        state_store_type,
    ) = _load_colabctl()
    del _accelerator, _unavailable_error, _client_type, _backend_type, _job_spec
    async with _backend(auth_mode=auth_mode, colab_bin=colab_bin) as (client, backend):
        state_store = state_store_type()
        while True:
            info = await backend.status(remote_id)
            _write_status(paths, info)
            state = getattr(info, "state", None)
            print(json.dumps(_info_json(info), sort_keys=True), flush=True)
            if bool(getattr(state, "is_terminal", False)):
                break
            await asyncio.sleep(poll_seconds)
        result = await backend.result(remote_id)
        result_state = _state_value(getattr(result, "state", "unknown"))
        result_text = str(getattr(result, "stdout", ""))
        _write_bounded_log(paths, result_text)
        _write_status(paths, result, result_state=result_state)
        artifacts: dict[str, object] | None = None
        retrieval_error: str | None = None
        try:
            artifacts = await _download_artifacts(client, state_store, paths, record)
        except ColabCtlBatchError as error:
            retrieval_error = str(error)
            print(f"artifact retrieval skipped: {error}", file=sys.stderr)
        if not keep_session and artifacts is not None:
            await _release_session(client, state_store, record)
    output: dict[str, object] = {
        "job_id": remote_id,
        "state": result_state,
        "retrieved": artifacts is not None,
        "session_released": not keep_session and artifacts is not None,
    }
    if artifacts is not None:
        output["artifact_count"] = len(artifacts.get("files", []))
    if retrieval_error is not None:
        output["retrieval_error"] = retrieval_error
    print(json.dumps(output, sort_keys=True))
    return 0 if bool(getattr(result, "ok", False)) and artifacts is not None else 1


async def _stop_command(
    argument: str,
    *,
    job_root: Path,
    keep_session: bool,
    auth_mode: str,
    colab_bin: str,
) -> int:
    paths, remote_id, record = _record_and_id(argument, job_root)
    if record is None:
        raise ColabCtlBatchError(
            "stop requires a local job record with its session name"
        )
    (
        _accelerator,
        _unavailable_error,
        _client_type,
        _backend_type,
        _job_spec,
        state_store_type,
    ) = _load_colabctl()
    del _accelerator, _unavailable_error, _client_type, _backend_type, _job_spec
    async with _backend(auth_mode=auth_mode, colab_bin=colab_bin) as (client, backend):
        state_store = state_store_type()
        info = await backend.status(remote_id)
        if not bool(getattr(getattr(info, "state", None), "is_terminal", False)):
            await backend.cancel(remote_id)
            info = await backend.status(remote_id)
        if not keep_session:
            await _release_session(client, state_store, record)
        if paths is not None:
            _write_status(paths, info, stop_requested=True)
    print(
        json.dumps({"job_id": remote_id, "stopped": not keep_session}, sort_keys=True)
    )
    return 0


async def _retrieve_command(
    argument: str,
    *,
    job_root: Path,
    auth_mode: str,
    colab_bin: str,
) -> int:
    paths, remote_id, record = _record_and_id(argument, job_root)
    if paths is None or record is None:
        raise ColabCtlBatchError("retrieve requires a local job record")
    (
        _accelerator,
        _unavailable_error,
        _client_type,
        _backend_type,
        _job_spec,
        state_store_type,
    ) = _load_colabctl()
    del _accelerator, _unavailable_error, _client_type, _backend_type, _job_spec
    async with _backend(auth_mode=auth_mode, colab_bin=colab_bin) as (
        client,
        backend,
    ):
        info = await backend.status(remote_id)
        if not bool(getattr(getattr(info, "state", None), "is_terminal", False)):
            raise ColabCtlBatchError(
                "retrieve requires a terminal colabctl job; use watch first"
            )
        artifacts = await _download_artifacts(client, state_store_type(), paths, record)
    print(
        json.dumps(
            {
                "job_id": remote_id,
                "run_directory": record["run_directory"],
                "artifact_count": len(artifacts.get("files", [])),
            },
            sort_keys=True,
        )
    )
    return 0


def _plan_command(arguments: argparse.Namespace) -> int:
    tiers = parse_gpu_tiers(arguments.gpu_tiers)
    trainer_module = _validate_trainer_module(arguments.trainer_module)
    manifest_path, manifest, manifest_relative_path = _validate_manifest(
        arguments.manifest
    )
    trainer_args = list(arguments.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]
    normalized = normalize_trainer_args(
        trainer_args,
        remote_manifest=f"{{SOURCE_ROOT}}/{manifest_relative_path}",
        remote_run_dir="{RUN_DIRECTORY}",
    )
    requirements = _requirements(trainer_module, arguments.requirements or ())
    print(
        json.dumps(
            {
                "state": "planned",
                "auth_required_for": "submit/status/logs/watch/stop/retrieve",
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": manifest.sha256,
                },
                "trainer_module": trainer_module,
                "trainer_args": list(normalized),
                "requirements": requirements,
                "gpu_tiers": list(tiers),
                "run_directory": str(Path(arguments.run_dir).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--auth", default="adc", help="colabctl auth mode")
    parser.add_argument("--colab-bin", default="colab", help="Google Colab CLI binary")


def _add_submission_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
    parser.add_argument(
        "--gpu-tiers",
        default=",".join(DEFAULT_GPU_TIERS),
        help="ordered escalation tiers, starting with T4",
    )
    parser.add_argument("--trainer-module", default=DEFAULT_TRAINER_MODULE)
    parser.add_argument("--job-name")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--req", dest="requirements", action="append")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--include-frame-store", action="store_true")
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="trainer arguments after `--`; local paths are rewritten remotely",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-colab")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser(
        "plan", help="validate a job without building or authenticating"
    )
    _add_submission_arguments(plan)
    submit = commands.add_parser(
        "submit", help="package and submit a detached colabctl job"
    )
    _add_submission_arguments(submit)
    _add_connection_arguments(submit)

    for name, help_text in (
        ("status", "read one job status"),
        ("logs", "read one job's bounded log tail"),
        ("watch", "wait for completion and retrieve artifacts"),
        ("stop", "cancel a job and release its runtime"),
        ("retrieve", "download and verify terminal-job artifacts"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("job")
        command.add_argument("--job-root", type=Path, default=DEFAULT_JOB_ROOT)
        _add_connection_arguments(command)
        if name == "logs":
            command.add_argument("--tail", type=int, default=200)
        if name == "watch":
            command.add_argument("--poll-seconds", type=float, default=15.0)
            command.add_argument("--keep-session", action="store_true")
        if name == "stop":
            command.add_argument("--keep-session", action="store_true")

    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "plan":
            return _plan_command(arguments)
        if arguments.command == "submit":
            trainer_args = list(arguments.trainer_args)
            if trainer_args and trainer_args[0] == "--":
                trainer_args = trainer_args[1:]
            paths, record = asyncio.run(
                submit_job(
                    run_directory=arguments.run_dir,
                    manifest_path=arguments.manifest,
                    job_root=arguments.job_root,
                    gpu_tiers=parse_gpu_tiers(arguments.gpu_tiers),
                    trainer_module=arguments.trainer_module,
                    trainer_args=trainer_args,
                    requirements=arguments.requirements or (),
                    job_name=arguments.job_name,
                    timeout=arguments.timeout,
                    resume_from=arguments.resume_from,
                    include_frame_store=arguments.include_frame_store,
                    auth_mode=arguments.auth,
                    colab_bin=arguments.colab_bin,
                )
            )
            print(
                json.dumps(
                    {
                        "job_id": record["job_id"],
                        "job_directory": str(paths.root),
                        "state": "running",
                        "auth_used": True,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "status":
            return asyncio.run(
                _status_command(
                    arguments.job,
                    job_root=arguments.job_root,
                    auth_mode=arguments.auth,
                    colab_bin=arguments.colab_bin,
                )
            )
        if arguments.command == "logs":
            if arguments.tail < 0:
                raise ValueError("log tail must not be negative")
            return asyncio.run(
                _logs_command(
                    arguments.job,
                    job_root=arguments.job_root,
                    tail=arguments.tail,
                    auth_mode=arguments.auth,
                    colab_bin=arguments.colab_bin,
                )
            )
        if arguments.command == "watch":
            return asyncio.run(
                _watch_command(
                    arguments.job,
                    job_root=arguments.job_root,
                    poll_seconds=arguments.poll_seconds,
                    keep_session=arguments.keep_session,
                    auth_mode=arguments.auth,
                    colab_bin=arguments.colab_bin,
                )
            )
        if arguments.command == "stop":
            return asyncio.run(
                _stop_command(
                    arguments.job,
                    job_root=arguments.job_root,
                    keep_session=arguments.keep_session,
                    auth_mode=arguments.auth,
                    colab_bin=arguments.colab_bin,
                )
            )
        if arguments.command == "retrieve":
            return asyncio.run(
                _retrieve_command(
                    arguments.job,
                    job_root=arguments.job_root,
                    auth_mode=arguments.auth,
                    colab_bin=arguments.colab_bin,
                )
            )
    except (ColabCtlBatchError, OSError, ValueError) as error:
        print(f"dodge-ng-colab: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"dodge-ng-colab: colabctl operation failed: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
