from __future__ import annotations

import asyncio
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from dodge.ng.colabctl_batch import (
    ColabCtlBatchError,
    _build_remote_script,
    _create_payload,
    _extract_artifacts,
    _submit_remote,
    create_resume_bundle,
    create_source_bundle,
    job_paths,
    main,
    normalize_trainer_args,
    parse_gpu_tiers,
)


def test_gpu_policy_is_ordered_and_t4_first() -> None:
    assert parse_gpu_tiers("t4,l4,A100") == ("T4", "L4", "A100")
    with pytest.raises(ValueError, match="start with T4"):
        parse_gpu_tiers("L4,T4")
    with pytest.raises(ValueError, match="cheaper"):
        parse_gpu_tiers("T4,A100,L4")
    with pytest.raises(ValueError, match="repeat"):
        parse_gpu_tiers("T4,T4")


def test_v133_colab_cli_kernel_client_api_is_available() -> None:
    client = pytest.importorskip("jupyter_kernel_client")
    assert hasattr(client, "KernelClient")
    assert hasattr(client, "JupyterSubprotocol")


def test_v132_capacity_error_advances_declared_gpu_ladder(monkeypatch) -> None:
    colabctl = pytest.importorskip("colabctl")

    class FakeAccelerator:
        def __init__(self, value):
            self.value = value

    class FakeUnavailableError(Exception):
        pass

    class FakeClient:
        def __init__(self, **_kwargs):
            self.transport = object()

    class FakeBackend:
        def __init__(self):
            self.tiers = []

        async def submit(self, spec):
            self.tiers.append(spec["accelerator"].value)
            if len(self.tiers) == 1:
                raise colabctl.TooManyAssignmentsError("HTTP 412")
            return SimpleNamespace(id="job-a100")

        async def aclose(self):
            return None

    backend = FakeBackend()

    class FakeStateStore:
        def get_job(self, job_id):
            return {"id": job_id}

    monkeypatch.setattr(
        "dodge.ng.colabctl_batch._load_colabctl",
        lambda: (
            FakeAccelerator,
            FakeUnavailableError,
            FakeClient,
            lambda _transport: backend,
            dict,
            FakeStateStore,
        ),
    )

    info, attempts, stored = asyncio.run(
        _submit_remote(
            Path("job.py"),
            gpu_tiers=("T4", "A100"),
            requirements=(),
            timeout=None,
            session_name="session",
            auth_mode="adc",
            colab_bin="colab",
        )
    )

    assert info.id == "job-a100"
    assert stored == {"id": "job-a100"}
    assert backend.tiers == ["T4", "A100"]
    assert attempts[0]["state"] == "capacity"
    assert attempts[1] == {"tier": "A100", "state": "submitted"}


def test_v134_launch_error_releases_owned_session(monkeypatch) -> None:
    pytest.importorskip("colabctl")

    class FakeAccelerator:
        def __init__(self, value):
            self.value = value

    class FakeUnavailableError(Exception):
        pass

    class FakeTransport:
        def __init__(self):
            self.stopped = []

        async def stop(self, name):
            self.stopped.append(name)

    transport = FakeTransport()

    class FakeClient:
        def __init__(self, **_kwargs):
            self.transport = transport

    class FakeBackend:
        async def submit(self, _spec):
            raise RuntimeError("detached launch failed")

        async def aclose(self):
            return None

    backend = FakeBackend()

    monkeypatch.setattr(
        "dodge.ng.colabctl_batch._load_colabctl",
        lambda: (
            FakeAccelerator,
            FakeUnavailableError,
            FakeClient,
            lambda _transport: backend,
            dict,
            lambda: None,
        ),
    )

    with pytest.raises(RuntimeError, match="detached launch failed"):
        asyncio.run(
            _submit_remote(
                Path("job.py"),
                gpu_tiers=("A100",),
                requirements=(),
                timeout=None,
                session_name="owned-session",
                auth_mode="adc",
                colab_bin="colab",
            )
        )

    assert transport.stopped == ["owned-session"]


def test_v135_lifecycle_commands_do_not_require_trainer_args(monkeypatch) -> None:
    async def fake_status(*_args, **_kwargs):
        return 0

    monkeypatch.setattr("dodge.ng.colabctl_batch._status_command", fake_status)
    assert main(["status", "colab-job"]) == 0


def test_trainer_paths_are_rewritten_without_touching_other_arguments() -> None:
    result = normalize_trainer_args(
        (
            "--trials",
            "1",
            "--manifest=local.json",
            "--run-dir",
            "local-run",
        ),
        remote_manifest="{SOURCE_ROOT}/context/kits/dodge-ng/ng-v1.json",
        remote_run_dir="{RUN_DIRECTORY}",
    )
    assert result[:2] == ("--trials", "1")
    assert result[-4:] == (
        "--manifest",
        "{SOURCE_ROOT}/context/kits/dodge-ng/ng-v1.json",
        "--run-dir",
        "{RUN_DIRECTORY}",
    )


def test_source_bundle_excludes_generated_trees(tmp_path: Path) -> None:
    project = tmp_path / "project"
    for directory in (
        project / "src" / "dodge",
        project / "native",
        project / "context" / "kits" / "dodge-ng",
    ):
        directory.mkdir(parents=True)
    (project / "src" / "dodge" / "trainer.py").write_text("print(1)\n")
    (project / "native" / "Cargo.toml").write_text("[workspace]\n")
    (project / "context" / "kits" / "dodge-ng" / "ng-v1.json").write_text("{}\n")
    (project / "pyproject.toml").write_text("[project]\n")
    (project / "uv.lock").write_text("version = 1\n")
    generated = project / "history" / "old.txt"
    generated.parent.mkdir()
    generated.write_text("do not bundle\n")

    count, total_bytes, digest = create_source_bundle(
        project, tmp_path / "source.tar.gz"
    )

    assert count == 5
    assert total_bytes > 0
    assert len(digest) == 64
    with tarfile.open(tmp_path / "source.tar.gz", "r:gz") as archive:
        names = archive.getnames()
    assert "source/src/dodge/trainer.py" in names
    assert "source/history/old.txt" not in names


def test_resume_bundle_is_small_and_path_preserving(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "trial-0000").mkdir(parents=True)
    (run / "study.db").write_bytes(b"study")
    (run / "trial-0000" / "checkpoint.pt").write_bytes(b"checkpoint")
    (run / ".pixel-frames.u8").write_bytes(b"large-frame-store")
    (run / "__pycache__").mkdir()
    (run / "__pycache__" / "ignored.pyc").write_bytes(b"cache")

    records = create_resume_bundle(run, tmp_path / "resume.tar.gz")

    assert {item["path"] for item in records} == {
        "study.db",
        "trial-0000/checkpoint.pt",
    }
    with tarfile.open(tmp_path / "resume.tar.gz", "r:gz") as archive:
        assert set(archive.getnames()) == {
            "study.db",
            "trial-0000/checkpoint.pt",
        }


def test_remote_script_is_self_contained_and_compiles(tmp_path: Path) -> None:
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"source")
    wheel = tmp_path / "dodge_native-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl"
    wheel.write_bytes(b"wheel")
    payload = tmp_path / "payload.tar.gz"
    _create_payload(source, wheel, payload)
    script = tmp_path / "job.py"

    _build_remote_script(
        payload,
        submission_id="submission-1",
        payload_sha256="payload-sha",
        source_sha256="source-sha",
        native_wheel_sha256="wheel-sha",
        resume_sha256=None,
        manifest_relative_path="context/kits/dodge-ng/ng-v1.json",
        manifest_sha256="manifest-sha",
        trainer_module="dodge.ng.hpo",
        trainer_args=(
            "--manifest",
            "{SOURCE_ROOT}/context/kits/dodge-ng/ng-v1.json",
            "--run-dir",
            "{RUN_DIRECTORY}",
        ),
        job_identity_sha256="identity-sha",
        include_frame_store=False,
        output_path=script,
    )

    compile(script.read_text(encoding="utf-8"), str(script), "exec")
    text = script.read_text(encoding="utf-8")
    assert "colabctl" not in text
    assert 'argument.replace("{SOURCE_ROOT}"' in text
    assert "PAYLOAD_B64" in text
    assert "manifest_body" in text
    assert 'SOURCE_PACKAGE_ROOT = SOURCE_ROOT / "src"' in text


def test_payload_preserves_valid_wheel_filename(tmp_path: Path) -> None:
    source = tmp_path / "source.tar.gz"
    source.write_bytes(b"source")
    wheel = tmp_path / "dodge_native-0.1.0-cp311-abi3-manylinux_2_34_x86_64.whl"
    wheel.write_bytes(b"wheel")
    payload = tmp_path / "payload.tar.gz"

    _create_payload(source, wheel, payload)

    with tarfile.open(payload, "r:gz") as archive:
        assert archive.getnames() == ["source.tar.gz", wheel.name]


def test_artifact_extraction_verifies_identity_and_is_idempotent(
    tmp_path: Path,
) -> None:
    identity = {
        "submission_id": "submission-1",
        "job_identity_sha256": "identity-sha",
        "manifest_sha256": "manifest-sha",
        "source_bundle_sha256": "source-sha",
        "native_wheel_sha256": "wheel-sha",
        "payload_sha256": "payload-sha",
    }
    data = b"metrics\n"
    manifest = {**identity, "files": [{"path": "metrics.jsonl", "sha256": ""}]}
    import hashlib

    manifest["files"][0]["sha256"] = hashlib.sha256(data).hexdigest()
    encoded = json.dumps(manifest, sort_keys=True).encode()
    archive = tmp_path / "artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        file_info = tarfile.TarInfo("metrics.jsonl")
        file_info.size = len(data)
        output.addfile(file_info, io.BytesIO(data))
        manifest_info = tarfile.TarInfo("artifact-manifest.json")
        manifest_info.size = len(encoded)
        output.addfile(manifest_info, io.BytesIO(encoded))

    destination = tmp_path / "returned"
    returned = _extract_artifacts(
        archive,
        destination,
        expected_submission_id="submission-1",
        expected_job_identity_sha256="identity-sha",
        expected_manifest_sha256="manifest-sha",
        expected_source_sha256="source-sha",
        expected_native_wheel_sha256="wheel-sha",
        expected_payload_sha256="payload-sha",
    )
    repeated = _extract_artifacts(
        archive,
        destination,
        expected_submission_id="submission-1",
        expected_job_identity_sha256="identity-sha",
        expected_manifest_sha256="manifest-sha",
        expected_source_sha256="source-sha",
        expected_native_wheel_sha256="wheel-sha",
        expected_payload_sha256="payload-sha",
    )

    assert returned["archive_sha256"] == repeated["archive_sha256"]
    assert (destination / "metrics.jsonl").read_bytes() == data

    with pytest.raises(ColabCtlBatchError, match="manifest_sha256"):
        _extract_artifacts(
            archive,
            tmp_path / "wrong",
            expected_submission_id="submission-1",
            expected_job_identity_sha256="identity-sha",
            expected_manifest_sha256="wrong",
            expected_source_sha256="source-sha",
            expected_native_wheel_sha256="wheel-sha",
            expected_payload_sha256="payload-sha",
        )


def test_job_paths_reject_traversal() -> None:
    with pytest.raises(ValueError, match="safe path"):
        job_paths(Path("/tmp/jobs"), "../escape")
