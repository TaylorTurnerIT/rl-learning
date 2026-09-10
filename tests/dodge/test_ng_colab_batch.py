from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest
import torch

from dodge.control import PROJECT_ROOT
from dodge.ng.colab_batch import (
    ColabBatchController,
    ColabBatchError,
    CommandResult,
    JobPaths,
    _allocation_may_escalate,
    _atomic_write_json,
    _session_exists,
    _spawn_detached_controller,
    _write_remote_scripts,
    create_artifact_archive,
    create_job,
    create_resume_bundle,
    extract_artifact_archive,
    extract_recovery_archive,
    load_job,
    load_status,
    normalize_trainer_args,
    parse_gpu_tiers,
    prepare_job,
    transition_status,
)
from dodge.ng.colab_worker import create_recovery_archive, run_worker


def _job(tmp_path: Path, **kwargs: object) -> tuple[JobPaths, dict[str, object]]:
    return create_job(
        project_root=PROJECT_ROOT,
        local_run_dir=tmp_path / "returned-run",
        job_root=tmp_path / "jobs",
        job_id="test-job",
        gpu_tiers=("T4", "L4"),
        trainer_args=("--total-steps", "64"),
        **kwargs,
    )


def _write_required_run(run_directory: Path, *, stopped: bool = False) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "checkpoint-latest.pt").write_bytes(b"checkpoint")
    (run_directory / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (run_directory / "run.json").write_text(
        json.dumps({"stopped_early": stopped}), encoding="utf-8"
    )
    dashboard = run_directory / "dashboard"
    dashboard.mkdir()
    (dashboard / "status.json").write_text(
        json.dumps({"state": "stopped" if stopped else "completed", "step": 9}),
        encoding="utf-8",
    )


def test_gpu_policy_and_remote_argument_normalization() -> None:
    assert parse_gpu_tiers("t4,l4,A100") == ("T4", "L4", "A100")
    with pytest.raises(ValueError, match="repeat"):
        parse_gpu_tiers("T4,T4")
    with pytest.raises(ValueError, match="unsupported"):
        parse_gpu_tiers("V100")

    result = normalize_trainer_args(
        (
            "--manifest=local.json",
            "--run-dir",
            "local-run",
            "--device",
            "cuda",
        ),
        remote_manifest="/content/source/manifest.json",
        remote_run_dir="/content/run",
    )
    assert result.count("--manifest") == 1
    assert result[result.index("--manifest") + 1] == "/content/source/manifest.json"
    assert result[result.index("--run-dir") + 1] == "/content/run"
    assert result[result.index("--native-pixel-boundary") + 1] == "fast"
    with pytest.raises(ValueError, match="device"):
        normalize_trainer_args(
            ("--device", "cpu"), remote_manifest="remote", remote_run_dir="run"
        )
    with pytest.raises(ValueError, match="must not be repeated"):
        normalize_trainer_args(
            ("--device", "auto", "--device=cuda"),
            remote_manifest="remote",
            remote_run_dir="run",
        )
    assert not _session_exists(
        CommandResult(("colab", "status"), 0, "Session 'missing' not found.", "")
    )


@pytest.mark.parametrize(
    "detail",
    (
        "TooManyAssignmentsError: HTTP 412 Precondition Failed",
        "status code 412: accelerator=T4 capacity is temporarily unavailable",
    ),
)
def test_colab_capacity_assignment_errors_are_escalatable(detail: str) -> None:
    assert _allocation_may_escalate(detail)
    assert not _allocation_may_escalate("authentication failed: invalid credentials")


def test_job_state_is_durable_and_monotonic(tmp_path: Path) -> None:
    paths, job = _job(tmp_path)

    assert job["manifest_sha256"]
    assert load_status(paths)["state"] == "planned"
    transition_status(paths, "provisioning")
    transition_status(paths, "running", step=4)
    transition_status(paths, "stopping")
    with pytest.raises(ColabBatchError, match="regression"):
        transition_status(paths, "running")
    transition_status(paths, "cancelled")
    with pytest.raises(ColabBatchError, match="terminal"):
        transition_status(paths, "failed")


def test_prepare_job_records_source_and_native_wheel_hashes(tmp_path: Path) -> None:
    paths, _ = _job(tmp_path)

    def fake_wheel(_project_root: Path, output_directory: Path) -> Path:
        output_directory.mkdir(parents=True)
        wheel = output_directory / "dodge_native-test.whl"
        wheel.write_bytes(b"native-wheel")
        return wheel

    prepared = prepare_job(paths, build_wheel=fake_wheel)

    source = prepared["source"]
    assert source["bundle_sha256"]
    assert source["wheel"]["sha256"]
    assert paths.source_bundle.is_file()
    with tarfile.open(paths.source_bundle, "r:gz") as archive:
        names = archive.getnames()
    assert "source/src/dodge/ng/colab_worker.py" in names
    assert not any(name.startswith("source/history/") for name in names)
    _write_remote_scripts(paths, prepared)
    for script in (
        paths.bootstrap,
        paths.poller,
        paths.stopper,
        paths.packer,
        paths.recovery_exporter,
    ):
        compile(script.read_text(encoding="utf-8"), str(script), "exec")
    assert "artifacts-controller.tar.gz" in paths.packer.read_text(encoding="utf-8")


def test_artifact_round_trip_is_allowlisted_and_hash_verified(tmp_path: Path) -> None:
    remote_run = tmp_path / "remote-run"
    _write_required_run(remote_run)
    (remote_run / ".pixel-frames.u8").write_bytes(b"large")
    (remote_run / "unrelated.txt").write_text("no", encoding="utf-8")
    archive = tmp_path / "artifacts.tar.gz"

    inventory = create_artifact_archive(
        remote_run,
        archive,
        job_id="job-1",
        manifest_sha256="manifest-1",
        job_identity_sha256="identity-1",
        source_bundle_sha256="source-1",
        native_wheel_sha256="wheel-1",
    )
    returned = tmp_path / "returned"
    extracted = extract_artifact_archive(
        archive,
        returned,
        expected_job_id="job-1",
        expected_manifest_sha256="manifest-1",
        expected_job_identity_sha256="identity-1",
        expected_source_bundle_sha256="source-1",
        expected_native_wheel_sha256="wheel-1",
    )

    assert inventory["complete"] is True
    assert extracted["complete"] is True
    assert (returned / "checkpoint-latest.pt").is_file()
    assert not (returned / ".pixel-frames.u8").exists()
    assert not (returned / "unrelated.txt").exists()
    with pytest.raises(ColabBatchError, match="manifest hash"):
        extract_artifact_archive(
            archive,
            tmp_path / "wrong",
            expected_job_id="job-1",
            expected_manifest_sha256="wrong",
            expected_job_identity_sha256="identity-1",
            expected_source_bundle_sha256="source-1",
            expected_native_wheel_sha256="wheel-1",
        )


def test_artifact_extraction_rejects_links(tmp_path: Path) -> None:
    archive = tmp_path / "hostile.tar.gz"
    manifest = {
        "job_id": "job-1",
        "manifest_sha256": "manifest-1",
        "files": [],
        "required": [],
    }
    encoded = json.dumps(manifest).encode()
    with tarfile.open(archive, "w:gz") as output:
        info = tarfile.TarInfo("artifact-manifest.json")
        info.size = len(encoded)
        output.addfile(info, io.BytesIO(encoded))
        link = tarfile.TarInfo("escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "/tmp"
        output.addfile(link)

    with pytest.raises(ColabBatchError, match="non-regular"):
        extract_artifact_archive(
            archive,
            tmp_path / "returned",
            expected_job_id="job-1",
            expected_manifest_sha256="manifest-1",
            expected_job_identity_sha256="identity-1",
            expected_source_bundle_sha256="source-1",
            expected_native_wheel_sha256="wheel-1",
        )


class _ProvisionCLI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def call(self, arguments: list[str], *, timeout: float) -> CommandResult:
        del timeout
        command = tuple(arguments)
        self.calls.append(command)
        if command[0] == "status":
            return CommandResult(("colab", *command), 0, "Session not found.", "")
        ok = command[0] != "status" and not (
            command[0] == "new" and command[-1] == "T4"
        )
        detail = "ok" if ok else "GPU capacity temporarily unavailable"
        return CommandResult(("colab", *command), 0 if ok else 1, detail, "")


class _PollCLI:
    def __init__(self, status: dict[str, object]) -> None:
        self.status = status

    def call(self, arguments: list[str], *, timeout: float) -> CommandResult:
        del timeout
        command = tuple(arguments)
        return CommandResult(("colab", *command), 0, json.dumps(self.status) + "\n", "")


def test_gpu_allocation_escalates_in_order_and_persists_session(tmp_path: Path) -> None:
    paths, _ = _job(tmp_path)
    cli = _ProvisionCLI()

    controller = ColabBatchController(paths, cli=cli)
    controller._provision()

    job = load_job(paths)
    assert job["gpu_policy"]["selected_tier"] == "L4"
    assert [item["tier"] for item in job["gpu_policy"]["attempts"]] == ["T4", "L4"]
    assert job["session"]["name"] == controller.session_name
    assert any(call[0] == "stop" for call in cli.calls)


def test_existing_session_is_never_cleaned_up_by_failed_submit(tmp_path) -> None:
    paths, _ = _job(tmp_path)

    class CollisionCLI(_ProvisionCLI):
        def call(self, arguments, *, timeout):
            if arguments[0] == "status":
                return CommandResult(tuple(arguments), 0, "Session exists", "")
            return super().call(arguments, timeout=timeout)

    cli = CollisionCLI()
    with pytest.raises(ColabBatchError, match="refusing to reuse"):
        ColabBatchController(paths, cli=cli).run()
    assert not any(call[0] == "stop" for call in cli.calls)


def test_watch_accepts_stopped_worker_without_regressing_local_state(
    tmp_path: Path,
) -> None:
    paths, job = _job(tmp_path)
    job["job_identity_sha256"] = "identity-1"
    job["session"]["name"] = "dodge-ng-test-job"
    _atomic_write_json(paths.manifest, job)
    transition_status(paths, "provisioning")
    transition_status(paths, "running")
    transition_status(paths, "stopping")
    remote = {
        "state": "stopped",
        "job_id": "test-job",
        "manifest_sha256": job["manifest_sha256"],
        "job_identity_sha256": job["job_identity_sha256"],
        "ownership_nonce": job["session"]["ownership_nonce"],
        "heartbeat_at": 10.0,
        "worker_pid": 99,
        "step": 12,
        "native_steps": 128,
    }

    result = ColabBatchController(paths, cli=_PollCLI(remote))._watch()

    assert result["state"] == "stopped"
    assert load_status(paths)["state"] == "stopping"


def test_watch_fails_closed_on_remote_identity_mismatch(tmp_path: Path) -> None:
    paths, job = _job(tmp_path)
    job["job_identity_sha256"] = "identity-1"
    job["session"]["name"] = "dodge-ng-test-job"
    _atomic_write_json(paths.manifest, job)
    remote = {
        "state": "running",
        "job_id": "another-job",
        "manifest_sha256": job["manifest_sha256"],
        "job_identity_sha256": job["job_identity_sha256"],
        "ownership_nonce": job["session"]["ownership_nonce"],
        "heartbeat_at": 10.0,
    }

    with pytest.raises(ColabBatchError, match="job id"):
        ColabBatchController(paths, cli=_PollCLI(remote))._watch()


def test_detached_controller_is_marked_launching_before_spawn(
    tmp_path: Path, monkeypatch
) -> None:
    paths, _ = _job(tmp_path)

    class FakeProcess:
        pid = 4321

    monkeypatch.setattr(
        "dodge.ng.colab_batch.subprocess.Popen", lambda *a, **k: FakeProcess()
    )

    assert _spawn_detached_controller(paths, keep_session=False) == 4321
    assert load_status(paths)["controller_launching_at"] > 0
    assert '"pid":4321' in paths.events.read_text(encoding="utf-8")


def test_remote_worker_reports_graceful_stop_and_packs_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "source"
    project_root.mkdir()
    run_directory = tmp_path / "run"
    _write_required_run(run_directory, stopped=True)
    status_path = tmp_path / "remote-status.json"

    class FakeProcess:
        pid = 7654

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        "dodge.ng.colab_worker._preflight", lambda *a, **k: {"cuda": True}
    )
    monkeypatch.setattr(
        "dodge.ng.colab_worker.subprocess.Popen", lambda *a, **k: FakeProcess()
    )

    result = run_worker(
        job_id="job-1",
        project_root=project_root,
        run_directory=run_directory,
        status_path=status_path,
        manifest_sha256="manifest-1",
        job_identity_sha256="identity-1",
        source_bundle_sha256="source-1",
        native_wheel_sha256="wheel-1",
        ownership_nonce="owner-1",
        gpu="T4",
        heartbeat_seconds=0.01,
        trainer_arguments=("--device", "cuda"),
        include_frame_store=False,
    )

    assert result == 0
    assert json.loads(status_path.read_text())["state"] == "stopped"
    assert (tmp_path / "artifacts.tar.gz").is_file()


def test_v95_download_failure_preserves_session_for_recovery(tmp_path, monkeypatch):
    paths, _ = _job(tmp_path)
    transition_status(paths, "running", allocation_started=True)
    cli = _ProvisionCLI()
    controller = ColabBatchController(paths, cli=cli)
    for method in ("_auth_preflight", "_provision", "_upload", "_launch"):
        monkeypatch.setattr(controller, method, lambda: None)
    monkeypatch.setattr(controller, "_watch", lambda: {"state": "completed"})
    monkeypatch.setattr(controller, "_download_remote_diagnostics", lambda: True)

    def broken_download(_remote=None):
        raise ColabBatchError("simulated transfer failure")

    monkeypatch.setattr(controller, "_retrieve", broken_download)
    with pytest.raises(ColabBatchError, match="transfer failure"):
        controller.run()
    assert load_status(paths)["state"] == "stopping"
    assert load_status(paths)["recovery_required"] is True
    assert not any(call[0] == "stop" for call in cli.calls)

    # A new supervisor can finish retrieval without restarting training.
    recovered = ColabBatchController(paths, cli=cli)
    monkeypatch.setattr(recovered, "_auth_preflight", lambda: None)
    monkeypatch.setattr(recovered, "_watch", lambda: {"state": "completed"})
    monkeypatch.setattr(recovered, "_retrieve", lambda _remote=None: None)
    monkeypatch.setattr(recovered, "_download_remote_diagnostics", lambda: True)
    monkeypatch.setattr("dodge.ng.colab_batch._session_exists", lambda result: True)
    assert recovered.recover()["state"] == "completed"
    assert load_status(paths)["recovery_required"] is False
    assert sum(call[0] == "stop" for call in cli.calls) == 1


@pytest.mark.parametrize("stage", ["planned", "provisioning", "running"])
def test_v96_stop_request_does_not_race_with_controller_state(tmp_path, stage):
    paths, _ = _job(tmp_path)
    transition_status(paths, stage)
    cli = _ProvisionCLI()
    ColabBatchController(paths, cli=cli).request_stop()
    assert paths.stop_request.is_file()
    assert load_status(paths)["state"] == stage
    assert cli.calls == []


def test_v96_stop_before_allocation_never_allocates(tmp_path):
    paths, _ = _job(tmp_path)
    cli = _ProvisionCLI()
    ColabBatchController(paths, cli=cli).request_stop()
    assert ColabBatchController(paths, cli=cli).run()["state"] == "cancelled"
    assert cli.calls == []


def test_v96_running_stop_uses_validated_remote_identity_immediately(tmp_path):
    paths, job = _job(tmp_path)
    job["job_identity_sha256"] = "identity-1"
    _atomic_write_json(paths.manifest, job)
    remote = {
        "state": "running",
        "job_id": job["job_id"],
        "manifest_sha256": job["manifest_sha256"],
        "job_identity_sha256": "identity-1",
        "ownership_nonce": job["session"]["ownership_nonce"],
        "heartbeat_at": 10.0,
        "worker_pid": 7,
    }
    transition_status(paths, "running", allocation_started=True, remote=remote)
    cli = _ProvisionCLI()
    ColabBatchController(paths, cli=cli).request_stop()
    assert paths.stop_request.is_file()
    assert [call[0] for call in cli.calls] == ["exec"]


def test_v95_installed_receipt_allows_retry_without_remote_calls(tmp_path):
    paths, job = _job(tmp_path)
    job["job_identity_sha256"] = "identity-1"
    job["source"] = {"bundle_sha256": "source-1", "wheel": {"sha256": "wheel-1"}}
    _atomic_write_json(paths.manifest, job)
    remote_run = tmp_path / "remote-run"
    _write_required_run(remote_run)
    create_artifact_archive(
        remote_run,
        paths.downloaded_archive,
        job_id=job["job_id"],
        manifest_sha256=job["manifest_sha256"],
        job_identity_sha256="identity-1",
        source_bundle_sha256="source-1",
        native_wheel_sha256="wheel-1",
    )
    output = Path(job["local_run_dir"])
    extract_artifact_archive(
        paths.downloaded_archive,
        output,
        expected_job_id=job["job_id"],
        expected_manifest_sha256=job["manifest_sha256"],
        expected_job_identity_sha256="identity-1",
        expected_source_bundle_sha256="source-1",
        expected_native_wheel_sha256="wheel-1",
    )
    cli = _ProvisionCLI()
    controller = ColabBatchController(paths, cli=cli)
    controller._retrieve()
    assert cli.calls == []
    assert paths.artifact_manifest.is_file()
    (output / "checkpoint-latest.pt").write_bytes(b"changed")
    with pytest.raises(ColabBatchError, match="installed artifact changed"):
        controller._retrieve()
    assert cli.calls == []


def test_v99_recovery_archive_round_trip_is_resumable(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    snapshot = run / "replay-abc.u8.gz"
    snapshot.write_bytes(b"compressed replay")
    (run / "metrics.jsonl").write_text('{"step":7}\n', encoding="utf-8")
    torch.save(
        {"step": 7, "replay_snapshot": {"path": snapshot.name}},
        run / "checkpoint-latest.pt",
    )
    archive = tmp_path / "recovery.tar"
    created = create_recovery_archive(run, archive, job_identity_sha256="identity-1")
    destination = tmp_path / "recovery" / "latest"
    extracted = extract_recovery_archive(
        archive, destination, expected_job_identity_sha256="identity-1"
    )
    assert created["checkpoint_step"] == extracted["checkpoint_step"] == 7
    assert (destination / "checkpoint-latest.pt").is_file()
    assert (destination / snapshot.name).read_bytes() == b"compressed replay"
    with pytest.raises(ColabBatchError, match="identity"):
        extract_recovery_archive(
            archive, tmp_path / "wrong", expected_job_identity_sha256="wrong"
        )


def test_v99_recovery_export_skips_before_first_checkpoint(tmp_path):
    run = tmp_path / "run"
    run.mkdir()

    result = create_recovery_archive(
        run,
        tmp_path / "recovery.tar",
        job_identity_sha256="identity-1",
    )

    assert result["job_identity_sha256"] == "identity-1"
    assert result["skipped"] == "checkpoint-not-yet-published"
    assert not (tmp_path / "recovery.tar").exists()


def test_v103_controller_ignores_matching_precheckpoint_noop(tmp_path):
    paths, job = _job(tmp_path)
    job["job_identity_sha256"] = "identity-1"
    _atomic_write_json(paths.manifest, job)

    class SkipCLI:
        def call(self, arguments, *, timeout):
            del timeout
            assert arguments[0] == "exec"
            return CommandResult(
                ("colab", *arguments),
                0,
                json.dumps(
                    {
                        "job_identity_sha256": "identity-1",
                        "checkpoint_step": None,
                        "files": [],
                        "skipped": "checkpoint-not-yet-published",
                    }
                ),
                "",
            )

    assert ColabBatchController(paths, cli=SkipCLI())._sync_recovery() is False
    assert not paths.recovery_archive.exists()


def test_v105_unchanged_recovery_boundary_skips_download(tmp_path):
    paths, job = _job(tmp_path)
    job["job_identity_sha256"] = "identity-1"
    job["session"]["name"] = "dodge-ng-test-job"
    _atomic_write_json(paths.manifest, job)
    transition_status(paths, "running")
    manifest = {
        "schema_version": 1,
        "job_identity_sha256": "identity-1",
        "checkpoint_step": 7,
        "files": [
            {"path": "checkpoint-latest.pt", "bytes": 3, "sha256": "checkpoint"},
            {"path": "metrics.jsonl", "bytes": 3, "sha256": "metrics-old"},
            {"path": "replay-abc.u8.gz", "bytes": 6, "sha256": "replay-"},
        ],
    }
    paths.recovery_directory.mkdir(parents=True)
    _atomic_write_json(paths.recovery_directory / "recovery-manifest.json", manifest)
    calls = []

    class UnchangedCLI:
        def call(self, arguments, *, timeout):
            del timeout
            calls.append(arguments)
            assert arguments[0] == "exec"
            return CommandResult(
                ("colab", *arguments),
                0,
                json.dumps(
                    {
                        **manifest,
                        "files": [
                            *manifest["files"][:1],
                            {
                                "path": "metrics.jsonl",
                                "bytes": 3,
                                "sha256": "metrics-new",
                            },
                            *manifest["files"][2:],
                        ],
                    }
                ),
                "",
            )

    assert ColabBatchController(paths, cli=UnchangedCLI())._sync_recovery() is True
    assert len(calls) == 1
    assert not paths.recovery_archive.exists()


def test_v100_resume_bundle_is_minimal_and_bound_to_job(tmp_path):
    run = tmp_path / "prior-run"
    run.mkdir()
    snapshot = run / "replay-resume.u8.gz"
    snapshot.write_bytes(b"replay")
    (run / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    torch.save(
        {"replay_snapshot": {"path": snapshot.name}},
        run / "checkpoint-latest.pt",
    )
    bundle = tmp_path / "resume.tar.gz"
    records = create_resume_bundle(run, bundle)
    assert {item["path"] for item in records} == {
        "checkpoint-latest.pt",
        "metrics.jsonl",
        snapshot.name,
    }
    with tarfile.open(bundle, "r:gz") as archive:
        assert set(archive.getnames()) == {item["path"] for item in records}

    paths, job = _job(tmp_path / "job", resume_from=run)
    assert "--resume" in job["trainer"]["argv"]

    def fake_wheel(_project_root: Path, output_directory: Path) -> Path:
        output_directory.mkdir(parents=True)
        wheel = output_directory / "dodge_native-test.whl"
        wheel.write_bytes(b"wheel")
        return wheel

    prepared = prepare_job(paths, build_wheel=fake_wheel)
    assert prepared["resume"]["bundle_sha256"]
    assert (
        prepared["job_identity"]["resume_bundle_sha256"]
        == prepared["resume"]["bundle_sha256"]
    )
