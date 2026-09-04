from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dodge.ng.dashboard import RunInspector
from dodge.ng.replay import _write_frame
from dodge.ng.telemetry import DashboardTelemetry, issue_control


def test_telemetry_keeps_latest_state_and_writes_metrics(tmp_path: Path) -> None:
    telemetry = DashboardTelemetry(tmp_path / "run")
    telemetry.publish({"state": "running", "step": 1, "record": {"step": 1}})
    _wait_for_file(telemetry.status_path)
    telemetry.publish({"state": "running", "step": 2, "record": {"step": 2}})
    telemetry.close()

    status = json.loads(telemetry.status_path.read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in telemetry.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert status["step"] == 2
    assert metrics[-1]["step"] == 2


def test_dashboard_control_is_consumed_once(tmp_path: Path) -> None:
    telemetry = DashboardTelemetry(tmp_path / "run")
    issue_control(telemetry.run_directory, "pause")
    assert telemetry.consume_control() == "pause"
    assert telemetry.consume_control() is None
    issue_control(telemetry.run_directory, "resume")
    assert telemetry.consume_control() == "resume"
    telemetry.close()


def test_run_inspector_rejects_replay_path_escape(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    replay_directory = run_directory / "dashboard" / "replays"
    replay_directory.mkdir(parents=True)
    (replay_directory / "frame.bin").write_bytes(b"frame")
    inspector = RunInspector(run_directory)

    assert (
        inspector.resolve_replay("frame.bin")
        == (replay_directory / "frame.bin").resolve()
    )
    assert inspector.resolve_replay("../frame.bin") is None
    assert inspector.resolve_replay("%2e%2e/frame.bin") is None


def test_replay_frame_writer_validates_palette_frame() -> None:
    pixels = np.zeros((1, 128, 128), dtype=np.uint8)
    result = SimpleNamespace(pixels=pixels)
    from io import BytesIO

    stream = BytesIO()
    assert _write_frame(stream, result) == 1
    assert len(stream.getvalue()) == 128 * 128


def _wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not path.is_file():
        time.sleep(0.01)
    assert path.is_file()
