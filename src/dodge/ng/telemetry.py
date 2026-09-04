"""Best-effort run telemetry and file-backed training controls."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final

TELEMETRY_VERSION: Final[int] = 1
VALID_CONTROLS: Final[frozenset[str]] = frozenset({"pause", "resume", "stop"})


class DashboardTelemetry:
    """Publish latest run state without making the trainer wait on I/O."""

    def __init__(self, run_directory: Path) -> None:
        self.run_directory = Path(run_directory)
        self.dashboard_directory = self.run_directory / "dashboard"
        self.dashboard_directory.mkdir(parents=True, exist_ok=True)
        self.status_path = self.dashboard_directory / "status.json"
        self.metrics_path = self.dashboard_directory / "metrics.jsonl"
        self.control_path = self.dashboard_directory / "control.json"
        self._lock = threading.Lock()
        self._latest: dict[str, object] | None = None
        self._wake = threading.Event()
        self._closed = False
        self._last_control_id: str | None = None
        self._control_mtime_ns = -1
        self._writer = threading.Thread(
            target=self._write_loop,
            name="dodge-dashboard-telemetry",
            daemon=True,
        )
        self._writer.start()

    def publish(self, payload: Mapping[str, object]) -> None:
        """Replace pending state; this call never waits for the writer."""
        item = dict(payload)
        item["telemetry_version"] = TELEMETRY_VERSION
        item["updated_at"] = time.time()
        with self._lock:
            if self._closed:
                return
            self._latest = item
            self._wake.set()

    def consume_control(self) -> str | None:
        """Consume one new valid control command, if the dashboard wrote one."""
        try:
            stat = self.control_path.stat()
        except OSError:
            return None
        if stat.st_mtime_ns < self._control_mtime_ns:
            return None
        self._control_mtime_ns = stat.st_mtime_ns
        try:
            payload = json.loads(self.control_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        command = payload.get("command")
        command_id = payload.get("id")
        if not isinstance(command, str) or command not in VALID_CONTROLS:
            return None
        if not isinstance(command_id, str) or command_id == self._last_control_id:
            return None
        self._last_control_id = command_id
        return command

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._wake.set()
        self._writer.join(timeout=2.0)

    def __enter__(self) -> DashboardTelemetry:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _write_loop(self) -> None:
        while True:
            self._wake.wait()
            while True:
                with self._lock:
                    item = self._latest
                    self._latest = None
                    closed = self._closed
                    if item is None:
                        self._wake.clear()
                if item is None:
                    if closed:
                        return
                    break
                try:
                    _atomic_write_json(self.status_path, item)
                    record = item.get("record")
                    if isinstance(record, dict):
                        with self.metrics_path.open("a", encoding="utf-8") as stream:
                            stream.write(
                                json.dumps(
                                    record,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                except (OSError, TypeError, ValueError):
                    # Telemetry must never turn a successful training step into
                    # a failed one because its optional files are unavailable.
                    continue


def issue_control(run_directory: Path, command: str) -> None:
    if command not in VALID_CONTROLS:
        raise ValueError(f"unsupported dashboard control: {command}")
    path = Path(run_directory) / "dashboard" / "control.json"
    _atomic_write_json(
        path,
        {
            "version": TELEMETRY_VERSION,
            "id": str(time.time_ns()),
            "command": command,
        },
    )


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
