"""Minimal WebSocket dashboard for a waypoint DQN run."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from dodge.ng.telemetry import issue_control

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "history" / "dodge" / "ng"
DASHBOARD_PAGE = Path(__file__).with_name("dashboard.html")


class RunInspector:
    def __init__(self, run_directory: Path) -> None:
        self.run_directory = Path(run_directory).resolve()
        self.dashboard_directory = self.run_directory / "dashboard"
        self.replay_directory = self.dashboard_directory / "replays"
        self._metrics_signature: tuple[int, int] | None = None
        self._metrics: list[dict[str, object]] = []

    def snapshot(self, *, replay_running: bool) -> dict[str, object]:
        status = _read_json(self.dashboard_directory / "status.json")
        if status is None:
            status = self._fallback_status()
        return {
            "type": "state",
            "status": status,
            "history": self._read_metrics(),
            "replays": self._read_replays(),
            "replay_running": replay_running,
        }

    def default_seed(self) -> int | None:
        status = _read_json(self.dashboard_directory / "status.json")
        if isinstance(status, dict):
            seeds = status.get("training_seeds")
            if isinstance(seeds, list) and seeds:
                return _integer_seed(seeds[0])
        run = _read_json(self.run_directory / "run.json")
        if isinstance(run, dict):
            evaluation = run.get("final_training_evaluation")
            if isinstance(evaluation, dict):
                seeds = evaluation.get("seeds")
                if isinstance(seeds, list) and seeds:
                    return _integer_seed(seeds[0])
        return None

    def resolve_replay(self, name: str) -> Path | None:
        if not name or Path(name).name != name:
            return None
        root = self.replay_directory.resolve()
        candidate = (root / unquote(name)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        return candidate

    def _fallback_status(self) -> dict[str, object]:
        run = _read_json(self.run_directory / "run.json")
        if not isinstance(run, dict):
            return {"state": "waiting", "step": 0, "total_steps": 0}
        config = run.get("config")
        config = config if isinstance(config, dict) else {}
        return {
            "state": "stopped" if run.get("stopped_early") else "completed",
            "step": run.get("updates_completed", 0),
            "total_steps": config.get("total_steps", 0),
            "native_steps": run.get("native_steps", 0),
            "replay_size": None,
            "best_inner": run.get("best_inner"),
            "config": config,
            "manifest_sha256": run.get("manifest_sha256"),
            "record": None,
            "final_training": _summary(run.get("final_training_evaluation")),
            "final_holdout": _summary(run.get("final_evaluation")),
        }

    def _read_metrics(self) -> list[dict[str, object]]:
        path = self.dashboard_directory / "metrics.jsonl"
        if not path.is_file():
            path = self.run_directory / "metrics.jsonl"
        try:
            stat = path.stat()
        except OSError:
            return self._metrics[-500:]
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self._metrics_signature:
            return self._metrics[-500:]
        metrics: list[dict[str, object]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                if isinstance(value, dict):
                    metrics.append(value)
        except (OSError, json.JSONDecodeError):
            metrics = self._metrics
        self._metrics_signature = signature
        self._metrics = metrics
        return metrics[-500:]

    def _read_replays(self) -> list[dict[str, object]]:
        if not self.replay_directory.is_dir():
            return []
        result: list[dict[str, object]] = []
        for path in sorted(self.replay_directory.glob("*.json")):
            metadata = _read_json(path)
            if not isinstance(metadata, dict):
                continue
            frame_file = metadata.get("frame_file")
            if not isinstance(frame_file, str):
                continue
            frame_path = self.resolve_replay(frame_file)
            if frame_path is None:
                continue
            item = dict(metadata)
            item["url"] = f"/replay/{frame_file}"
            result.append(item)
        return result


class DashboardApplication:
    def __init__(self, run_directory: Path) -> None:
        self.inspector = RunInspector(run_directory)
        self.run_directory = self.inspector.run_directory
        self._replay_process: subprocess.Popen[bytes] | None = None
        self._replay_lock = threading.Lock()

    def run(self, host: str, http_port: int, websocket_port: int) -> None:
        handler = _http_handler(self)
        http_server = ThreadingHTTPServer((host, http_port), handler)
        http_thread = threading.Thread(
            target=http_server.serve_forever,
            name="dodge-dashboard-http",
            daemon=True,
        )
        http_thread.start()
        display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        print(f"http://{display_host}:{http_port}/?ws={websocket_port}")
        try:
            asyncio.run(self._run_websocket(host, websocket_port))
        except KeyboardInterrupt:
            pass
        finally:
            http_server.shutdown()
            http_server.server_close()
            http_thread.join(timeout=2.0)

    async def _run_websocket(self, host: str, port: int) -> None:
        try:
            from websockets.asyncio.server import serve
        except ModuleNotFoundError as error:
            raise RuntimeError("install the `websockets` dependency") from error
        async with serve(self._client, host, port, max_size=2 * 1024 * 1024):
            await asyncio.Future()

    async def _client(self, connection: object) -> None:
        from websockets.exceptions import ConnectionClosed

        last_payload = ""
        while True:
            snapshot = self.inspector.snapshot(replay_running=self._replay_running())
            encoded = json.dumps(snapshot, separators=(",", ":"))
            if encoded != last_payload:
                await connection.send(encoded)  # type: ignore[attr-defined]
                last_payload = encoded
            try:
                message = await asyncio.wait_for(  # type: ignore[attr-defined]
                    connection.recv(),  # type: ignore[attr-defined]
                    timeout=0.25,
                )
            except TimeoutError:
                continue
            except ConnectionClosed:
                return
            if isinstance(message, str):
                try:
                    parsed = json.loads(message)
                except json.JSONDecodeError:
                    reply = {"type": "error", "message": "invalid JSON"}
                else:
                    reply = self._message(parsed)
                await connection.send(  # type: ignore[attr-defined]
                    json.dumps(reply, separators=(",", ":"))
                )

    def _message(self, message: object) -> dict[str, object]:
        if not isinstance(message, dict):
            return {"type": "error", "message": "invalid message"}
        message_type = message.get("type")
        if message_type == "control":
            command = message.get("command")
            if not isinstance(command, str):
                return {"type": "error", "message": "invalid control"}
            try:
                issue_control(self.run_directory, command)
            except (OSError, ValueError) as error:
                return {"type": "error", "message": str(error)}
            return {"type": "ack", "command": command}
        if message_type == "replay":
            seed = message.get("seed", self.inspector.default_seed())
            if not isinstance(seed, int) or isinstance(seed, bool):
                return {"type": "error", "message": "invalid seed"}
            try:
                started = self._start_replay(seed)
            except (OSError, ValueError, RuntimeError) as error:
                return {"type": "error", "message": str(error)}
            return {"type": "replay", "started": started, "seed": seed}
        return {"type": "error", "message": "invalid message type"}

    def _start_replay(self, seed: int) -> bool:
        with self._replay_lock:
            if self._replay_process is not None:
                if self._replay_process.poll() is None:
                    return False
                self._replay_process = None
            log_path = self.run_directory / "dashboard" / "replay.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab") as log:
                self._replay_process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "dodge.ng.replay",
                        "--run-dir",
                        str(self.run_directory),
                        "--seed",
                        str(seed),
                    ],
                    cwd=PROJECT_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        return True

    def _replay_running(self) -> bool:
        with self._replay_lock:
            return (
                self._replay_process is not None and self._replay_process.poll() is None
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-dashboard")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--websocket-port", type=int, default=8766)
    arguments = parser.parse_args(argv)
    run_directory = arguments.run_dir or _latest_run(DEFAULT_RUN_ROOT)
    if run_directory is None:
        print("dodge-ng-dashboard: no waypoint DQN run found", file=sys.stderr)
        return 1
    if not run_directory.is_dir():
        print(f"dodge-ng-dashboard: run directory does not exist: {run_directory}")
        return 1
    if not 1 <= arguments.http_port <= 65_535:
        parser.error("--http-port must be between 1 and 65535")
    if not 1 <= arguments.websocket_port <= 65_535:
        parser.error("--websocket-port must be between 1 and 65535")
    try:
        DashboardApplication(run_directory).run(
            arguments.host,
            arguments.http_port,
            arguments.websocket_port,
        )
    except (OSError, RuntimeError) as error:
        print(f"dodge-ng-dashboard: {error}", file=sys.stderr)
        return 1
    return 0


def _http_handler(application: DashboardApplication) -> type[BaseHTTPRequestHandler]:
    page = DASHBOARD_PAGE.read_bytes()

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path in {"/", "/index.html"}:
                self._send(page, "text/html; charset=utf-8")
                return
            if path.startswith("/replay/"):
                replay = application.inspector.resolve_replay(
                    path.removeprefix("/replay/")
                )
                if replay is None:
                    self.send_error(404)
                    return
                content_type = (
                    "application/json"
                    if replay.suffix == ".json"
                    else "application/octet-stream"
                )
                try:
                    self._send(replay.read_bytes(), content_type)
                except OSError:
                    self.send_error(404)
                return
            self.send_error(404)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


def _latest_run(root: Path) -> Path | None:
    candidates = [path for path in root.glob("waypoint-dqn-*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _summary(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    summary = value.get("summary")
    return summary if isinstance(summary, dict) else None


def _integer_seed(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


if __name__ == "__main__":
    raise SystemExit(main())
