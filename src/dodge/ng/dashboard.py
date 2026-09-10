"""Minimal WebSocket dashboard for danger-map DDQ runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from dodge.ng.telemetry import issue_control

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "history" / "dodge" / "ng"
DASHBOARD_PAGE = Path(__file__).with_name("dashboard.html")
LOCAL_RUN_PREFIX = "run:"
COLAB_JOB_PREFIX = "job:"
CATALOG_CACHE_SECONDS = 1.0
CLIENT_ACTIVE_POLL_SECONDS = 0.5
CLIENT_IDLE_POLL_SECONDS = 2.0
MAX_CACHED_INSPECTORS = 128
METRICS_LIMIT = 500
ACTIVE_STATES = frozenset({"starting", "provisioning", "running", "stopping"})


class RunInspector:
    def __init__(self, run_directory: Path, *, run_id: str | None = None) -> None:
        self.run_directory = Path(run_directory).resolve()
        self.dashboard_directory = self.run_directory / "dashboard"
        self.replay_directory = self.dashboard_directory / "replays"
        self.run_id = run_id
        self.replay_url_prefix = (
            f"/replay/{quote(run_id, safe='')}" if run_id else "/replay"
        )
        self._status_cache_key: tuple[object, ...] | None = None
        self._status_cache: dict[str, object] | None = None
        self._run_metadata_signature: tuple[int, int, int] | None = None
        self._run_metadata_loaded = False
        self._run_metadata_cache: dict[str, object] | None = None
        self._metrics_cache_key: tuple[object, ...] | None = None
        self._metrics_signature: tuple[int, int] | None = None
        self._metrics: list[dict[str, object]] = []
        self._replays_cache_key: tuple[tuple[object, ...], ...] | None = None
        self._replays: list[dict[str, object]] = []
        self._can_record_cache_key: tuple[object, ...] | None = None
        self._can_record_cache = False
        self._inspection_cache: dict[str, object] | None = None
        self._inspection_run: dict[str, object] | None = None
        self._inspection_config: dict[str, object] | None = None
        self._inspection_contract: dict[str, object] | None = None
        self.inspection_revision = 0
        self.status_revision = 0
        self.metrics_revision = 0
        self.replay_revision = 0
        self.snapshot_revision = 0
        self._snapshot_cache_key: tuple[object, ...] | None = None
        self._snapshot_cache: dict[str, object] | None = None

    def status(self) -> dict[str, object]:
        status_path = self.dashboard_directory / "status.json"
        status_signature = _file_signature(status_path)
        if status_signature is None:
            cache_key = (
                "fallback",
                status_signature,
                _file_signature(self.run_directory / "run.json"),
            )
        else:
            cache_key = ("status", status_signature)
        if cache_key == self._status_cache_key and self._status_cache is not None:
            return self._status_cache
        status = _read_json(status_path)
        self._status_cache_key = cache_key
        self._status_cache = status if status is not None else self._fallback_status()
        self.status_revision += 1
        self._snapshot_cache_key = None
        return self._status_cache

    def run_metadata(self) -> dict[str, object] | None:
        path = self.run_directory / "run.json"
        signature = _file_signature(path)
        if self._run_metadata_loaded and signature == self._run_metadata_signature:
            return self._run_metadata_cache
        self._run_metadata_signature = signature
        self._run_metadata_loaded = True
        self._run_metadata_cache = _read_json(path)
        self._inspection_cache = None
        self._inspection_run = None
        self._inspection_config = None
        self._inspection_contract = None
        self._snapshot_cache_key = None
        return self._run_metadata_cache

    def snapshot(self, *, replay_running: bool) -> dict[str, object]:
        status = self.status()
        run = self.run_metadata()
        history = self._read_metrics()
        replays = self._read_replays()
        inspection = self.inspection(status=status, run=run)
        cache_key = (
            self._status_cache_key,
            self._metrics_cache_key,
            self._replays_cache_key,
            self.inspection_revision,
            replay_running,
        )
        if cache_key == self._snapshot_cache_key and self._snapshot_cache is not None:
            return self._snapshot_cache
        self.snapshot_revision += 1
        self._snapshot_cache_key = cache_key
        self._snapshot_cache = {
            "type": "state",
            "status": status,
            "history": history,
            "replays": replays,
            "replay_running": replay_running,
            "inspection": inspection,
            "inspection_revision": self.inspection_revision,
            "status_revision": self.status_revision,
            "metrics_revision": self.metrics_revision,
            "replay_revision": self.replay_revision,
        }
        return self._snapshot_cache

    def default_seed(self) -> int | None:
        status = self.status()
        if isinstance(status, dict):
            seeds = status.get("training_seeds")
            if isinstance(seeds, list) and seeds:
                return _integer_seed(seeds[0])
        run = self.run_metadata()
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

    def can_record(self) -> bool:
        checkpoint = self.run_directory / "checkpoint-best.pt"
        best_signature = _file_signature(checkpoint)
        if not checkpoint.is_file():
            checkpoint = self.run_directory / "checkpoint-latest.pt"
        latest_signature = _file_signature(checkpoint)
        cache_key = (
            best_signature,
            latest_signature,
            self._run_metadata_signature,
        )
        if cache_key == self._can_record_cache_key:
            return self._can_record_cache
        run = self.run_metadata()
        self._can_record_cache_key = cache_key
        self._can_record_cache = bool(
            latest_signature is not None
            and (
                not isinstance(run, dict)
                or run.get("kind") == "dodge_ng_waypoint_dqn_run"
            )
        )
        return self._can_record_cache

    def _fallback_status(self) -> dict[str, object]:
        run = self.run_metadata()
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
        signature = _file_signature(path)
        cache_key = (str(path) if signature is not None else None, signature)
        if cache_key == self._metrics_cache_key:
            return self._metrics
        self._metrics_cache_key = cache_key
        self._metrics_signature = (
            (signature[0], signature[1]) if signature is not None else None
        )
        self._metrics = (
            _read_metrics_tail(path, limit=METRICS_LIMIT)
            if signature is not None
            else []
        )
        self.metrics_revision += 1
        self._snapshot_cache_key = None
        return self._metrics

    def _read_replays(self) -> list[dict[str, object]]:
        signature = self._replay_metadata_signature()
        if signature == self._replays_cache_key:
            return self._replays
        if not signature:
            self._replays_cache_key = signature
            self._replays = []
            self.replay_revision += 1
            self._snapshot_cache_key = None
            return self._replays
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
            playback_start, playback_frame_count = _replay_playback_window(item)
            item["playback_start"] = playback_start
            item["playback_frame_count"] = playback_frame_count
            item["url"] = f"{self.replay_url_prefix}/{quote(frame_file, safe='')}"
            danger_file = item.get("danger_file")
            if isinstance(danger_file, str):
                danger_path = self.resolve_replay(danger_file)
                if danger_path is not None:
                    item["danger_url"] = (
                        f"{self.replay_url_prefix}/{quote(danger_file, safe='')}"
                    )
            result.append(item)
        self._replays_cache_key = signature
        self._replays = result
        self.replay_revision += 1
        self._snapshot_cache_key = None
        return self._replays

    def _replay_metadata_signature(self) -> tuple[tuple[object, ...], ...]:
        if not self.replay_directory.is_dir():
            return ()
        try:
            paths = sorted(self.replay_directory.glob("*.json"))
        except OSError:
            return ()
        signature: list[tuple[object, ...]] = []
        for path in paths:
            file_signature = _file_signature(path)
            if file_signature is not None:
                signature.append((path.name, *file_signature))
        return tuple(signature)

    def inspection(
        self,
        *,
        status: dict[str, object] | None = None,
        run: dict[str, object] | None = None,
    ) -> dict[str, object]:
        status = status if isinstance(status, dict) else self.status()
        run = run if isinstance(run, dict) else self.run_metadata()
        run = run if isinstance(run, dict) else {}
        config = _mapping(run.get("config")) or _mapping(status.get("config")) or {}
        contract = _mapping(run.get("contract")) or {}
        if (
            self._inspection_cache is not None
            and self._inspection_run is run
            and self._inspection_config == config
            and self._inspection_contract == contract
        ):
            return self._inspection_cache

        actions = _string_list(run.get("actions"))
        if not actions:
            actions = _string_list(contract.get("actions"))
        target = _mapping(contract.get("target")) or {}
        model_metadata = _mapping(run.get("model")) or {}
        selected_model = _string(run.get("selected_model"))
        selected_step = _integer(run.get("selected_model_step"))
        selected_payload = (
            _mapping(run.get(selected_model)) if selected_model else None
        )
        if selected_step is None and selected_payload is not None:
            selected_step = _integer(selected_payload.get("step"))
        if selected_step is None:
            best_inner = _mapping(run.get("best_inner")) or {}
            selected_step = _integer(best_inner.get("step"))

        cadence = _mapping(contract.get("cadence")) or {}
        cadence = dict(cadence)
        for key in (
            "checkpoint_every",
            "eval_every",
            "target_update_interval",
            "train_frequency",
            "warmup_steps",
            "total_steps",
            "max_episode_steps",
        ):
            if key not in cadence and key in config:
                cadence[key] = config[key]

        target_model = {
            "name": model_metadata.get("name")
            or model_metadata.get("model")
            or contract.get("model"),
            "architecture": model_metadata.get("architecture")
            or model_metadata.get("type")
            or contract.get("pixel_architecture")
            or target.get("network"),
            "algorithm": target.get("algorithm"),
            "network": target.get("network"),
            "hidden_size": config.get("hidden_size"),
            "action_count": len(actions) if actions else None,
            "actions": actions,
        }
        input_contract = {
            "mode": run.get("observation_mode") or config.get("observation_mode"),
            "size": run.get("observation_size") or contract.get("observation_size"),
            "source": run.get("observation_source")
            or contract.get("observation_source"),
            "contract": run.get("observation_contract")
            or contract.get("observation_contract"),
            "grid_shape": run.get("grid_shape") or contract.get("grid_shape"),
            "point_count": run.get("point_count"),
            "pixel_shape": run.get("pixel_shape") or contract.get("pixel_shape"),
        }
        hazard = _mapping(contract.get("hazard"))
        if hazard is not None:
            input_contract["hazard"] = hazard
        checkpoint: dict[str, object] = {
            "selection": selected_model,
            "step": selected_step,
            "manifest_sha256": run.get("manifest_sha256"),
        }
        for key in ("checkpoint_file", "selected_checkpoint"):
            value = _string(run.get(key))
            if value is not None:
                checkpoint["file"] = value
                break
        self._inspection_run = run
        self._inspection_config = config
        self._inspection_contract = contract
        self.inspection_revision += 1
        self._inspection_cache = {
            "source": "run.json" if run else "status.json",
            "checkpoint": checkpoint,
            "model": target_model,
            "input": input_contract,
            "cadence": cadence,
            "config": config,
        }
        self._snapshot_cache_key = None
        return self._inspection_cache


class RunCatalog:
    """Read-only catalog of local DQN runs and durable Colab job records."""

    def __init__(self, run_root: Path, *, forced_run: Path | None = None) -> None:
        self.run_root = Path(run_root).resolve()
        self.forced_run = Path(forced_run).resolve() if forced_run else None
        self._inspectors: OrderedDict[str, RunInspector] = OrderedDict()
        self._entries_cache: list[dict[str, object]] = []
        self._entries_cache_key: tuple[object, ...] | None = None
        self._entries_checked_at = 0.0
        self._entries_loaded = False
        self.revision = 0

    def entries(self, *, force: bool = False) -> list[dict[str, object]]:
        cache_key = (
            _file_signature(self.run_root),
            _file_signature(self.run_root / "colab-jobs"),
        )
        now = time.monotonic()
        if (
            self._entries_loaded
            and not force
            and cache_key == self._entries_cache_key
            and now - self._entries_checked_at < CATALOG_CACHE_SECONDS
        ):
            return self._entries_cache

        local_paths = self._local_paths()
        job_records = [
            self._job_record(path)
            for path in self._job_paths()
        ]
        job_entries = [record[0] for record in job_records]
        job_states: dict[Path, str] = {}
        for _entry, job, status in job_records:
            run_directory = _string(job.get("run_directory")) or _string(
                job.get("local_run_dir")
            )
            job_state = _string(status.get("state"))
            if run_directory is not None and job_state is not None:
                job_states.setdefault(Path(run_directory).resolve(), job_state)

        entries = [self._local_entry(path, job_states) for path in local_paths]
        entries.extend(job_entries)
        entries = sorted(entries, key=_entry_sort_key)
        if not self._entries_loaded or entries != self._entries_cache:
            self.revision += 1
        self._entries_cache = entries
        self._entries_cache_key = cache_key
        self._entries_checked_at = now
        self._entries_loaded = True
        self._prune_inspectors()
        return self._entries_cache

    def entry(
        self, item_id: str | None, *, force: bool = False
    ) -> dict[str, object] | None:
        if not isinstance(item_id, str):
            return None
        return next(
            (item for item in self.entries(force=force) if item["id"] == item_id),
            None,
        )

    def default_id(self) -> str | None:
        entries = self.entries()
        return str(entries[0]["id"]) if entries else None

    def inspector(self, item_id: str | None) -> RunInspector | None:
        entry = self.entry(item_id)
        if entry is None or entry.get("source") != "local":
            return None
        name = entry.get("name")
        if not isinstance(name, str) or not _safe_relative_path(name):
            return None
        path = self._safe_child(name)
        if path is None or not path.is_dir():
            return None
        item_id = str(entry["id"])
        inspector = self._inspectors.get(item_id)
        if inspector is None or inspector.run_directory != path:
            inspector = RunInspector(path, run_id=item_id)
            self._inspectors[item_id] = inspector
        self._inspectors.move_to_end(item_id)
        return inspector

    def snapshot(
        self,
        item_id: str | None,
        *,
        replay_running: bool,
    ) -> dict[str, object]:
        entry = self.entry(item_id)
        inspector = self.inspector(item_id)
        if entry is not None and inspector is not None:
            snapshot = inspector.snapshot(replay_running=replay_running)
            status = snapshot.get("status")
            state = entry.get("state")
            if (
                isinstance(status, dict)
                and isinstance(state, str)
                and status.get("state") != state
            ):
                snapshot["status"] = {**status, "state": state}
            return snapshot
        if entry is not None:
            status = dict(entry.get("status", {}))
            return {
                "type": "state",
                "status": status,
                "history": [],
                "replays": [],
                "replay_running": False,
                "inspection": None,
                "inspection_revision": 0,
                "status_revision": 0,
                "metrics_revision": 0,
                "replay_revision": 0,
            }
        return {
            "type": "state",
            "status": {"state": "waiting", "step": 0, "total_steps": 0},
            "history": [],
            "replays": [],
            "replay_running": False,
            "inspection": None,
            "inspection_revision": 0,
            "status_revision": 0,
            "metrics_revision": 0,
            "replay_revision": 0,
        }

    def resolve_replay(self, item_id: str, name: str) -> Path | None:
        inspector = self.inspector(item_id)
        return inspector.resolve_replay(name) if inspector is not None else None

    def _local_paths(self) -> list[Path]:
        paths: list[Path] = []
        try:
            candidates = list(self.run_root.iterdir())
        except OSError:
            candidates = []
        if self.forced_run is not None and self.forced_run not in candidates:
            candidates.append(self.forced_run)
        for path in candidates:
            if path.name.startswith(".") or path.name == "colab-jobs":
                continue
            if not path.is_dir() or not self._inside_root(path):
                continue
            if _looks_like_run(path):
                paths.append(path.resolve())
                continue
            if (path / "hpo.json").is_file():
                try:
                    trial_paths = list(path.glob("trial-*"))
                except OSError:
                    trial_paths = []
                paths.extend(
                    trial.resolve()
                    for trial in trial_paths
                    if trial.is_dir()
                    and self._inside_root(trial)
                    and _looks_like_run(trial)
                )
        return sorted(set(paths), key=lambda path: path.name)

    def _job_paths(self) -> list[Path]:
        job_root = self.run_root / "colab-jobs"
        try:
            candidates = list(job_root.iterdir())
        except OSError:
            candidates = []
        return sorted(
            {
                path.resolve()
                for path in candidates
                if path.is_dir()
                and not path.name.startswith(".")
                and self._inside_root(path)
                and _read_json(path / "job.json") is not None
            },
            key=lambda path: path.name,
        )

    def _local_entry(
        self, path: Path, job_states: dict[Path, str] | None = None
    ) -> dict[str, object]:
        name = path.relative_to(self.run_root).as_posix()
        item_id = f"{LOCAL_RUN_PREFIX}{name}"
        inspector = self._inspectors.get(item_id)
        if inspector is None:
            inspector = RunInspector(path, run_id=item_id)
            self._inspectors[item_id] = inspector
        self._inspectors.move_to_end(item_id)
        status = inspector.status()
        run = inspector.run_metadata() or {}
        config = _mapping(status.get("config")) or _mapping(run.get("config")) or {}
        state = _string(status.get("state")) or _run_state(run)
        state = self._reconcile_local_state(path, state, job_states)
        step = _integer(status.get("step"))
        if step is None:
            step = _integer(run.get("updates_completed")) or 0
        total_steps = _integer(status.get("total_steps"))
        if total_steps is None:
            total_steps = _integer(config.get("total_steps")) or 0
        return {
            "id": item_id,
            "name": name,
            "source": "local",
            "state": state,
            "step": step,
            "total_steps": total_steps,
            "native_steps": _integer(status.get("native_steps"))
            or _integer(run.get("native_steps"))
            or 0,
            "updated_at": _updated_at(status, path),
            "mode": _run_mode(config, run),
            "gpu": None,
            "can_record": inspector.can_record(),
            "replay_count": len(inspector._read_replays()),
            "status": status,
        }

    def _reconcile_local_state(
        self,
        path: Path,
        state: str,
        job_states: dict[Path, str] | None = None,
    ) -> str:
        """Reflect a terminal Colab job when its trainer was killed abruptly."""
        if state not in {"starting", "provisioning", "running", "stopping"}:
            return state
        target = path.resolve()
        if job_states is not None:
            job_state = job_states.get(target)
            return (
                job_state
                if job_state in {"succeeded", "completed", "failed", "cancelled", "stopped"}
                else state
            )
        for job_path in self._job_paths():
            job = _read_json(job_path / "job.json") or {}
            run_directory = _string(job.get("run_directory")) or _string(
                job.get("local_run_dir")
            )
            if run_directory is None or Path(run_directory).resolve() != target:
                continue
            job_status = _read_json(job_path / "status.json") or {}
            job_state = _string(job_status.get("state"))
            if job_state in {"succeeded", "completed"}:
                return "completed"
            if job_state in {"failed", "cancelled", "stopped"}:
                return job_state
        return state

    def _job_record(
        self, path: Path
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        job = _read_json(path / "job.json")
        if job is None:
            return {}, {}, {}
        status = _read_json(path / "status.json") or {}
        return self._job_entry(path, job=job, status=status), job, status

    def _job_entry(
        self,
        path: Path,
        *,
        job: dict[str, object] | None = None,
        status: dict[str, object] | None = None,
    ) -> dict[str, object]:
        job = job if isinstance(job, dict) else _read_json(path / "job.json") or {}
        status = (
            status
            if isinstance(status, dict)
            else _read_json(path / "status.json") or {}
        )
        job_id = (
            _string(status.get("job_id")) or _string(job.get("job_id")) or path.name
        )
        remote = _mapping(status.get("remote")) or {}
        step = _integer(status.get("step"))
        if step is None:
            step = _integer(remote.get("step")) or 0
        total_steps = _integer(status.get("total_steps"))
        if total_steps is None:
            trainer = _mapping(job.get("trainer")) or {}
            total_steps = _integer(trainer.get("total_steps")) or 0
        detail = _string(status.get("detail")) or _string(status.get("error"))
        return {
            "id": f"{COLAB_JOB_PREFIX}{path.name}",
            "name": job_id,
            "source": "colab",
            "state": _string(status.get("state")) or "unknown",
            "step": step,
            "total_steps": total_steps,
            "native_steps": _integer(status.get("native_steps"))
            or _integer(remote.get("native_steps"))
            or 0,
            "updated_at": _updated_at(status, path),
            "mode": "Colab job",
            "gpu": _string(status.get("accelerator"))
            or _string(status.get("selected_gpu")),
            "detail": detail,
            "can_record": False,
            "replay_count": 0,
            "status": {
                **status,
                "state": _string(status.get("state")) or "unknown",
                "step": step,
                "total_steps": total_steps,
                "native_steps": _integer(status.get("native_steps"))
                or _integer(remote.get("native_steps"))
                or 0,
            },
        }

    def _prune_inspectors(self) -> None:
        while len(self._inspectors) > MAX_CACHED_INSPECTORS:
            self._inspectors.popitem(last=False)

    def _safe_child(self, name: str) -> Path | None:
        if not _safe_relative_path(name):
            return None
        candidate = (self.run_root / name).resolve()
        return candidate if self._inside_root(candidate) else None

    def _inside_root(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.run_root)
        except ValueError:
            return False
        return True


class DashboardApplication:
    def __init__(
        self,
        run_directory: Path | None = None,
        *,
        run_root: Path | None = None,
        enable_controls: bool = False,
    ) -> None:
        forced_run = Path(run_directory).resolve() if run_directory else None
        if run_root is None:
            run_root = forced_run.parent if forced_run else DEFAULT_RUN_ROOT
        else:
            run_root = Path(run_root).resolve()
            if forced_run is not None:
                try:
                    forced_run.relative_to(run_root)
                except ValueError:
                    run_root = forced_run.parent
        self.catalog = RunCatalog(run_root, forced_run=forced_run)
        self.selected_run_id = (
            f"{LOCAL_RUN_PREFIX}{forced_run.relative_to(self.catalog.run_root).as_posix()}"
            if forced_run
            else None
        )
        self.enable_controls = enable_controls
        self._replay_process: subprocess.Popen[bytes] | None = None
        self._replay_run_id: str | None = None
        self._replay_lock = threading.Lock()
        self._snapshot_cache_key: tuple[object, ...] | None = None
        self._snapshot_cache: dict[str, object] | None = None
        self._encoded_snapshot_cache: str | None = None

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

        last_payload: str | None = None
        while True:
            try:
                snapshot = self._snapshot()
                encoded = self._encoded_snapshot(snapshot)
                if encoded != last_payload:
                    await connection.send(encoded)  # type: ignore[attr-defined]
                    last_payload = encoded
                message = await asyncio.wait_for(  # type: ignore[attr-defined]
                    connection.recv(),  # type: ignore[attr-defined]
                    timeout=self._client_poll_seconds(snapshot),
                )
            except TimeoutError:
                continue
            except ConnectionClosed:
                return
            except (ConnectionError, OSError):
                return
            if isinstance(message, str):
                try:
                    parsed = json.loads(message)
                except json.JSONDecodeError:
                    reply = {"type": "error", "message": "invalid JSON"}
                else:
                    reply = self._message(parsed)
                try:
                    await connection.send(  # type: ignore[attr-defined]
                        json.dumps(reply, separators=(",", ":"))
                    )
                except (ConnectionClosed, ConnectionError, OSError):
                    return

    def _message(self, message: object) -> dict[str, object]:
        if not isinstance(message, dict):
            return {"type": "error", "message": "invalid message"}
        message_type = message.get("type")
        if message_type == "select":
            run_id = message.get("run_id")
            if (
                not isinstance(run_id, str)
                or self.catalog.entry(run_id, force=True) is None
            ):
                return {"type": "error", "message": "unknown run"}
            self.selected_run_id = run_id
            self._invalidate_snapshot_cache()
            return {"type": "ack", "run_id": run_id}
        if message_type == "control":
            if not self.enable_controls:
                return {"type": "error", "message": "dashboard is read-only"}
            command = message.get("command")
            if not isinstance(command, str):
                return {"type": "error", "message": "invalid control"}
            inspector = self.catalog.inspector(self.selected_run_id)
            if inspector is None:
                return {"type": "error", "message": "selected item is not a local run"}
            try:
                issue_control(inspector.run_directory, command)
            except (OSError, ValueError) as error:
                return {"type": "error", "message": str(error)}
            return {"type": "ack", "command": command}
        if message_type == "replay":
            inspector = self.catalog.inspector(self.selected_run_id)
            if inspector is None:
                return {
                    "type": "error",
                    "message": "selected item has no local checkpoint",
                }
            if not inspector.can_record():
                return {
                    "type": "error",
                    "message": (
                        "selected local run has no compatible waypoint checkpoint"
                    ),
                }
            seed = message.get("seed", inspector.default_seed())
            if not isinstance(seed, int) or isinstance(seed, bool):
                return {"type": "error", "message": "invalid seed"}
            try:
                started = self._start_replay(inspector, seed)
            except (OSError, ValueError, RuntimeError) as error:
                return {"type": "error", "message": str(error)}
            return {"type": "replay", "started": started, "seed": seed}
        return {"type": "error", "message": "invalid message type"}

    def _snapshot(self) -> dict[str, object]:
        entries = self.catalog.entries()
        valid_ids = {str(item["id"]) for item in entries}
        if self.selected_run_id not in valid_ids:
            self.selected_run_id = self.catalog.default_id()
        selected = self.catalog.entry(self.selected_run_id)
        selected_inspector = self.catalog.inspector(self.selected_run_id)
        replay_running = self._replay_running(self.selected_run_id)
        snapshot = self.catalog.snapshot(
            self.selected_run_id,
            replay_running=replay_running,
        )
        snapshot_key = (
            self.catalog.revision,
            self.selected_run_id,
            selected_inspector.snapshot_revision if selected_inspector else 0,
            replay_running,
        )
        if snapshot_key == self._snapshot_cache_key and self._snapshot_cache is not None:
            return self._snapshot_cache
        self._snapshot_cache_key = snapshot_key
        self._encoded_snapshot_cache = None
        self._snapshot_cache = {
            **snapshot,
            "runs": entries,
            "selected_run_id": self.selected_run_id,
            "selected_run": selected,
            "controls_enabled": self.enable_controls,
            "catalog_revision": self.catalog.revision,
        }
        return self._snapshot_cache

    def _encoded_snapshot(self, snapshot: dict[str, object] | None = None) -> str:
        if snapshot is None:
            snapshot = self._snapshot()
        if self._encoded_snapshot_cache is None:
            self._encoded_snapshot_cache = json.dumps(
                snapshot, separators=(",", ":")
            )
        return self._encoded_snapshot_cache

    def _client_poll_seconds(self, snapshot: dict[str, object]) -> float:
        status = snapshot.get("status")
        state = status.get("state") if isinstance(status, dict) else None
        return (
            CLIENT_ACTIVE_POLL_SECONDS
            if state in ACTIVE_STATES
            else CLIENT_IDLE_POLL_SECONDS
        )

    def _invalidate_snapshot_cache(self) -> None:
        self._snapshot_cache_key = None
        self._snapshot_cache = None
        self._encoded_snapshot_cache = None

    def _start_replay(self, inspector: RunInspector, seed: int) -> bool:
        with self._replay_lock:
            if self._replay_process is not None:
                if self._replay_process.poll() is None:
                    return False
                self._replay_process = None
            log_path = inspector.run_directory / "dashboard" / "replay.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab") as log:
                self._replay_process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "dodge.ng.replay",
                        "--run-dir",
                        str(inspector.run_directory),
                        "--seed",
                        str(seed),
                    ],
                    cwd=PROJECT_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self._replay_run_id = inspector.run_id
        return True

    def _replay_running(self, run_id: str | None = None) -> bool:
        with self._replay_lock:
            running = (
                self._replay_process is not None and self._replay_process.poll() is None
            )
            return running and (run_id is None or run_id == self._replay_run_id)

    def resolve_replay(self, encoded_path: str) -> Path | None:
        run_part, separator, replay_part = encoded_path.partition("/")
        if not separator:
            return None
        return self.catalog.resolve_replay(
            unquote(run_part),
            unquote(replay_part),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-dashboard")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--websocket-port", type=int, default=8766)
    parser.add_argument(
        "--enable-controls",
        action="store_true",
        help="allow pause/resume/stop for explicitly selected local runs",
    )
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.http_port <= 65_535:
        parser.error("--http-port must be between 1 and 65535")
    if not 1 <= arguments.websocket_port <= 65_535:
        parser.error("--websocket-port must be between 1 and 65535")
    run_directory = arguments.run_dir.resolve() if arguments.run_dir else None
    if run_directory is not None and not run_directory.is_dir():
        print(f"dodge-ng-dashboard: run directory does not exist: {run_directory}")
        return 1
    application = DashboardApplication(
        run_directory,
        run_root=arguments.run_root,
        enable_controls=arguments.enable_controls,
    )
    if not application.catalog.entries():
        print("dodge-ng-dashboard: no NG runs or Colab jobs found", file=sys.stderr)
        return 1
    try:
        application.run(
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
                replay = application.resolve_replay(path.removeprefix("/replay/"))
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
    try:
        candidates = [path for path in root.iterdir() if _looks_like_run(path)]
    except OSError:
        candidates = []
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _looks_like_run(path: Path) -> bool:
    return path.is_dir() and (
        (path / "run.json").is_file() or (path / "dashboard" / "status.json").is_file()
    )


def _safe_relative_path(value: str) -> bool:
    path = Path(value)
    return (
        bool(value)
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _entry_sort_key(entry: dict[str, object]) -> tuple[int, float, str]:
    state = _string(entry.get("state")) or "unknown"
    state_priority = {
        "running": 0,
        "starting": 1,
        "provisioning": 2,
        "stopping": 3,
        "waiting": 4,
        "completed": 5,
        "stopped": 6,
        "cancelled": 7,
        "failed": 8,
        "unknown": 9,
    }.get(state, 9)
    updated_at = entry.get("updated_at")
    updated = float(updated_at) if isinstance(updated_at, (int, float)) else 0.0
    return state_priority, -updated, str(entry.get("name", ""))


def _mapping(value: object) -> dict[str, object] | None:
    return dict(value) if isinstance(value, dict) else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _file_signature(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size, stat.st_ino


def _updated_at(status: dict[str, object], path: Path) -> float:
    value = status.get("updated_at")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return path.stat().st_mtime_ns / 1_000_000_000
    except OSError:
        return 0.0


def _run_state(run: dict[str, object]) -> str:
    if run.get("stopped_early"):
        return "stopped"
    return "completed"


def _run_mode(config: dict[str, object], run: dict[str, object]) -> str:
    if config.get("observation_mode") == "hazard":
        grid_size = _integer(config.get("hazard_grid_size"))
        return (
            f"Danger-map DDQ · {grid_size}×{grid_size}"
            if grid_size
            else "Danger-map DDQ"
        )
    if run.get("kind") == "dodge_ng_pixel_dqn_run":
        return "Pixel DQN"
    return "Waypoint DQN"


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_metrics_tail(path: Path, *, limit: int) -> list[dict[str, object]]:
    if limit < 1:
        return []
    chunks: list[bytes] = []
    newline_count = 0
    try:
        with path.open("rb") as stream:
            position = stream.seek(0, 2)
            while position > 0 and newline_count <= limit:
                read_size = min(64 * 1024, position)
                position -= read_size
                stream.seek(position)
                chunk = stream.read(read_size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
    except OSError:
        return []
    lines = b"".join(reversed(chunks)).splitlines()
    metrics: list[dict[str, object]] = []
    for line in lines[-limit:]:
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            metrics.append(value)
    return metrics


def _summary(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    summary = value.get("summary")
    return summary if isinstance(summary, dict) else None


def _integer_seed(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _replay_playback_window(metadata: dict[str, object]) -> tuple[int, int]:
    """Return the playable frame window, excluding legacy boundary frames."""
    frame_count = _nonnegative_integer(metadata.get("frame_count"))
    if frame_count is None:
        return 0, 0
    explicit_start = _nonnegative_integer(metadata.get("playback_start"))
    explicit_count = _nonnegative_integer(metadata.get("playback_frame_count"))
    if (
        explicit_start is not None
        and explicit_count is not None
        and explicit_start + explicit_count <= frame_count
    ):
        return explicit_start, explicit_count

    if frame_count == 0:
        return 0, 0
    start = 1
    terminal = 1 if metadata.get("done") is True else 0
    return start, max(0, frame_count - start - terminal)


def _nonnegative_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
