#!/usr/bin/env python3
"""Run bounded, local Dodge NG remediation pilots.

The runner deliberately keeps orchestration separate from the learner.  DQN
experiments are child processes with one shared, frozen configuration; fixed
and current-float-TTC controls run in-process on the first training seeds.
The default is a dry plan.  ``--execute`` is required to create artifacts or
start a trainer, and no remote or dashboard operation is performed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_MANIFEST = PROJECT_ROOT / "context" / "kits" / "dodge-ng" / "ng-v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "history" / "dodge" / "ng" / (
    "remediation-cpu-20260909"
)
MAX_PILOT_UPDATES = 1_024
DEFAULT_PILOT_UPDATES = MAX_PILOT_UPDATES
DEFAULT_WARMUP = 128
DEFAULT_CONTROL_SEEDS = 10
DEFAULT_RUN_TIMEOUT = 900.0
DEFAULT_OVERALL_TIMEOUT = 1_800.0

# These are the current hazard run values, with only the bounded pilot budget
# and useful pilot cadence changed.  The runner owns these flags so a baseline
# and candidate cannot accidentally drift in replay, timing, or evaluation
# contracts.  Representation/controller flags remain candidate arguments.
CURRENT_CONFIG: dict[str, object] = {
    "batch_size": 128,
    "replay_capacity": 8_192,
    "learning_rate": 0.00024074661619742944,
    "weight_decay": 0.0001,
    "gamma": 0.99,
    "n_step": 5,
    "train_frequency": 1,
    "target_update_interval": 500,
    "hidden_size": 256,
    "observation_mode": "hazard",
    "grid_spacing": 32,
    "hazard_grid_size": 16,
    "prediction_horizon_frames": 32,
    "spawn_halo_radius": 1,
    "hazard_observation_cadence": "every_step",
    "hold_decisions": 1,
    "steering_tolerance": 2.0,
    "arrival_latching": True,
    "ban_corner_nodes": False,
    "corner_node_penalty": 0.0,
    "step_frames": 3,
    # This is a pilot safety cap, not the 9,000-frame objective.  The final
    # objective must use a cap above 9,000 native frames.
    "max_episode_steps": 1_024,
    "native_lanes": 32,
    "native_execution": "parallel",
    "reset_mode": "native-startup",
    "training_lives": 1,
    "life_loss_penalty": 0.0,
    # Preserve the current run's high-exploration schedule.  At 1,024 updates
    # epsilon remains about .981; the report calls this limitation out.
    "epsilon_decay_steps": 50_000,
    "epsilon_final": 0.05,
    "checkpoint_every": 512,
    "eval_every": 256,
    "device": "cpu",
    "torch_threads": 8,
}

# Extra learner arguments are intentionally open-ended so the learner may add
# a trajectory-input flag without requiring an orchestration-file edit.  The
# flags below are common-run invariants and must not differ by candidate.
PROTECTED_EXTRA_FLAGS = frozenset(
    {
        "--manifest",
        "--run-dir",
        "--resume",
        "--total-steps",
        "--batch-size",
        "--replay-capacity",
        "--learning-rate",
        "--weight-decay",
        "--gamma",
        "--n-step",
        "--warmup-steps",
        "--train-frequency",
        "--target-update-interval",
        "--hidden-size",
        "--grid-spacing",
        "--hold-decisions",
        "--decision-interval",
        "--steering-tolerance",
        "--arrival-latching",
        "--no-arrival-latching",
        "--ban-corner-nodes",
        "--corner-node-penalty",
        "--step-frames",
        "--max-episode-steps",
        "--native-lanes",
        "--native-execution",
        "--reset-mode",
        "--training-lives",
        "--life-loss-penalty",
        "--epsilon-decay-steps",
        "--epsilon-final",
        "--checkpoint-every",
        "--eval-every",
        "--seed",
        "--device",
        "--torch-threads",
        "--skip-holdout",
        "--skip-training-evaluation",
    }
)


def _number(value: object, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _integer(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _summary(value: Mapping[str, object] | None) -> dict[str, object] | None:
    if value is None:
        return None
    nested = value.get("summary")
    return dict(nested) if isinstance(nested, Mapping) else dict(value)


def _summary_row(summary: Mapping[str, object] | None) -> dict[str, object] | None:
    if summary is None:
        return None
    return {
        key: summary.get(key)
        for key in (
            "count",
            "mean_survival_frames",
            "median_survival_frames",
            "p10_survival_frames",
            "worst_survival_frames",
            "best_survival_frames",
            "horizon_completion_fraction",
        )
    }


def _parse_extra_json(raw: str | None, label: str) -> list[str]:
    if raw is None:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} must be a JSON string array: {error}") from error
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a JSON string array")
    return list(value)


def _extra_args(arguments: argparse.Namespace, label: str) -> list[str]:
    repeated = list(getattr(arguments, f"{label}_arg"))
    raw = getattr(arguments, f"{label}_args_json")
    values = repeated + _parse_extra_json(raw, f"--{label}-args-json")
    for token in values:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        if flag in PROTECTED_EXTRA_FLAGS:
            raise ValueError(
                f"{label} extra argument {flag} is runner-owned; "
                "change the shared runner option instead"
            )
    return values


def _as_path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def _command_text(command: Sequence[str]) -> str:
    return shlex.join(str(item) for item in command)


def _common_trainer_args(arguments: argparse.Namespace, run_directory: Path) -> list[str]:
    values: list[str] = [
        "--manifest",
        str(arguments.manifest),
        "--run-dir",
        str(run_directory),
        "--total-steps",
        str(arguments.total_steps),
        "--batch-size",
        str(arguments.batch_size),
        "--replay-capacity",
        str(arguments.replay_capacity),
        "--learning-rate",
        str(arguments.learning_rate),
        "--weight-decay",
        str(arguments.weight_decay),
        "--gamma",
        str(arguments.gamma),
        "--n-step",
        str(arguments.n_step),
        "--warmup-steps",
        str(arguments.warmup_steps),
        "--train-frequency",
        str(arguments.train_frequency),
        "--target-update-interval",
        str(arguments.target_update_interval),
        "--hidden-size",
        str(arguments.hidden_size),
        "--observation-mode",
        str(arguments.observation_mode),
        "--grid-spacing",
        str(arguments.grid_spacing),
        "--hazard-grid-size",
        str(arguments.hazard_grid_size),
        "--prediction-horizon-frames",
        str(arguments.prediction_horizon_frames),
        "--spawn-halo-radius",
        str(arguments.spawn_halo_radius),
        "--hazard-observation-cadence",
        str(arguments.hazard_observation_cadence),
        "--hold-decisions",
        str(arguments.hold_decisions),
        "--steering-tolerance",
        str(arguments.steering_tolerance),
        "--corner-node-penalty",
        str(arguments.corner_node_penalty),
        "--step-frames",
        str(arguments.step_frames),
        "--max-episode-steps",
        str(arguments.max_episode_steps),
        "--native-lanes",
        str(arguments.native_lanes),
        "--native-execution",
        str(arguments.native_execution),
        "--reset-mode",
        str(arguments.reset_mode),
        "--training-lives",
        str(arguments.training_lives),
        "--life-loss-penalty",
        str(arguments.life_loss_penalty),
        "--epsilon-decay-steps",
        str(arguments.epsilon_decay_steps),
        "--epsilon-final",
        str(arguments.epsilon_final),
        "--checkpoint-every",
        str(arguments.checkpoint_every),
        "--eval-every",
        str(arguments.eval_every),
        "--seed",
        str(arguments.seed),
        "--device",
        str(arguments.device),
        "--torch-threads",
        str(arguments.torch_threads),
        "--skip-holdout",
    ]
    if arguments.arrival_latching:
        values.append("--arrival-latching")
    if arguments.ban_corner_nodes:
        values.append("--ban-corner-nodes")
    return values


def _trainer_command(
    arguments: argparse.Namespace,
    label: str,
    run_directory: Path,
    extra: Sequence[str],
) -> list[str]:
    return [
        str(arguments.python),
        "-m",
        arguments.trainer_module,
        *_common_trainer_args(arguments, run_directory),
        *extra,
    ]


def _experiment_plan(
    arguments: argparse.Namespace,
    labels: Sequence[str],
    baseline_extra: Sequence[str],
    candidate_extra: Sequence[str],
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for label in labels:
        if label not in {"baseline", "candidate"}:
            continue
        extra = baseline_extra if label == "baseline" else candidate_extra
        command = _trainer_command(
            arguments,
            label,
            _as_path(arguments.output_dir) / "runs" / label,
            extra,
        )
        plan.append(
            {
                "label": label,
                "configured": bool(label != "candidate" or extra),
                "extra_args": list(extra),
                "command": [str(item) for item in command],
                "command_text": _command_text(command),
                "holdout_flag": "--skip-holdout",
            }
        )
    return plan


def _runtime_environment(output_directory: Path, arguments: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    python_path = str(SRC_ROOT)
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_path = f"{python_path}{os.pathsep}{existing_python_path}"
    environment.update(
        {
            "PYTHONPATH": python_path,
            "DODGE_HEADLESS": "1",
            "OMP_NUM_THREADS": str(arguments.torch_threads),
            "MKL_NUM_THREADS": str(arguments.torch_threads),
            "OPENBLAS_NUM_THREADS": str(arguments.torch_threads),
            "NUMEXPR_NUM_THREADS": str(arguments.torch_threads),
            "RAYON_NUM_THREADS": str(arguments.rayon_threads),
        }
    )
    temp_directory = output_directory / ".launch-tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)
    environment["TMPDIR"] = str(temp_directory)

    # Direct .venv execution already resolves the current native extension on
    # this host.  Add the compiler runtime when available for reproducible
    # launches outside devenv, without replacing the user's library path.
    try:
        native_library = subprocess.run(
            ["gcc", "-print-file-name=libstdc++.so.6"],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        native_library = ""
    if native_library and Path(native_library).is_file():
        library_directory = str(Path(native_library).parent)
        current_library_path = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = (
            f"{library_directory}{os.pathsep}{current_library_path}"
            if current_library_path
            else library_directory
        )
    return environment


def _terminate_process(process: subprocess.Popen[bytes], grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _run_trainer(
    command: Sequence[str],
    output_directory: Path,
    arguments: argparse.Namespace,
    timeout_seconds: float,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    log_path = output_directory / "trainer.log"
    started = time.perf_counter()
    timed_out = False
    return_code: int | None = None
    launch_error: str | None = None
    environment = _runtime_environment(_as_path(arguments.output_dir), arguments)
    with log_path.open("wb") as log:
        try:
            process = subprocess.Popen(
                [str(item) for item in command],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            launch_error = f"{type(error).__name__}: {error}"
        else:
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process(process, min(arguments.stop_grace_seconds, 30.0))
                return_code = process.returncode
    wall_seconds = time.perf_counter() - started
    run_path = output_directory / "run.json"
    run_record: dict[str, object] | None = None
    if run_path.is_file():
        try:
            value = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = None
        if isinstance(value, dict):
            run_record = value
    result: dict[str, object] = {
        "label": output_directory.name,
        "status": (
            "timeout"
            if timed_out
            else "launch_failed"
            if launch_error
            else "completed"
            if return_code == 0
            else "failed"
        ),
        "return_code": return_code,
        "timed_out": timed_out,
        "launch_error": launch_error,
        "wall_seconds": wall_seconds,
        "command": [str(item) for item in command],
        "command_text": _command_text(command),
        "log": str(log_path),
    }
    if run_record is not None:
        result["run"] = run_record
        result["training"] = _summary(
            run_record.get("final_training_evaluation")
            if isinstance(run_record.get("final_training_evaluation"), Mapping)
            else None
        )
        result["inner"] = _summary(
            run_record.get("final_validation")
            if isinstance(run_record.get("final_validation"), Mapping)
            else None
        )
        result["holdout_guard"] = bool(
            run_record.get("holdout_evaluated") is False
            and run_record.get("final_evaluation") is None
        )
        result["learning"] = _learning_metrics(output_directory, run_record)
    else:
        result["holdout_guard"] = False
    return result


def _load_metrics(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _sum_field(rows: Iterable[Mapping[str, object]], field: str) -> float:
    return sum(_number(row.get(field)) for row in rows)


def _action_counts(rows: Sequence[Mapping[str, object]]) -> list[int]:
    counts = [0] * 9
    for row in rows:
        values = row.get("waypoint_action_counts")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for index, value in enumerate(values[:9]):
            counts[index] += _integer(value)
    return counts


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts
        if count > 0
    )


def _learning_metrics(run_directory: Path, run_record: Mapping[str, object]) -> dict[str, object]:
    rows = _load_metrics(run_directory / "metrics.jsonl")
    epsilon = [_number(row.get("epsilon")) for row in rows if "epsilon" in row]
    evaluations = [
        {
            "step": _integer(row.get("step")),
            **_summary_row(row.get("inner_validation"))
            if isinstance(row.get("inner_validation"), Mapping)
            else {},
        }
        for row in rows
        if isinstance(row.get("inner_validation"), Mapping)
    ]
    counts = _action_counts(rows)
    config = run_record.get("config")
    config_map = config if isinstance(config, Mapping) else {}
    total_steps = _integer(run_record.get("updates_completed"))
    lanes = _integer(config_map.get("native_lanes"), 0)
    hold = _integer(config_map.get("hold_decisions"), 0)
    step_frames = _integer(config_map.get("step_frames"), 0)
    native_steps = _integer(run_record.get("native_steps"))
    optimizer_updates = sum(
        1
        for row in rows
        if "loss" in row and math.isfinite(_number(row.get("loss"), math.nan))
    )
    first_inner = evaluations[0] if evaluations else None
    last_inner = evaluations[-1] if evaluations else None
    first_mean = _number(first_inner.get("mean_survival_frames"), math.nan) if first_inner else math.nan
    last_mean = _number(last_inner.get("mean_survival_frames"), math.nan) if last_inner else math.nan
    return {
        "metric_rows": len(rows),
        "evaluation_points": evaluations,
        "first_inner_mean": None if not math.isfinite(first_mean) else first_mean,
        "last_inner_mean": None if not math.isfinite(last_mean) else last_mean,
        "inner_mean_delta": (
            None
            if not math.isfinite(first_mean) or not math.isfinite(last_mean)
            else last_mean - first_mean
        ),
        "action_counts": counts,
        "action_entropy_nats": _entropy(counts),
        "epsilon_first": epsilon[0] if epsilon else None,
        "epsilon_last": epsilon[-1] if epsilon else None,
        "expected_exploration_fraction": (
            statistics.fmean(epsilon) if epsilon else None
        ),
        "actual_random_fraction": None,
        "optimizer_updates": optimizer_updates,
        "configured_updates": total_steps,
        "native_lane_steps": native_steps,
        "actual_native_frames": native_steps * step_frames,
        "nominal_native_frames": total_steps * lanes * hold * step_frames,
        "phase_seconds": {
            "collection": _sum_field(rows, "collection_seconds"),
            "learning": _sum_field(rows, "learning_seconds"),
            "evaluation": _sum_field(rows, "evaluation_seconds"),
            "checkpoint": _sum_field(rows, "checkpoint_seconds"),
        },
        "last_loss": _number(rows[-1].get("loss"), math.nan)
        if rows and "loss" in rows[-1]
        else None,
        "last_q_mean": _number(rows[-1].get("q_mean"), math.nan)
        if rows and "q_mean" in rows[-1]
        else None,
        "last_td_error": _number(rows[-1].get("td_error"), math.nan)
        if rows and "td_error" in rows[-1]
        else None,
    }


def _control_summary(
    seeds: Sequence[int], survival: Sequence[int], terminated: Sequence[bool]
) -> dict[str, object]:
    from dodge.ng.report import summarize_evaluation

    return summarize_evaluation(
        {
            "seeds": [int(seed) for seed in seeds],
            "survival_frames": [int(value) for value in survival],
            "terminated": [bool(value) for value in terminated],
        }
    )


def _selected_rows(result: object, lanes: Sequence[int]) -> tuple[Any, ...]:
    import numpy as np

    lane_ids = np.asarray(getattr(result, "lane_ids"))
    selected: list[int] = []
    for lane in lanes:
        matches = np.flatnonzero(lane_ids == lane)
        if len(matches) != 1:
            raise RuntimeError(f"native reset did not return lane {lane}")
        selected.append(int(matches[0]))
    return tuple(selected)


def _reset_control_lanes(
    environment: Any,
    lanes: Sequence[int],
    grid_size: int,
    *,
    hazard: bool,
) -> tuple[Any, Any | None]:
    import numpy as np

    lane_values = np.asarray(lanes, dtype=np.uint32)
    seed_values = np.zeros(len(lanes), dtype=np.uint32)
    ml_result = environment.reset_ml_lanes_with_centered_startup(
        lane_values,
        seed_values,
        grid_size,
    )
    if not hazard:
        return ml_result, None
    hazard_result = environment.hazard_observations(
        grid_size,
        prediction_horizon_frames=32,
        spawn_halo_radius=1,
    )
    selected = _selected_rows(hazard_result, lanes)
    return hazard_result, selected


def _control_environment(arguments: argparse.Namespace) -> Any:
    from dodge.native.batch import NativeBatchEnvironment

    return NativeBatchEnvironment(
        step_frames=arguments.step_frames,
        execution=arguments.native_execution,
        full_state=False,
        pixels=False,
        board=False,
        ml=True,
        ml_grid_spacing=arguments.grid_spacing,
    )


def _evaluate_fixed_action(
    arguments: argparse.Namespace,
    seeds: Sequence[int],
    action_index: int,
) -> dict[str, object]:
    import numpy as np

    from dodge.dataset import ACTION_CHOICES

    environment = _control_environment(arguments)
    lane_count = len(seeds)
    active = np.ones(lane_count, dtype=bool)
    decision_steps = np.zeros(lane_count, dtype=np.int64)
    survival = np.zeros(lane_count, dtype=np.int64)
    terminated = np.zeros(lane_count, dtype=bool)
    started = time.perf_counter()
    try:
        environment.reset_ml_batch_with_centered_startup(
            np.asarray(seeds, dtype=np.uint32),
            arguments.hazard_grid_size,
        )
        while bool(active.any()):
            actions = np.full(lane_count, action_index, dtype=np.uint8)
            actions[~active] = 0
            result = environment.step_ml_batch(actions)
            completed: list[int] = []
            for lane in range(lane_count):
                if not active[lane]:
                    continue
                decision_steps[lane] += 1
                survival[lane] += int(result.rewards[lane])
                native_done = bool(result.done[lane])
                truncated = (
                    not native_done
                    and decision_steps[lane] >= arguments.max_episode_steps
                )
                if native_done or truncated:
                    active[lane] = False
                    terminated[lane] = native_done
                    completed.append(lane)
                    if truncated:
                        survival[lane] = (
                            arguments.max_episode_steps * arguments.step_frames
                        )
            if completed and bool(active.any()):
                _reset_control_lanes(
                    environment,
                    completed,
                    arguments.hazard_grid_size,
                    hazard=False,
                )
    finally:
        environment.close()
    return {
        "kind": "fixed_action",
        "action_index": action_index,
        "action": ACTION_CHOICES[action_index],
        "seeds": [int(seed) for seed in seeds],
        "summary": _control_summary(seeds, survival, terminated),
        "wall_seconds": time.perf_counter() - started,
    }


def _evaluate_float_ttc_planner(
    arguments: argparse.Namespace,
    seeds: Sequence[int],
) -> dict[str, object]:
    import numpy as np

    from dodge.dataset import ACTION_CHOICES
    from dodge.ng.waypoint import WaypointController, WaypointGrid

    grid = WaypointGrid.centered(arguments.hazard_grid_size)
    controller = WaypointController(
        grid,
        tolerance=arguments.steering_tolerance,
        arrival_latching=False,
    )
    environment = _control_environment(arguments)
    lane_count = len(seeds)
    active = np.ones(lane_count, dtype=bool)
    decision_steps = np.zeros(lane_count, dtype=np.int64)
    survival = np.zeros(lane_count, dtype=np.int64)
    terminated = np.zeros(lane_count, dtype=bool)
    waypoint_counts = np.zeros(len(ACTION_CHOICES), dtype=np.int64)
    native_counts = np.zeros(len(ACTION_CHOICES), dtype=np.int64)
    selected_ttc: list[float] = []
    hazard_queries = 0
    finite_ttc_values = 0
    total_ttc_values = 0
    started = time.perf_counter()
    try:
        environment.reset_ml_batch_with_centered_startup(
            np.asarray(seeds, dtype=np.uint32),
            arguments.hazard_grid_size,
        )
        hazard_result = environment.hazard_observations(
            arguments.hazard_grid_size,
            prediction_horizon_frames=arguments.prediction_horizon_frames,
            spawn_halo_radius=arguments.spawn_halo_radius,
        )
        while bool(active.any()):
            hazard_queries += 1
            ttc = np.asarray(hazard_result.ttc_reference, dtype=np.float32).reshape(
                lane_count,
                arguments.hazard_grid_size,
                arguments.hazard_grid_size,
            )
            positions = np.asarray(hazard_result.player_positions, dtype=np.float32)
            finite_ttc_values += int(np.isfinite(ttc).sum())
            total_ttc_values += int(ttc.size)
            actions = np.zeros(lane_count, dtype=np.uint8)
            for lane in np.flatnonzero(active):
                lane_index = int(lane)
                current = grid.nearest_cell(
                    float(positions[lane_index, 0]),
                    float(positions[lane_index, 1]),
                )
                scores: list[float] = []
                targets: list[tuple[int, int]] = []
                for waypoint_action in range(len(ACTION_CHOICES)):
                    target = grid.neighbor_cell(current, waypoint_action)
                    targets.append(target)
                    scores.append(float(ttc[lane_index, target[1], target[0]]))
                waypoint_action = int(np.argmax(np.asarray(scores, dtype=np.float32)))
                target = targets[waypoint_action]
                native_action = controller.native_action_index_for_position(
                    float(positions[lane_index, 0]),
                    float(positions[lane_index, 1]),
                    target,
                )
                actions[lane_index] = native_action
                waypoint_counts[waypoint_action] += 1
                native_counts[native_action] += 1
                selected_ttc.append(scores[waypoint_action])
            result = environment.step_ml_batch(actions)
            completed: list[int] = []
            for lane in range(lane_count):
                if not active[lane]:
                    continue
                decision_steps[lane] += 1
                survival[lane] += int(result.rewards[lane])
                native_done = bool(result.done[lane])
                truncated = (
                    not native_done
                    and decision_steps[lane] >= arguments.max_episode_steps
                )
                if native_done or truncated:
                    active[lane] = False
                    terminated[lane] = native_done
                    completed.append(lane)
                    if truncated:
                        survival[lane] = (
                            arguments.max_episode_steps * arguments.step_frames
                        )
            if not bool(active.any()):
                break
            if completed:
                _reset_control_lanes(
                    environment,
                    completed,
                    arguments.hazard_grid_size,
                    hazard=False,
                )
            hazard_result = environment.hazard_observations(
                arguments.hazard_grid_size,
                prediction_horizon_frames=arguments.prediction_horizon_frames,
                spawn_halo_radius=arguments.spawn_halo_radius,
            )
    finally:
        environment.close()
    return {
        "kind": "current_float_ttc_planner",
        "description": (
            "greedy max frozen-center float32 TTC over the nine neighboring "
            "centered waypoints, then current position steering"
        ),
        "observation_encoding": "ttc_reference_float32",
        "planner_is_oracle": False,
        "seeds": [int(seed) for seed in seeds],
        "summary": _control_summary(seeds, survival, terminated),
        "waypoint_action_counts": waypoint_counts.astype(int).tolist(),
        "native_action_counts": native_counts.astype(int).tolist(),
        "hazard_queries": hazard_queries,
        "finite_ttc_fraction": (
            finite_ttc_values / total_ttc_values if total_ttc_values else 0.0
        ),
        "selected_ttc_mean": (
            statistics.fmean(selected_ttc) if selected_ttc else None
        ),
        "wall_seconds": time.perf_counter() - started,
    }


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _write_json(path: Path, value: object) -> None:
    _write_atomic(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _format_summary(summary: Mapping[str, object] | None) -> str:
    if summary is None:
        return "unavailable"
    return (
        f"mean={_number(summary.get('mean_survival_frames')):.1f}, "
        f"median={_number(summary.get('median_survival_frames')):.1f}, "
        f"p10={_number(summary.get('p10_survival_frames')):.1f}, "
        f"worst={_number(summary.get('worst_survival_frames')):.1f}, "
        f"complete={_number(summary.get('horizon_completion_fraction')):.1%}"
    )


def _learning_evidence(
    experiments: Sequence[Mapping[str, object]],
    controls: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    fixed = [
        _number(_summary(item.get("summary")).get("mean_survival_frames"))
        for item in controls
        if item.get("kind") == "fixed_action" and _summary(item.get("summary"))
    ]
    planner_summary = next(
        (
            _summary(item.get("summary"))
            for item in controls
            if item.get("kind") == "current_float_ttc_planner"
        ),
        None,
    )
    planner_mean = (
        _number(planner_summary.get("mean_survival_frames"))
        if planner_summary
        else None
    )
    results: list[dict[str, object]] = []
    for item in experiments:
        learning = item.get("learning")
        if not isinstance(learning, Mapping):
            continue
        inner_delta = learning.get("inner_mean_delta")
        inner_mean = learning.get("last_inner_mean")
        result = {
            "label": item.get("label"),
            "status": item.get("status"),
            "inner_trend_positive": (
                isinstance(inner_delta, (int, float)) and inner_delta > 0
            ),
            "beats_best_fixed_mean": (
                isinstance(inner_mean, (int, float))
                and bool(fixed)
                and float(inner_mean) > max(fixed)
            ),
            "beats_planner_mean": (
                isinstance(inner_mean, (int, float))
                and planner_mean is not None
                and float(inner_mean) > planner_mean
            ),
            "optimizer_updates": learning.get("optimizer_updates"),
            "action_entropy_nats": learning.get("action_entropy_nats"),
            "expected_exploration_fraction": learning.get(
                "expected_exploration_fraction"
            ),
            "actual_random_fraction": learning.get("actual_random_fraction"),
            "note": (
                "screening only: a 1,024-update run with current epsilon is "
                "not evidence for the 9,000-frame gate"
            ),
        }
        results.append(result)
    return {
        "controls_best_fixed_mean": max(fixed) if fixed else None,
        "planner_mean": planner_mean,
        "runs": results,
        "promotion_gate": (
            "learned final inner/training policy must beat same-seed fixed "
            "controls and the current-float-TTC planner on mean without a "
            "p10 regression, with nondegenerate action use and optimizer updates"
        ),
    }


def _markdown_report(report: Mapping[str, object]) -> str:
    lines = [
        "# Dodge NG remediation CPU pilot",
        "",
        "This is a bounded training-side screening report. Holdout evaluation was "
        "not requested or allowed.",
        "",
        "## Commands",
        "",
        "The following commands are the exact child commands recorded in the JSON "
        "plan. Baseline and candidate use one shared frozen configuration; the "
        "candidate adds only its declared extra arguments.",
        "",
        "```text",
    ]
    for item in report.get("plan", []):
        if isinstance(item, Mapping):
            lines.append(f"{item.get('label')}: {item.get('command_text')}")
    lines.extend(
        [
            "```",
            "",
            "## Experiment results",
            "",
            "| Run | Status | Wall seconds | Inner | Training | Holdout guard |",
            "|---|---|---:|---|---|---|",
        ]
    )
    for item in report.get("experiments", []):
        if not isinstance(item, Mapping):
            continue
        inner = item.get("inner")
        training = item.get("training")
        lines.append(
            f"| {item.get('label')} | {item.get('status')} | "
            f"{_number(item.get('wall_seconds')):.1f} | "
            f"{_format_summary(inner if isinstance(inner, Mapping) else None)} | "
            f"{_format_summary(training if isinstance(training, Mapping) else None)} | "
            f"{item.get('holdout_guard')} |"
        )
    lines.extend(
        [
            "",
            "## Controls",
            "",
            "Controls use the first training seeds, centered startup, the same "
            "three-frame decision cadence and pilot cap, and no holdout seeds.",
            "",
            "| Control | Wall seconds | Summary |",
            "|---|---:|---|",
        ]
    )
    for item in report.get("controls", []):
        if not isinstance(item, Mapping):
            continue
        label = item.get("action", item.get("kind"))
        lines.append(
            f"| {label} | {_number(item.get('wall_seconds')):.1f} | "
            f"{_format_summary(item.get('summary') if isinstance(item.get('summary'), Mapping) else None)} |"
        )
    evidence = report.get("learning_evidence")
    if isinstance(evidence, Mapping):
        lines.extend(
            [
                "",
                "## Learning evidence",
                "",
                str(evidence.get("promotion_gate")),
                "",
            ]
        )
        for item in evidence.get("runs", []):
            if isinstance(item, Mapping):
                lines.append(
                    f"- {item.get('label')}: trend-positive="
                    f"{item.get('inner_trend_positive')}, fixed-control-beat="
                    f"{item.get('beats_best_fixed_mean')}, planner-beat="
                    f"{item.get('beats_planner_mean')}, optimizer-updates="
                    f"{item.get('optimizer_updates')}, entropy="
                    f"{_number(item.get('action_entropy_nats')):.3f}."
                )
    lines.extend(
        [
            "",
            "## Limits and estimates",
            "",
            "- The pilot cap is 3,072 native frames per episode (1,024 decisions "
            "× 3), so it cannot establish the 9,000-frame objective.",
            "- Current epsilon decay is 50,000 updates; epsilon is still about "
            "0.981 at the end of this pilot. Actual random-action fraction is "
            "not emitted by the current learner and is recorded as unavailable.",
            "- The current hazard field is fixed-center float32 TTC and retains its "
            "N² counterfactual workload. A trajectory-input candidate is not "
            "credited until its decision and learning evidence is measured.",
        ]
    )
    estimates = report.get("estimates")
    if isinstance(estimates, Mapping):
        lines.extend(
            [
                "",
                f"Unmeasured estimate basis: {estimates.get('basis')}.",
                f"Per DQN run estimate: {estimates.get('per_dqn_run')}; "
                f"controls: {estimates.get('controls')}; full serial pilot: "
                f"{estimates.get('full_serial_pilot')}.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _estimates(arguments: argparse.Namespace) -> dict[str, object]:
    # Remediation's measured 16,384-update run had 2,598.05 s of summed
    # phases.  Scale conservatively for this 1,024-update screening run and
    # its more frequent eval/checkpoint boundary.  These are planning values;
    # the measured wall field is authoritative after execution.
    phase_seconds_per_update = 2_598.05 / 16_384
    scaled_seconds = phase_seconds_per_update * arguments.total_steps
    return {
        "basis": (
            "REMEDIATION_20260909 phase sum 2,598.05 s / 16,384 updates, "
            "scaled to 1,024 updates with finalization and contention allowance"
        ),
        "per_dqn_run": "3-6 minutes under an uncontended CPU slot",
        "controls": "1-4 minutes; TTC planner is the uncertain component",
        "full_serial_pilot": "8-16 minutes for baseline, candidate, planner, and nine fixed controls",
        "scaled_phase_seconds": scaled_seconds,
        "contention_caveat": "dashboard/benchmark CPU contention can exceed these estimates",
        "pilot_updates": arguments.total_steps,
    }


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if not 1 <= arguments.total_steps <= MAX_PILOT_UPDATES:
        raise ValueError(f"--total-steps must be between 1 and {MAX_PILOT_UPDATES}")
    if arguments.warmup_steps < 1 or arguments.warmup_steps >= arguments.total_steps:
        raise ValueError("--warmup-steps must be positive and below --total-steps")
    if arguments.native_lanes < 1 or arguments.native_lanes > 32:
        raise ValueError("--native-lanes must be between 1 and 32")
    if arguments.control_seed_count < 1 or arguments.control_seed_count > 10:
        raise ValueError("--control-seed-count must be between 1 and 10")
    if arguments.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be positive")
    labels = [value.strip() for value in arguments.experiments.split(",") if value.strip()]
    allowed = {"baseline", "candidate", "planner", "fixed"}
    unknown = set(labels) - allowed
    if unknown:
        raise ValueError(f"unknown experiment labels: {sorted(unknown)}")
    if not labels:
        raise ValueError("--experiments must select at least one experiment")
    if arguments.fixed_actions:
        try:
            action_values = [int(value) for value in arguments.fixed_actions.split(",")]
        except ValueError as error:
            raise ValueError("--fixed-actions must be comma-separated integers") from error
        if any(not 0 <= value < 9 for value in action_values):
            raise ValueError("--fixed-actions action indexes must be in 0..8")
    else:
        raise ValueError("--fixed-actions must not be empty")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dodge-ng-remediation-experiments")
    parser.add_argument("--execute", action="store_true", help="run child trainers and controls")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--trainer-module", default="dodge.ng.dqn")
    parser.add_argument(
        "--experiments",
        default="baseline,candidate,planner,fixed",
        help="comma-separated subset of baseline,candidate,planner,fixed",
    )
    parser.add_argument("--candidate-arg", action="append", default=[])
    parser.add_argument("--baseline-arg", action="append", default=[])
    parser.add_argument("--candidate-args-json")
    parser.add_argument("--baseline-args-json")
    parser.add_argument("--require-candidate", action="store_true")
    parser.add_argument("--total-steps", type=int, default=DEFAULT_PILOT_UPDATES)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--batch-size", type=int, default=int(CURRENT_CONFIG["batch_size"]))
    parser.add_argument("--replay-capacity", type=int, default=int(CURRENT_CONFIG["replay_capacity"]))
    parser.add_argument("--learning-rate", type=float, default=float(CURRENT_CONFIG["learning_rate"]))
    parser.add_argument("--weight-decay", type=float, default=float(CURRENT_CONFIG["weight_decay"]))
    parser.add_argument("--gamma", type=float, default=float(CURRENT_CONFIG["gamma"]))
    parser.add_argument("--n-step", type=int, default=int(CURRENT_CONFIG["n_step"]))
    parser.add_argument("--train-frequency", type=int, default=int(CURRENT_CONFIG["train_frequency"]))
    parser.add_argument("--target-update-interval", type=int, default=int(CURRENT_CONFIG["target_update_interval"]))
    parser.add_argument("--hidden-size", type=int, default=int(CURRENT_CONFIG["hidden_size"]))
    parser.add_argument("--observation-mode", choices=("legacy", "hazard"), default="hazard")
    parser.add_argument("--grid-spacing", type=int, default=int(CURRENT_CONFIG["grid_spacing"]))
    parser.add_argument("--hazard-grid-size", type=int, default=int(CURRENT_CONFIG["hazard_grid_size"]))
    parser.add_argument("--prediction-horizon-frames", type=int, default=int(CURRENT_CONFIG["prediction_horizon_frames"]))
    parser.add_argument("--spawn-halo-radius", type=int, default=int(CURRENT_CONFIG["spawn_halo_radius"]))
    parser.add_argument("--hazard-observation-cadence", choices=("every_step", "decision_boundary"), default="every_step")
    parser.add_argument("--hold-decisions", type=int, default=int(CURRENT_CONFIG["hold_decisions"]))
    parser.add_argument("--steering-tolerance", type=float, default=float(CURRENT_CONFIG["steering_tolerance"]))
    parser.add_argument("--arrival-latching", action="store_true", default=True)
    parser.add_argument("--ban-corner-nodes", action="store_true", default=False)
    parser.add_argument("--corner-node-penalty", type=float, default=float(CURRENT_CONFIG["corner_node_penalty"]))
    parser.add_argument("--step-frames", type=int, default=int(CURRENT_CONFIG["step_frames"]))
    parser.add_argument("--max-episode-steps", type=int, default=int(CURRENT_CONFIG["max_episode_steps"]))
    parser.add_argument("--native-lanes", type=int, default=int(CURRENT_CONFIG["native_lanes"]))
    parser.add_argument("--native-execution", choices=("serial", "parallel"), default="parallel")
    parser.add_argument("--reset-mode", choices=("native-startup", "legacy"), default="native-startup")
    parser.add_argument("--training-lives", type=int, default=int(CURRENT_CONFIG["training_lives"]))
    parser.add_argument("--life-loss-penalty", type=float, default=float(CURRENT_CONFIG["life_loss_penalty"]))
    parser.add_argument("--epsilon-decay-steps", type=int, default=int(CURRENT_CONFIG["epsilon_decay_steps"]))
    parser.add_argument("--epsilon-final", type=float, default=float(CURRENT_CONFIG["epsilon_final"]))
    parser.add_argument("--checkpoint-every", type=int, default=int(CURRENT_CONFIG["checkpoint_every"]))
    parser.add_argument("--eval-every", type=int, default=int(CURRENT_CONFIG["eval_every"]))
    parser.add_argument("--seed", type=int, default=2_026_0903)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=int(CURRENT_CONFIG["torch_threads"]))
    parser.add_argument("--rayon-threads", type=int, default=1)
    parser.add_argument("--control-seed-count", type=int, default=DEFAULT_CONTROL_SEEDS)
    parser.add_argument("--fixed-actions", default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--trainer-timeout-seconds", type=float, default=DEFAULT_RUN_TIMEOUT)
    parser.add_argument("--overall-timeout-seconds", type=float, default=DEFAULT_OVERALL_TIMEOUT)
    parser.add_argument("--stop-grace-seconds", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        _validate_arguments(arguments)
        arguments.manifest = _as_path(arguments.manifest)
        arguments.output_dir = _as_path(arguments.output_dir)
        arguments.python = _as_path(arguments.python)
        baseline_extra = _extra_args(arguments, "baseline")
        candidate_extra = _extra_args(arguments, "candidate")
        from dodge.ng.manifest import load_manifest

        manifest = load_manifest(arguments.manifest)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    labels = [value.strip() for value in arguments.experiments.split(",") if value.strip()]
    plan = _experiment_plan(arguments, labels, baseline_extra, candidate_extra)
    candidate_configured = bool(candidate_extra)
    if arguments.require_candidate and "candidate" in labels and not candidate_configured:
        parser.error("--require-candidate requires at least one candidate extra argument")
    pilot_seeds = tuple(manifest.training_seeds[: arguments.control_seed_count])
    plan_record: dict[str, object] = {
        "schema_version": 1,
        "kind": "dodge_ng_remediation_cpu_pilot",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "manifest": {
            "path": str(arguments.manifest),
            "sha256": manifest.sha256,
            "training_control_seeds": list(pilot_seeds),
            "holdout_used": False,
        },
        "shared_config": {
            key: getattr(arguments, key)
            for key in (
                "total_steps",
                "warmup_steps",
                "batch_size",
                "replay_capacity",
                "learning_rate",
                "weight_decay",
                "gamma",
                "n_step",
                "train_frequency",
                "target_update_interval",
                "hidden_size",
                "observation_mode",
                "hazard_grid_size",
                "prediction_horizon_frames",
                "spawn_halo_radius",
                "hazard_observation_cadence",
                "hold_decisions",
                "step_frames",
                "max_episode_steps",
                "native_lanes",
                "native_execution",
                "reset_mode",
                "epsilon_decay_steps",
                "epsilon_final",
                "checkpoint_every",
                "eval_every",
                "seed",
                "device",
                "torch_threads",
            )
        },
        "controls": {
            "seeds": list(pilot_seeds),
            "fixed_actions": [int(value) for value in arguments.fixed_actions.split(",")],
            "planner": "current float32 frozen-center TTC; no oracle/full-state scorer",
        },
        "plan": plan,
        "execution": {
            "mode": "serial child processes; in-process controls",
            "remote_writes": False,
            "holdout_evaluation": False,
            "candidate_configured": candidate_configured,
        },
        "estimates": _estimates(arguments),
    }
    if not arguments.execute:
        print(json.dumps(plan_record, indent=2, sort_keys=True, default=str))
        return 0

    output_directory = arguments.output_dir
    if output_directory.exists() and any(output_directory.iterdir()):
        parser.error(
            f"output directory is non-empty; choose a new --output-dir: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_json(output_directory / "experiment-plan.json", plan_record)
    experiment_results: list[dict[str, object]] = []
    controls: list[dict[str, object]] = []
    started = time.perf_counter()
    deadline = started + arguments.overall_timeout_seconds
    for item in plan:
        label = str(item["label"])
        if label == "candidate" and not candidate_configured:
            experiment_results.append(
                {
                    **item,
                    "status": "not_configured",
                    "wall_seconds": 0.0,
                    "holdout_guard": False,
                }
            )
            continue
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            experiment_results.append({**item, "status": "overall_timeout"})
            break
        result = _run_trainer(
            [str(value) for value in item["command"]],
            output_directory / "runs" / label,
            arguments,
            min(arguments.trainer_timeout_seconds, remaining),
        )
        experiment_results.append({**item, **result})
        if result.get("status") not in {"completed"}:
            break
    if "planner" in labels and time.perf_counter() < deadline:
        try:
            controls.append(_evaluate_float_ttc_planner(arguments, pilot_seeds))
        except Exception as error:
            controls.append(
                {
                    "kind": "current_float_ttc_planner",
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    if "fixed" in labels and time.perf_counter() < deadline:
        for value in [int(item) for item in arguments.fixed_actions.split(",")]:
            if time.perf_counter() >= deadline:
                break
            try:
                controls.append(_evaluate_fixed_action(arguments, pilot_seeds, value))
            except Exception as error:
                controls.append(
                    {
                        "kind": "fixed_action",
                        "action_index": value,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                break
    plan_record["experiments"] = experiment_results
    plan_record["controls_results"] = controls
    plan_record["elapsed_seconds"] = time.perf_counter() - started
    plan_record["learning_evidence"] = _learning_evidence(experiment_results, controls)
    _write_json(output_directory / "experiment.json", plan_record)
    _write_atomic(output_directory / "EXPERIMENT_REPORT.md", _markdown_report(plan_record))
    print(
        json.dumps(
            {
                "output_dir": str(output_directory),
                "elapsed_seconds": plan_record["elapsed_seconds"],
                "experiment_statuses": [
                    {"label": item.get("label"), "status": item.get("status")}
                    for item in experiment_results
                ],
                "control_count": len(controls),
                "holdout_used": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
