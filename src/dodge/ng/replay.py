"""Record deterministic pixel replays from waypoint DQN checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import BinaryIO, Literal

import numpy as np
import torch

from dodge.control import ControlRuntimeError
from dodge.native.batch import NativeBatchEnvironment, NativeBatchResult
from dodge.native.differential import FRAME_HEIGHT, FRAME_SIZE, FRAME_WIDTH
from dodge.ng.dqn import (
    WAYPOINT_DQN_VERSION,
    WAYPOINT_OBSERVATION_SIZE,
    DQNConfig,
    DuelingWaypointDQN,
    config_from_json,
    evaluate_waypoint_dqn,
    waypoint_controller_for_config,
)
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from dodge.ng.pixel_regression import (
    compare_saved_replay,
    unavailable_pixel_regression,
)

REPLAY_VERSION = 3
REPRESENTATIVE_REPLAY_SET_VERSION = 1
REPRESENTATIVE_ROLES: tuple[str, ...] = ("best", "mean", "bad")
ResetMode = Literal["native-startup", "legacy"]
RESET_MODES: tuple[ResetMode, ...] = ("native-startup", "legacy")


def record_replay(
    run_directory: Path,
    seed: int,
    *,
    checkpoint: Path | None = None,
    max_steps: int | None = None,
    reset_mode: ResetMode | None = None,
) -> dict[str, object]:
    run_directory = Path(run_directory).resolve()
    if not 0 <= seed <= 32_767:
        raise ValueError("replay seed must be between 0 and 32767")
    checkpoint_path = checkpoint or _checkpoint_path(run_directory)
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise ControlRuntimeError(f"DQN checkpoint does not exist: {checkpoint_path}")
    payload = _load_checkpoint_payload(checkpoint_path)
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("DQN checkpoint configuration is invalid")
    config = config_from_json(config_payload)
    config.validate()
    if reset_mode is None:
        configured_mode = config_payload.get("reset_mode")
        reset_mode = configured_mode if configured_mode in RESET_MODES else "legacy"
    if reset_mode not in RESET_MODES:
        raise ValueError(f"replay reset mode must be one of {RESET_MODES}")
    model = DuelingWaypointDQN(hidden_size=config.hidden_size)
    model_state = payload.get("best_model_state") or payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("DQN checkpoint has no model state")
    try:
        model.load_state_dict(model_state)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"DQN checkpoint model state is invalid: {error}") from error
    model.eval()

    best_inner = payload.get("best_inner")
    checkpoint_step = (
        int(best_inner["step"])
        if isinstance(best_inner, dict)
        else int(payload.get("step", 0))
    )
    if max_steps is None:
        max_steps = config.max_episode_steps
    if max_steps < 1:
        raise ValueError("replay maximum steps must be positive")

    replay_directory = run_directory / "dashboard" / "replays"
    replay_directory.mkdir(parents=True, exist_ok=True)
    stem = f"seed-{seed}-checkpoint-{checkpoint_step:06d}-{reset_mode}"
    frame_path = replay_directory / f"{stem}.bin"
    metadata_path = replay_directory / f"{stem}.json"
    temporary_frame_path = frame_path.with_name(f".{frame_path.name}.tmp")
    controller = waypoint_controller_for_config(config)
    native_steps = 0
    frame_count = 0
    done = False
    last_frame = 0
    initial_frame = 0
    survival_frames = 0
    native_action_trace: list[int] = []
    saved_frame_numbers: list[int] = []
    temporary_frame_path.unlink(missing_ok=True)
    try:
        with NativeBatchEnvironment(
            step_frames=config.step_frames,
            execution="serial",
            full_state=False,
            pixels=True,
            board=False,
            ml=True,
            ml_grid_spacing=config.grid_spacing,
        ) as environment:
            if reset_mode == "legacy":
                result = environment.reset_batch([seed])
            else:
                result = environment.reset_batch_with_startup([seed])
            observations, positions = _replay_state(result)
            initial_frame = int(result.frames[0])
            with temporary_frame_path.open("wb") as stream:
                for _ in range(max_steps):
                    with torch.inference_mode():
                        action = int(
                            model(torch.from_numpy(observations)).argmax(dim=1)[0]
                        )
                    target_cell = controller.grid.target_cell_for_action(
                        float(positions[0, 0]),
                        float(positions[0, 1]),
                        action,
                    )
                    arrived = controller.arrival_latching and controller.target_reached(
                        float(positions[0, 0]),
                        float(positions[0, 1]),
                        target_cell,
                    )
                    for _ in range(config.hold_decisions):
                        native_action = controller.native_action_index_for_position(
                            float(positions[0, 0]),
                            float(positions[0, 1]),
                            target_cell,
                            arrived=arrived,
                        )
                        native_action_trace.append(native_action)
                        result = environment.step_batch([native_action])
                        observations, positions = _replay_state(result)
                        native_steps += 1
                        survival_frames += int(result.rewards[0])
                        if bool(result.done[0]):
                            done = True
                            break
                        if controller.arrival_latching and not arrived:
                            arrived = controller.target_reached(
                                float(positions[0, 0]),
                                float(positions[0, 1]),
                                target_cell,
                            )
                        frame_count += _write_frame(stream, result)
                        last_frame = int(result.frames[0])
                        saved_frame_numbers.append(last_frame)
                    if done:
                        break
        metadata = {
            "version": REPLAY_VERSION,
            "kind": "dodge_ng_waypoint_dqn_replay",
            "seed": seed,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_step": checkpoint_step,
            "reset_mode": reset_mode,
            "manifest_sha256": payload.get("manifest_sha256"),
            "config": config.to_json(),
            "frame_file": frame_path.name,
            "frame_count": frame_count,
            "playback_start": 0,
            "playback_frame_count": frame_count,
            "playback_start_frame": initial_frame,
            "survival_frames": survival_frames,
            "frame_width": FRAME_WIDTH,
            "frame_height": FRAME_HEIGHT,
            "frame_bytes": FRAME_SIZE,
            "step_frames": config.step_frames,
            "native_steps": native_steps,
            "last_frame": last_frame,
            "done": done,
            "action_trace": {
                "encoding": "native_action_index_u8",
                "actions": native_action_trace,
                "saved_frame_numbers": saved_frame_numbers,
                "initial_frame": initial_frame,
                "step_frames": config.step_frames,
            },
            "created_at": time.time(),
        }
        temporary_frame_path.replace(frame_path)
        try:
            pixel_regression = compare_saved_replay(run_directory, metadata)
        except Exception as error:
            pixel_regression = unavailable_pixel_regression(metadata, error)
        metadata["pixel_regression"] = pixel_regression
        _atomic_write_json(metadata_path, metadata)
        if pixel_regression.get("status") != "passed":
            raise ControlRuntimeError(
                "saved replay failed original-cartridge pixel regression: "
                f"{pixel_regression.get('status', 'unknown')}"
            )
        return metadata
    except Exception:
        temporary_frame_path.unlink(missing_ok=True)
        raise


def select_representative_replays(
    evaluation: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Select deterministic best, mean, and bad runs from one evaluation."""
    raw_seeds = evaluation.get("seeds")
    raw_survival = evaluation.get("survival_frames")
    summary = evaluation.get("summary")
    if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes)):
        raise ValueError("representative evaluation seeds are invalid")
    if not isinstance(raw_survival, Sequence) or isinstance(raw_survival, (str, bytes)):
        raise ValueError("representative evaluation survival values are invalid")
    if len(raw_seeds) == 0 or len(raw_seeds) != len(raw_survival):
        raise ValueError("representative evaluation seed/value lengths differ")
    records: list[tuple[int, int]] = []
    for raw_seed, raw_frames in zip(raw_seeds, raw_survival, strict=True):
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise ValueError("representative evaluation seed is invalid")
        if isinstance(raw_frames, bool) or not isinstance(raw_frames, int):
            raise ValueError("representative evaluation survival value is invalid")
        if raw_frames < 0:
            raise ValueError("representative evaluation survival value is negative")
        records.append((raw_seed, raw_frames))

    if isinstance(summary, Mapping):
        raw_mean = summary.get("mean_survival_frames")
    else:
        raw_mean = None
    if isinstance(raw_mean, bool) or not isinstance(raw_mean, (int, float)):
        raw_mean = sum(frames for _, frames in records) / len(records)
    mean = float(raw_mean)
    if not math.isfinite(mean):
        raise ValueError("representative evaluation mean is not finite")

    best_seed, best_frames = min(records, key=lambda item: (-item[1], item[0]))
    bad_seed, bad_frames = min(records, key=lambda item: (item[1], item[0]))
    mean_seed, mean_frames = min(
        records,
        key=lambda item: (abs(item[1] - mean), item[0]),
    )
    selected = {
        "best": (best_seed, best_frames),
        "mean": (mean_seed, mean_frames),
        "bad": (bad_seed, bad_frames),
    }
    return {
        role: {
            "replay_role": role,
            "selection_split": "training",
            "seed": seed,
            "survival_frames": frames,
            "selection_mean_survival_frames": mean,
            "selection_distance_from_mean": abs(frames - mean),
        }
        for role, (seed, frames) in selected.items()
    }


def record_representative_replays(
    run_directory: Path,
    *,
    checkpoint: Path | None = None,
    manifest_path: Path | None = None,
    max_steps: int | None = None,
    reset_mode: ResetMode | None = None,
) -> dict[str, object]:
    """Record one training-split replay for each comparison role."""
    run_directory = Path(run_directory).resolve()
    checkpoint_path = (checkpoint or _checkpoint_path(run_directory)).resolve()
    if not checkpoint_path.is_file():
        raise ControlRuntimeError(f"DQN checkpoint does not exist: {checkpoint_path}")
    payload = _load_checkpoint_payload(checkpoint_path)
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("DQN checkpoint configuration is invalid")
    config = config_from_json(config_payload)
    config.validate()
    manifest = load_manifest(manifest_path or DEFAULT_MANIFEST_PATH)
    if payload.get("manifest_sha256") != manifest.sha256:
        raise ValueError("DQN checkpoint manifest does not match representative set")

    model = _load_model(payload, config)
    evaluation = evaluate_waypoint_dqn(model, manifest.training_seeds, config)
    selected = select_representative_replays(evaluation)
    replay_directory = run_directory / "dashboard" / "replays"
    replay_directory.mkdir(parents=True, exist_ok=True)
    roles: dict[str, dict[str, object]] = {}
    for role in REPRESENTATIVE_ROLES:
        selection = selected[role]
        metadata = record_replay(
            run_directory,
            int(selection["seed"]),
            checkpoint=checkpoint_path,
            max_steps=max_steps,
            reset_mode=reset_mode,
        )
        metadata.update(selection)
        metadata_path = replay_directory / (
            f"{Path(str(metadata['frame_file'])).stem}.json"
        )
        _atomic_write_json(metadata_path, metadata)
        roles[role] = {
            "seed": selection["seed"],
            "survival_frames": selection["survival_frames"],
            "frame_file": metadata["frame_file"],
            "metadata_file": metadata_path.name,
        }

    result = {
        "version": REPRESENTATIVE_REPLAY_SET_VERSION,
        "kind": "dodge_ng_representative_replay_set",
        "manifest_sha256": manifest.sha256,
        "selection_split": "training",
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_step": _checkpoint_step(payload),
        "training_evaluation": evaluation,
        "roles": roles,
        "created_at": time.time(),
    }
    _atomic_write_json(
        run_directory / "dashboard" / "representative-replays.json",
        result,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-replay")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--reset-mode", choices=RESET_MODES)
    parser.add_argument(
        "--representative",
        action="store_true",
        help="record deterministic best/mean/bad training-split replays",
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.representative:
            metadata = record_representative_replays(
                arguments.run_dir,
                checkpoint=arguments.checkpoint,
                manifest_path=arguments.manifest,
                max_steps=arguments.max_steps,
                reset_mode=arguments.reset_mode,
            )
        else:
            if arguments.seed is None:
                parser.error("--seed is required unless --representative is used")
            metadata = record_replay(
                arguments.run_dir,
                arguments.seed,
                checkpoint=arguments.checkpoint,
                max_steps=arguments.max_steps,
                reset_mode=arguments.reset_mode,
            )
    except (ControlRuntimeError, OSError, RuntimeError, ValueError) as error:
        print(f"dodge-ng-replay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, sort_keys=True))
    return 0


def _checkpoint_path(run_directory: Path) -> Path:
    best = run_directory / "checkpoint-best.pt"
    return best if best.is_file() else run_directory / "checkpoint-latest.pt"


def _checkpoint_step(payload: Mapping[str, object]) -> int:
    best_inner = payload.get("best_inner")
    if isinstance(best_inner, Mapping):
        step = best_inner.get("step")
        if isinstance(step, int) and not isinstance(step, bool):
            return step
    step = payload.get("step", 0)
    return step if isinstance(step, int) and not isinstance(step, bool) else 0


def _load_checkpoint_payload(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ControlRuntimeError(f"could not load DQN checkpoint: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("DQN checkpoint must contain an object")
    if payload.get("kind") != "dodge_ng_waypoint_dqn_checkpoint":
        raise ValueError("DQN checkpoint kind is invalid")
    if payload.get("version") != WAYPOINT_DQN_VERSION:
        raise ValueError("DQN checkpoint version is invalid")
    return payload


def _load_model(payload: Mapping[str, object], config: DQNConfig) -> DuelingWaypointDQN:
    model = DuelingWaypointDQN(hidden_size=config.hidden_size)
    model_state = payload.get("best_model_state") or payload.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError("DQN checkpoint has no model state")
    try:
        model.load_state_dict(model_state)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"DQN checkpoint model state is invalid: {error}") from error
    model.eval()
    return model


def _write_frame(stream: BinaryIO, result: NativeBatchResult) -> int:
    pixels = result.pixels
    if pixels is None or pixels.ndim < 2 or pixels.shape[0] != 1:
        raise ControlRuntimeError("native replay result has no one-lane pixels")
    frame = np.asarray(pixels[0], dtype=np.uint8)
    if frame.size != FRAME_SIZE:
        raise ControlRuntimeError(
            "native replay frame has unexpected size: "
            f"expected {FRAME_SIZE}, got {frame.size}"
        )
    if np.any(frame > 15):
        raise ControlRuntimeError(
            "native replay frame contains invalid palette indexes"
        )
    stream.write(frame.reshape(FRAME_HEIGHT, FRAME_WIDTH).tobytes())
    return 1


def _replay_state(result: NativeBatchResult) -> tuple[np.ndarray, np.ndarray]:
    observations = result.ml_observation
    positions = result.player_positions
    if observations is None or positions is None:
        raise ControlRuntimeError("native replay result has no ML state")
    if observations.shape != (1, WAYPOINT_OBSERVATION_SIZE) or positions.shape != (
        1,
        2,
    ):
        raise ControlRuntimeError("native replay result has invalid ML state shape")
    return observations, positions


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
