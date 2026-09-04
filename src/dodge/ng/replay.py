"""Record deterministic pixel replays from waypoint DQN checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import BinaryIO

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
)
from dodge.ng.waypoint import WaypointController, WaypointGrid

REPLAY_VERSION = 1


def record_replay(
    run_directory: Path,
    seed: int,
    *,
    checkpoint: Path | None = None,
    max_steps: int | None = None,
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
    config = DQNConfig(**config_payload)
    config.validate()
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
    stem = f"seed-{seed}-checkpoint-{checkpoint_step:06d}"
    frame_path = replay_directory / f"{stem}.bin"
    metadata_path = replay_directory / f"{stem}.json"
    temporary_frame_path = frame_path.with_name(f".{frame_path.name}.tmp")
    grid = WaypointGrid(config.grid_spacing)
    controller = WaypointController(grid)
    native_steps = 0
    frame_count = 0
    done = False
    last_frame = 0
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
            result = environment.reset_batch([seed])
            observations, positions = _replay_state(result)
            with temporary_frame_path.open("wb") as stream:
                frame_count += _write_frame(stream, result)
                last_frame = int(result.frames[0])
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
                    for _ in range(config.hold_decisions):
                        native_action = controller.native_action_index_for_position(
                            float(positions[0, 0]),
                            float(positions[0, 1]),
                            target_cell,
                        )
                        result = environment.step_batch([native_action])
                        observations, positions = _replay_state(result)
                        frame_count += _write_frame(stream, result)
                        native_steps += 1
                        last_frame = int(result.frames[0])
                        if bool(result.done[0]):
                            done = True
                            break
                    if done:
                        break
        temporary_frame_path.replace(frame_path)
        metadata = {
            "version": REPLAY_VERSION,
            "kind": "dodge_ng_waypoint_dqn_replay",
            "seed": seed,
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_step": checkpoint_step,
            "manifest_sha256": payload.get("manifest_sha256"),
            "config": config.to_json(),
            "frame_file": frame_path.name,
            "frame_count": frame_count,
            "frame_width": FRAME_WIDTH,
            "frame_height": FRAME_HEIGHT,
            "frame_bytes": FRAME_SIZE,
            "step_frames": config.step_frames,
            "native_steps": native_steps,
            "last_frame": last_frame,
            "done": done,
            "created_at": time.time(),
        }
        _atomic_write_json(metadata_path, metadata)
        return metadata
    except Exception:
        temporary_frame_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-replay")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int)
    arguments = parser.parse_args(argv)
    try:
        metadata = record_replay(
            arguments.run_dir,
            arguments.seed,
            checkpoint=arguments.checkpoint,
            max_steps=arguments.max_steps,
        )
    except (ControlRuntimeError, OSError, RuntimeError, ValueError) as error:
        print(f"dodge-ng-replay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, sort_keys=True))
    return 0


def _checkpoint_path(run_directory: Path) -> Path:
    best = run_directory / "checkpoint-best.pt"
    return best if best.is_file() else run_directory / "checkpoint-latest.pt"


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
