"""Run the current Python waypoint-DQN hot loop without campaign overhead."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dodge.native.batch import NativeBatchEnvironment
from dodge.ng.dqn import (
    DQNConfig,
    DuelingWaypointDQN,
    NStepAccumulator,
    ReplayBuffer,
    _collect_macro_transition,
    _learn_step,
    _native_ml_state,
    _seed_everything,
)
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from dodge.ng.waypoint import WaypointController, WaypointGrid


def run(args: argparse.Namespace) -> dict[str, object]:
    config = DQNConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        warmup_steps=args.warmup,
        grid_spacing=args.grid_spacing,
        hold_decisions=args.hold_decisions,
        step_frames=args.step_frames,
        max_episode_steps=args.max_episode_steps,
        native_lanes=args.lanes,
        native_execution="serial" if args.serial else "parallel",
        seed=args.seed,
    )
    config.validate()
    manifest = load_manifest(args.manifest)
    manifest.validate()
    if len(manifest.training_seeds) < config.native_lanes:
        raise ValueError("native lane count exceeds training seed count")

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_interop_threads)
    _seed_everything(config.seed)
    device = torch.device("cpu")
    grid = WaypointGrid(config.grid_spacing)
    controller = WaypointController(grid)
    model = DuelingWaypointDQN(hidden_size=config.hidden_size)
    target_model = DuelingWaypointDQN(hidden_size=config.hidden_size)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity, model.input_size)
    accumulator = NStepAccumulator(
        config.native_lanes,
        config.n_step,
        config.gamma,
        replay,
    )
    rng = np.random.default_rng(config.seed)
    environment = NativeBatchEnvironment(
        step_frames=config.step_frames,
        execution=config.native_execution,
        full_state=False,
        pixels=False,
        board=False,
        ml=True,
        ml_grid_spacing=config.grid_spacing,
    )
    episode_steps = np.zeros(config.native_lanes, dtype=np.int64)
    seed_cursor = 0
    total_native_steps = 0
    learner_updates = 0
    step = 0
    try:
        initial_seeds = np.asarray(
            manifest.training_seeds[: config.native_lanes], dtype=np.uint32
        )
        result = environment.reset_ml_batch_with_startup(initial_seeds)
        current_observations, current_positions = _native_ml_state(result)
        current_observations = current_observations.copy()
        current_positions = current_positions.copy()
        started = time.perf_counter()
        while step < config.total_steps:
            (
                current_observations,
                current_positions,
                seed_cursor,
                native_steps,
                _collection,
            ) = _collect_macro_transition(
                environment,
                current_observations,
                current_positions,
                controller,
                model,
                config,
                episode_steps,
                manifest.training_seeds,
                seed_cursor,
                accumulator,
                    rng,
                    device,
                    step,
            )
            total_native_steps += native_steps
            step += 1
            if (
                not args.no_learning
                and step >= config.warmup_steps
                and replay.size >= config.batch_size
            ):
                _learn_step(
                    model,
                    target_model,
                    optimizer,
                    replay,
                    config,
                    rng,
                    device,
                )
                learner_updates += 1
            if step % config.target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())
        elapsed = time.perf_counter() - started
    finally:
        environment.close()

    checksum = float(
        np.sum(
            current_observations.reshape(-1)
            * np.arange(current_observations.size, dtype=np.float32)
            + current_observations.reshape(-1)
        )
    )
    return {
        "backend": "python-pytorch-hot-loop",
        "manifest_sha256": manifest.sha256,
        "lanes": config.native_lanes,
        "collection_steps": step,
        "learner_updates": learner_updates,
        "native_steps": total_native_steps,
        "replay": replay.size,
        "elapsed_s": elapsed,
        "steps_per_s": step / max(elapsed, 1e-9),
        "native_steps_per_s": total_native_steps / max(elapsed, 1e-9),
        "checksum": checksum,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="benchmark-python-hot-loop")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--grid-spacing", type=int, default=32)
    parser.add_argument("--hold-decisions", type=int, default=8)
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2_026_0903)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--no-learning", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
