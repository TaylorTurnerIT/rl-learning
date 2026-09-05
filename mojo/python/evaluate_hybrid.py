"""Evaluate a saved Mojo/PyTorch hybrid waypoint-DQN checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dodge.ng.dqn import DQNConfig, DuelingWaypointDQN, evaluate_waypoint_dqn
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, load_manifest


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    manifest = load_manifest(args.manifest)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Mojo checkpoint must contain an object")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Mojo checkpoint has no model_state_dict")

    model = DuelingWaypointDQN(hidden_size=args.hidden_size)
    model.load_state_dict(state)
    model.eval()
    config = DQNConfig(
        total_steps=1,
        batch_size=1,
        replay_capacity=1,
        learning_rate=1e-4,
        gamma=0.99,
        n_step=3,
        warmup_steps=1,
        train_frequency=1,
        target_update_interval=1,
        hidden_size=args.hidden_size,
        grid_spacing=args.grid_spacing,
        hold_decisions=args.hold_decisions,
        step_frames=args.step_frames,
        max_episode_steps=args.max_episode_steps,
        native_lanes=args.lanes,
        native_execution="parallel",
        reset_mode="native-startup",
        checkpoint_every=1,
        eval_every=1,
        seed=args.seed,
        device="cpu",
    )
    config.validate()
    if args.split == "training":
        seeds = manifest.training_seeds
    elif args.split == "holdout":
        seeds = manifest.holdout_seeds
    else:
        seeds = manifest.sample_space
    result = evaluate_waypoint_dqn(model, seeds, config)
    return {
        "backend": "mojo-collection-python-pytorch-evaluation",
        "checkpoint": str(args.checkpoint),
        "manifest_sha256": manifest.sha256,
        "split": args.split,
        "summary": result["summary"],
        "seeds": result["seeds"],
        "survival_frames": result["survival_frames"],
        "terminated": result["terminated"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluate-mojo-hybrid")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--split", choices=("training", "holdout", "all"), default="all"
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--grid-spacing", type=int, default=32)
    parser.add_argument("--hold-decisions", type=int, default=8)
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=2_000)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2_026_0903)
    args = parser.parse_args(argv)
    print(json.dumps(evaluate(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
