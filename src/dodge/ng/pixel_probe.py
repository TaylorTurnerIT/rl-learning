"""Supervised probe for pixel perception under the exact waypoint controller."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from dodge.control import PROJECT_ROOT, ControlRuntimeError
from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import NativeBatchEnvironment
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest
from dodge.ng.pixel_dqn import PaletteSpatialEncoder
from dodge.rl.ppo import PixelFeatureEncoder

ProbeArchitecture = Literal["scalar-fast", "palette-spatial"]
ProbeCollectionPolicy = Literal["random", "oracle", "mixed"]


@dataclass(frozen=True, slots=True)
class PixelProbeConfig:
    states_per_seed: int = 12
    max_decisions_per_seed: int = 800
    pixel_stack: int = 4
    step_frames: int = 4
    oracle_hold_decisions: int = 16
    grid_spacing: int = 32
    steering_tolerance: float = 2.0
    min_margin: float = 1.0
    validation_fraction: float = 0.2
    batch_size: int = 128
    epochs: int = 30
    tiny_epochs: int = 120
    hidden_size: int = 128
    learning_rate: float = 3e-4
    native_lanes: int = 8
    seed: int = 2_026_0906
    device: str = "auto"
    progress_seconds: float = 10.0
    collection_policy: ProbeCollectionPolicy = "random"
    oracle_random_fraction: float = 0.2

    def validate(self) -> None:
        if any(
            value < 1
            for value in (
                self.states_per_seed,
                self.max_decisions_per_seed,
                self.pixel_stack,
                self.oracle_hold_decisions,
                self.grid_spacing,
                self.batch_size,
                self.epochs,
                self.tiny_epochs,
                self.hidden_size,
                self.native_lanes,
            )
        ):
            raise ValueError("pixel probe counts must be positive")
        if not 3 <= self.step_frames <= 5:
            raise ValueError("pixel probe step frames must be between 3 and 5")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("pixel probe validation fraction must be in (0, 0.5)")
        if self.min_margin < 0 or not math.isfinite(self.min_margin):
            raise ValueError("pixel probe margin must be finite and non-negative")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("pixel probe device is invalid")
        if not math.isfinite(self.progress_seconds) or self.progress_seconds <= 0:
            raise ValueError("pixel probe progress interval must be positive")
        if self.collection_policy not in {"random", "oracle", "mixed"}:
            raise ValueError("pixel probe collection policy is invalid")
        if not 0 <= self.oracle_random_fraction <= 1:
            raise ValueError("pixel probe random fraction must be between 0 and 1")


class PaletteSpatialClassifier(nn.Module):
    """Learn palette identity and retain a 4x4 spatial map."""

    def __init__(self, stack_size: int, hidden_size: int) -> None:
        super().__init__()
        self.features = PaletteSpatialEncoder(stack_size, hidden_size)
        self.head = nn.Linear(hidden_size, len(ACTION_CHOICES))

    def forward(self, pixels: Tensor) -> Tensor:
        return self.head(self.features(pixels))


class ScalarFastClassifier(nn.Module):
    def __init__(self, stack_size: int, hidden_size: int) -> None:
        super().__init__()
        self.features = PixelFeatureEncoder(stack_size, hidden_size, "fast")
        self.head = nn.Linear(hidden_size, len(ACTION_CHOICES))

    def forward(self, pixels: Tensor) -> Tensor:
        return self.head(self.features(pixels))


def _model(architecture: ProbeArchitecture, config: PixelProbeConfig) -> nn.Module:
    if architecture == "scalar-fast":
        return ScalarFastClassifier(config.pixel_stack, config.hidden_size)
    if architecture == "palette-spatial":
        return PaletteSpatialClassifier(config.pixel_stack, config.hidden_size)
    raise ValueError(f"unknown probe architecture: {architecture}")


def _split_training_seeds(
    manifest: SeedManifest, config: PixelProbeConfig
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    seeds = np.asarray(manifest.training_seeds, dtype=np.uint32)
    np.random.default_rng(config.seed).shuffle(seeds)
    validation_count = max(1, round(len(seeds) * config.validation_fraction))
    return tuple(map(int, seeds[validation_count:])), tuple(
        map(int, seeds[:validation_count])
    )


def collect_probe_dataset(
    manifest: SeedManifest, config: PixelProbeConfig
) -> dict[str, np.ndarray]:
    """Collect decisive training-seed states and exact waypoint scores."""
    config.validate()
    train_seeds, validation_seeds = _split_training_seeds(manifest, config)
    all_seeds = (*train_seeds, *validation_seeds)
    seed_role = {seed: 0 for seed in train_seeds} | {
        seed: 1 for seed in validation_seeds
    }
    counts = {seed: 0 for seed in all_seeds}
    decisions = {seed: 0 for seed in all_seeds}
    pixels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    seeds: list[int] = []
    roles: list[int] = []
    frames: list[int] = []
    rng = np.random.default_rng(config.seed)
    started = time.monotonic()
    last_progress = started
    lane_count = min(config.native_lanes, len(all_seeds))
    cursor = lane_count
    lane_seeds = list(all_seeds[:lane_count])
    with NativeBatchEnvironment(
        step_frames=config.step_frames,
        execution="parallel",
        full_state=True,
        pixels=True,
        board=False,
        ml_grid_spacing=config.grid_spacing,
    ) as environment:
        result = environment.reset_batch_with_startup(lane_seeds)
        current_snapshots = list(result.snapshot_bytes)
        current_frames = result.frames.copy()
        stacks = np.repeat(result.pixels[:, None], config.pixel_stack, axis=1).astype(
            np.uint8, copy=False
        )
        while any(
            counts[seed] < config.states_per_seed
            and decisions[seed] < config.max_decisions_per_seed
            for seed in all_seeds
        ):
            snapshot_values = [
                value for value in current_snapshots if value is not None
            ]
            if len(snapshot_values) != lane_count:
                raise ControlRuntimeError("pixel probe requires one snapshot per lane")
            oracle = environment.score_waypoint_actions(
                snapshot_values,
                config.oracle_hold_decisions,
                grid_spacing=config.grid_spacing,
                tolerance=config.steering_tolerance,
            )
            ordered = np.sort(oracle, axis=1)
            margins = ordered[:, -1] - ordered[:, -2]
            for lane, seed in enumerate(lane_seeds):
                decisions[seed] += 1
                if (
                    counts[seed] < config.states_per_seed
                    and margins[lane] >= config.min_margin
                ):
                    pixels.append(stacks[lane].copy())
                    scores.append(oracle[lane].copy())
                    seeds.append(seed)
                    roles.append(seed_role[seed])
                    frames.append(int(current_frames[lane]))
                    counts[seed] += 1
            random_actions = rng.integers(
                0, len(ACTION_CHOICES), size=lane_count, dtype=np.uint8
            )
            oracle_actions = oracle.argmax(axis=1).astype(np.uint8)
            if config.collection_policy == "random":
                actions = random_actions
            elif config.collection_policy == "oracle":
                actions = oracle_actions
            else:
                random_mask = rng.random(lane_count) < config.oracle_random_fraction
                actions = np.where(random_mask, random_actions, oracle_actions).astype(
                    np.uint8
                )
            result = environment.step_batch(actions)
            current_snapshots = list(result.snapshot_bytes)
            current_frames = result.frames.copy()
            stacks = np.concatenate((stacks[:, 1:], result.pixels[:, None]), axis=1)
            reset_lanes = [
                lane
                for lane, seed in enumerate(lane_seeds)
                if result.done[lane]
                or counts[seed] >= config.states_per_seed
                or decisions[seed] >= config.max_decisions_per_seed
            ]
            if reset_lanes:
                reset_seeds = []
                for lane in reset_lanes:
                    old_seed = lane_seeds[lane]
                    candidate = old_seed
                    for _ in range(len(all_seeds)):
                        candidate = all_seeds[cursor % len(all_seeds)]
                        cursor += 1
                        if (
                            counts[candidate] < config.states_per_seed
                            and decisions[candidate] < config.max_decisions_per_seed
                        ):
                            break
                    lane_seeds[lane] = candidate
                    reset_seeds.append(candidate)
                reset = environment.reset_lanes_with_startup(reset_lanes, reset_seeds)
                result_pixels = reset.pixels
                for position, lane in enumerate(reset_lanes):
                    stacks[lane] = np.repeat(
                        result_pixels[position][None], config.pixel_stack, axis=0
                    )
                # Refresh full-state snapshots for all lanes without advancing.
                # Reset results cover selected lanes; unchanged lane snapshots remain
                # in the prior step result.
                for position, lane in enumerate(reset_lanes):
                    current_snapshots[lane] = reset.snapshot_bytes[position]
                    current_frames[lane] = reset.frames[position]
            now = time.monotonic()
            if now - last_progress >= config.progress_seconds:
                completed = sum(
                    counts[seed] >= config.states_per_seed
                    or decisions[seed] >= config.max_decisions_per_seed
                    for seed in all_seeds
                )
                print(
                    "pixel-probe collect: "
                    f"examples={len(pixels)} seeds={completed}/{len(all_seeds)} "
                    f"elapsed={now - started:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                last_progress = now
        if not pixels:
            raise ControlRuntimeError("pixel probe found no decisive states")
    return {
        "pixels": np.stack(pixels),
        "scores": np.stack(scores),
        "labels": np.asarray(scores).argmax(axis=1).astype(np.int64),
        "seeds": np.asarray(seeds, dtype=np.uint32),
        "roles": np.asarray(roles, dtype=np.uint8),
        "frames": np.asarray(frames, dtype=np.uint32),
    }


def _evaluate(
    model: nn.Module,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(data["pixels"][indices]).to(device))
    choices = logits.argmax(1).cpu().numpy()
    labels = data["labels"][indices]
    scores = data["scores"][indices]
    regret = scores.max(axis=1) - scores[np.arange(len(indices)), choices]
    return {
        "count": float(len(indices)),
        "accuracy": float(np.mean(choices == labels)),
        "zero_regret_fraction": float(np.mean(regret == 0)),
        "mean_oracle_regret_frames": float(regret.mean()),
    }


def train_probe(
    data: dict[str, np.ndarray],
    config: PixelProbeConfig,
    architecture: ProbeArchitecture,
) -> dict[str, object]:
    resolved_device = (
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if config.device == "auto"
        else config.device
    )
    device = torch.device(resolved_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ControlRuntimeError(
            "pixel probe requested CUDA but no CUDA device is available"
        )
    torch.manual_seed(config.seed)
    train_indices = np.flatnonzero(data["roles"] == 0)
    validation_indices = np.flatnonzero(data["roles"] == 1)
    if not len(train_indices) or not len(validation_indices):
        raise ControlRuntimeError("pixel probe seed split produced an empty dataset")
    model = _model(architecture, config).to(device)
    counts = np.bincount(data["labels"][train_indices], minlength=len(ACTION_CHOICES))
    weights = np.divide(
        counts.sum(),
        np.maximum(counts, 1) * len(ACTION_CHOICES),
    )
    loss_function = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights.astype(np.float32)).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    started = time.monotonic()
    for _epoch in range(config.epochs):
        order = rng.permutation(train_indices)
        model.train()
        for offset in range(0, len(order), config.batch_size):
            batch = order[offset : offset + config.batch_size]
            logits = model(torch.from_numpy(data["pixels"][batch]).to(device))
            labels = torch.from_numpy(data["labels"][batch]).to(device)
            loss = loss_function(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    result = {
        "architecture": architecture,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seconds": time.monotonic() - started,
        "train": _evaluate(model, data, train_indices, device),
        "validation": _evaluate(model, data, validation_indices, device),
    }
    tiny = train_indices[: min(128, len(train_indices))]
    tiny_model = _model(architecture, config).to(device)
    tiny_optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    for _epoch in range(config.tiny_epochs):
        logits = tiny_model(torch.from_numpy(data["pixels"][tiny]).to(device))
        labels = torch.from_numpy(data["labels"][tiny]).to(device)
        loss = nn.functional.cross_entropy(logits, labels)
        tiny_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        tiny_optimizer.step()
    result["tiny_overfit"] = _evaluate(tiny_model, data, tiny, device)
    return result


def run_probe(
    config: PixelProbeConfig,
    output_directory: Path,
    manifest: SeedManifest,
) -> dict[str, object]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_directory.mkdir(parents=True, exist_ok=False)
    data = collect_probe_dataset(manifest, config)
    np.savez_compressed(output_directory / "dataset.npz", **data)
    results = [
        train_probe(data, config, architecture)
        for architecture in ("scalar-fast", "palette-spatial")
    ]
    report = {
        "schema_version": 1,
        "manifest_sha256": manifest.sha256,
        "holdout_used": False,
        "config": asdict(config),
        "examples": len(data["labels"]),
        "train_seed_count": len(set(data["seeds"][data["roles"] == 0].tolist())),
        "validation_seed_count": len(set(data["seeds"][data["roles"] == 1].tolist())),
        "architectures": results,
    }
    (output_directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-pixel-probe")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "history" / "dodge" / "ng" / "pixel-probe",
    )
    parser.add_argument("--states-per-seed", type=int, default=12)
    parser.add_argument("--max-decisions-per-seed", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--tiny-epochs", type=int, default=120)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress-seconds", type=float, default=10.0)
    parser.add_argument(
        "--collection-policy",
        choices=("random", "oracle", "mixed"),
        default="random",
    )
    parser.add_argument("--oracle-random-fraction", type=float, default=0.2)
    arguments = parser.parse_args(argv)
    config = PixelProbeConfig(
        states_per_seed=arguments.states_per_seed,
        max_decisions_per_seed=arguments.max_decisions_per_seed,
        epochs=arguments.epochs,
        tiny_epochs=arguments.tiny_epochs,
        device=arguments.device,
        progress_seconds=arguments.progress_seconds,
        collection_policy=arguments.collection_policy,
        oracle_random_fraction=arguments.oracle_random_fraction,
    )
    report = run_probe(config, arguments.output_dir, load_manifest(arguments.manifest))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
