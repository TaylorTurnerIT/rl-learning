from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from dodge.control import PROJECT_ROOT, ControlRuntimeError
from dodge.dataset import ACTION_CHOICES
from dodge.native.assets import PICO8_PALETTE
from dodge.native.batch import NativeBatchEnvironment, NativeBatchResult
from dodge.native.differential import NativeSnapshot, decode_native_snapshot
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from dodge.ng.teacher import CounterfactualCache
from dodge.rl.ppo import _advance_pixel_stack, _initial_pixel_stack

RELEVANCE_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "history" / "dodge" / "ng" / "relevance-audit"
DEFAULT_LOOKAHEAD_STEPS = (8, 16, 32, 64)
FRAME_WIDTH = 128
FRAME_HEIGHT = 128
GAME_MODE = "game"

Partition = Literal["training", "holdout"]
Execution = Literal["serial", "parallel"]
ObservationMode = Literal["board", "board_full", "pixels", "none"]


@dataclass(frozen=True, slots=True)
class RelevanceConfig:
    """Configuration for a read-only policy decision-relevance audit."""

    manifest_path: Path = DEFAULT_MANIFEST_PATH
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
    checkpoint: Path | None = None
    action: str | None = None
    partition: Partition = "training"
    lookahead_steps: tuple[int, ...] = DEFAULT_LOOKAHEAD_STEPS
    gate_lookahead_steps: int = DEFAULT_LOOKAHEAD_STEPS[0]
    step_frames: int = 4
    sample_every: int = 4
    max_samples_per_seed: int = 64
    max_episode_steps: int = 2_000
    margin_threshold: float = 4.0
    min_decisive_fraction: float = 0.10
    min_decisive_seeds: int = 1
    near_collision_horizon_frames: float = 32.0
    native_lanes: int = 32
    native_execution: Execution = "parallel"
    visual_seeds: tuple[int, ...] = ()
    difficulty: int = 2
    patterns_enabled: bool = True
    powerups_enabled: bool = True

    def validate(self) -> None:
        if (self.checkpoint is None) == (self.action is None):
            raise ValueError("provide exactly one of checkpoint or action")
        if self.action is not None and self.action not in ACTION_CHOICES:
            raise ValueError(f"unknown fixed action: {self.action}")
        if self.partition not in {"training", "holdout"}:
            raise ValueError("partition must be training or holdout")
        if not self.lookahead_steps or any(
            isinstance(value, bool) or value < 1 for value in self.lookahead_steps
        ):
            raise ValueError("lookahead steps must be positive")
        if tuple(sorted(set(self.lookahead_steps))) != self.lookahead_steps:
            raise ValueError("lookahead steps must be sorted and unique")
        if self.gate_lookahead_steps not in self.lookahead_steps:
            raise ValueError("gate lookahead must be one configured horizon")
        if not 3 <= self.step_frames <= 5:
            raise ValueError("step frames must be between 3 and 5")
        if self.sample_every < 1 or self.max_samples_per_seed < 1:
            raise ValueError("sampling limits must be positive")
        if self.max_episode_steps < 1:
            raise ValueError("maximum episode steps must be positive")
        if self.margin_threshold < 0:
            raise ValueError("margin threshold must not be negative")
        if not 0 <= self.min_decisive_fraction <= 1:
            raise ValueError("minimum decisive fraction must be between 0 and 1")
        if self.min_decisive_seeds < 1 or self.native_lanes < 1:
            raise ValueError("seed and lane limits must be positive")
        if self.near_collision_horizon_frames < 0:
            raise ValueError("near-collision horizon must not be negative")
        if self.native_execution not in {"serial", "parallel"}:
            raise ValueError("native execution must be serial or parallel")
        if len(set(self.visual_seeds)) != len(self.visual_seeds):
            raise ValueError("visual seeds must be unique")
        if not 1 <= self.difficulty <= 3:
            raise ValueError("difficulty must be between 1 and 3")

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        value["manifest_path"] = str(self.manifest_path)
        value["output_directory"] = str(self.output_directory)
        value["checkpoint"] = None if self.checkpoint is None else str(self.checkpoint)
        value["lookahead_steps"] = list(self.lookahead_steps)
        value["visual_seeds"] = list(self.visual_seeds)
        return value


@dataclass(slots=True)
class _PathMetrics:
    seed: int
    actions: list[int] = field(default_factory=list)
    first_enemy_frame: int | None = None
    first_aoe_frame: int | None = None
    first_pattern_frame: int | None = None
    first_near_collision_frame: int | None = None
    first_action_change_frame: int | None = None
    terminal_frame: int | None = None
    truncated: bool = False
    action_changes: int = 0
    max_action_run: int = 0
    current_action: int | None = None
    current_run: int = 0
    sample_count: int = 0
    sampled_frames: set[int] = field(default_factory=set)
    first_decisive_frames: dict[int, int | None] = field(default_factory=dict)
    visual_checkpoints: dict[str, dict[str, object]] = field(default_factory=dict)

    def add_action(self, action: int, frame: int) -> None:
        self.actions.append(action)
        if self.current_action == action:
            self.current_run += 1
        else:
            if self.current_action is not None:
                self.action_changes += 1
                if self.first_action_change_frame is None:
                    self.first_action_change_frame = frame
            self.current_action = action
            self.current_run = 1
        self.max_action_run = max(self.max_action_run, self.current_run)

    def to_json(self, lookahead_steps: Sequence[int]) -> dict[str, object]:
        action_counts = Counter(ACTION_CHOICES[action] for action in self.actions)
        total = len(self.actions)
        entropy = 0.0
        if total:
            entropy = float(
                -sum(
                    (count / total) * math.log2(count / total)
                    for count in action_counts.values()
                )
            )
        return {
            "seed": self.seed,
            "decision_count": total,
            "unique_actions": len(action_counts),
            "first_action": (
                None if not self.actions else ACTION_CHOICES[self.actions[0]]
            ),
            "action_counts": dict(action_counts),
            "action_entropy": entropy,
            "action_changes": self.action_changes,
            "action_change_rate": (
                self.action_changes / max(total - 1, 1) if total else 0.0
            ),
            "max_action_run": self.max_action_run,
            "first_enemy_frame": self.first_enemy_frame,
            "first_aoe_frame": self.first_aoe_frame,
            "first_pattern_frame": self.first_pattern_frame,
            "first_near_collision_frame": self.first_near_collision_frame,
            "first_action_change_frame": self.first_action_change_frame,
            "first_decisive_frames": {
                str(steps): self.first_decisive_frames.get(steps)
                for steps in lookahead_steps
            },
            "terminal_frame": self.terminal_frame,
            "truncated": self.truncated,
            "sample_count": self.sample_count,
            "visual_checkpoints": self.visual_checkpoints,
        }


@dataclass(slots=True)
class _Sample:
    seed: int
    frame: int
    action_index: int
    enemy_count: int
    aoe_count: int
    pattern_active: bool
    nearest_gap: float | None
    time_to_intersection: float | None
    pixel_hash: int
    snapshot: bytes
    horizons: dict[int, dict[str, object]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _PolicyInfo:
    label: str
    observation_mode: ObservationMode
    pixel_stack: int = 1


class _Policy:
    info: _PolicyInfo

    def actions(
        self,
        pixels: np.ndarray | None,
        pixel_stack: np.ndarray | None,
        board: np.ndarray | None,
    ) -> np.ndarray:
        raise NotImplementedError


class _FixedPolicy(_Policy):
    def __init__(self, action: str) -> None:
        self.action_index = ACTION_CHOICES.index(action)
        self.info = _PolicyInfo(f"fixed:{action}", "none")

    def actions(
        self,
        pixels: np.ndarray | None,
        pixel_stack: np.ndarray | None,
        board: np.ndarray | None,
    ) -> np.ndarray:
        if pixels is None:
            raise ControlRuntimeError("fixed policy requires native lane count")
        return np.full(pixels.shape[0], self.action_index, dtype=np.uint8)


class _TorchPolicy(_Policy):
    def __init__(self, model: object, info: _PolicyInfo) -> None:
        self.model = model
        self.info = info

    def actions(
        self,
        pixels: np.ndarray | None,
        pixel_stack: np.ndarray | None,
        board: np.ndarray | None,
    ) -> np.ndarray:
        import torch

        if self.info.observation_mode == "pixels":
            if pixel_stack is None:
                raise ControlRuntimeError(
                    "pixel checkpoint requires pixel observations"
                )
            observations = pixel_stack
        elif board is not None:
            observations = board
        else:
            raise ControlRuntimeError("board checkpoint requires board observations")
        with torch.inference_mode():
            logits, _ = self.model(torch.from_numpy(observations))
        return logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8, copy=True)


def _load_policy(config: RelevanceConfig) -> _Policy:
    if config.action is not None:
        return _FixedPolicy(config.action)
    if config.checkpoint is None:
        raise ValueError("checkpoint policy is missing a path")
    try:
        import torch

        from dodge.rl.ppo import DodgeActorCriticCNN, PixelActorCriticCNN

        payload = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ControlRuntimeError(
            f"could not load relevance checkpoint {config.checkpoint}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("relevance checkpoint must contain an object")
    actions = payload.get("actions")
    if tuple(actions or ()) != ACTION_CHOICES:
        raise ValueError("relevance checkpoint action contract is invalid")
    mode = payload.get("observation_mode")
    stored_config = payload.get("config")
    if not isinstance(stored_config, Mapping):
        raise ValueError("relevance checkpoint configuration is missing")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("relevance checkpoint model state is missing")
    if mode == "pixels":
        stack_size = int(stored_config.get("pixel_stack", 4))
        architecture = stored_config.get("pixel_architecture", "small")
        if architecture not in {"fast", "small", "current"}:
            raise ValueError("relevance pixel architecture is invalid")
        weight = state_dict.get("features.projection.0.weight")
        hidden_size = int(weight.shape[0]) if hasattr(weight, "shape") else 128
        model = PixelActorCriticCNN(
            stack_size=stack_size,
            hidden_size=hidden_size,
            architecture=architecture,
        )
        info = _PolicyInfo(f"checkpoint:{config.checkpoint}", "pixels", stack_size)
    elif mode in {"board", "board_full"}:
        weight = state_dict.get("features.projection.0.weight")
        hidden_size = int(weight.shape[0]) if hasattr(weight, "shape") else 256
        model = DodgeActorCriticCNN(hidden_size=hidden_size)
        info = _PolicyInfo(f"checkpoint:{config.checkpoint}", mode)
    else:
        raise ValueError("relevance checkpoint observation mode is invalid")
    try:
        model.load_state_dict(state_dict)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            f"relevance checkpoint weights are invalid: {error}"
        ) from error
    model.eval()
    return _TorchPolicy(model, info)


def build_decision_relevance_audit(config: RelevanceConfig) -> dict[str, object]:
    """Run policy replay and write machine-readable and human-readable artifacts."""

    config.validate()
    manifest = load_manifest(config.manifest_path)
    seeds = (
        manifest.training_seeds
        if config.partition == "training"
        else manifest.holdout_seeds
    )
    if not seeds:
        raise ValueError("selected manifest partition is empty")
    if any(seed not in seeds for seed in config.visual_seeds):
        raise ValueError("visual seeds must belong to selected partition")
    policy = _load_policy(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    paths, samples, cache = _collect_rollouts(config, seeds, policy)
    horizon_summaries, nonmutation_verified = _score_samples(
        config,
        samples,
        paths,
        cache,
    )
    gate = _build_gate(config, horizon_summaries, paths, len(samples))
    diagnostic: dict[str, object] = {
        "schema_version": RELEVANCE_SCHEMA_VERSION,
        "kind": "dodge_ng_decision_relevance_audit",
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.sha256,
        "partition": config.partition,
        "seed_scope": "training_only"
        if config.partition == "training"
        else "holdout_report_only",
        "seeds": list(seeds),
        "sample_count": len(samples),
        "policy": {
            "label": policy.info.label,
            "observation_mode": policy.info.observation_mode,
            "pixel_stack": policy.info.pixel_stack,
        },
        "config": config.to_json(),
        "counterfactual_cache": cache.to_json(),
        "source_state_nonmutation_verified": nonmutation_verified,
        "gate": gate,
        "horizons": horizon_summaries,
        "paths": [
            path.to_json(config.lookahead_steps)
            for path in sorted(paths.values(), key=lambda item: item.seed)
        ],
    }
    _write_json(config.output_directory / "relevance.json", diagnostic)
    (config.output_directory / "RELEVANCE.md").write_text(
        _markdown(diagnostic), encoding="utf-8"
    )
    _write_plots(config.output_directory, diagnostic)
    return diagnostic


def _collect_rollouts(
    config: RelevanceConfig,
    seeds: Sequence[int],
    policy: _Policy,
) -> tuple[dict[int, _PathMetrics], list[_Sample], CounterfactualCache]:
    paths = {int(seed): _PathMetrics(int(seed)) for seed in seeds}
    samples: list[_Sample] = []
    cache = CounterfactualCache()
    visual_seeds = set(config.visual_seeds)
    for start in range(0, len(seeds), config.native_lanes):
        local_seeds = tuple(
            int(seed) for seed in seeds[start : start + config.native_lanes]
        )
        local_paths, local_samples = _collect_batch(
            config,
            local_seeds,
            policy,
            visual_seeds,
        )
        paths.update(local_paths)
        samples.extend(local_samples)
    return paths, samples, cache


def _collect_batch(
    config: RelevanceConfig,
    seeds: Sequence[int],
    policy: _Policy,
    visual_seeds: set[int],
) -> tuple[dict[int, _PathMetrics], list[_Sample]]:
    paths = {int(seed): _PathMetrics(int(seed)) for seed in seeds}
    samples: list[_Sample] = []
    board_enabled = policy.info.observation_mode in {"board", "board_full"}
    with NativeBatchEnvironment(
        step_frames=config.step_frames,
        execution=config.native_execution,
        full_state=True,
        pixels=True,
        board=board_enabled,
        include_offscreen_board=policy.info.observation_mode == "board_full",
        difficulty=config.difficulty,
        patterns_enabled=config.patterns_enabled,
        powerups_enabled=config.powerups_enabled,
    ) as environment:
        result = environment.reset_batch(np.asarray(seeds, dtype=np.uint32))
        current_snapshots = list(result.snapshot_bytes)
        current_pixels = _require_pixels(result)
        current_pixel_stack = (
            _initial_pixel_stack(current_pixels, policy.info.pixel_stack)
            if policy.info.observation_mode == "pixels"
            else None
        )
        current_board = _copy_optional(result.board) if board_enabled else None
        active = np.ones(len(seeds), dtype=bool)
        game_decisions = [0] * len(seeds)
        for _ in range(config.max_episode_steps):
            if not active.any():
                break
            actions = policy.actions(current_pixels, current_pixel_stack, current_board)
            if actions.shape != (len(seeds),):
                raise ControlRuntimeError(
                    "relevance policy returned invalid action shape"
                )
            actions = np.asarray(actions, dtype=np.uint8)
            actions[~active] = 0
            decoded = [
                _decode_optional_snapshot(snapshot) for snapshot in current_snapshots
            ]
            for lane, is_active in enumerate(active):
                if not is_active:
                    continue
                snapshot = decoded[lane]
                if snapshot is None or snapshot.mode != GAME_MODE:
                    continue
                seed = int(seeds[lane])
                path = paths[seed]
                frame = int(snapshot.frame)
                action = int(actions[lane])
                path.add_action(action, frame)
                metrics = _hazard_metrics(snapshot)
                first_enemy = (
                    metrics["enemy_count"] > 0 and path.first_enemy_frame is None
                )
                first_aoe = metrics["aoe_count"] > 0 and path.first_aoe_frame is None
                first_pattern = (
                    bool(metrics["pattern_active"]) and path.first_pattern_frame is None
                )
                first_near = (
                    metrics["time_to_intersection"] is not None
                    and metrics["time_to_intersection"]
                    <= config.near_collision_horizon_frames
                    and path.first_near_collision_frame is None
                )
                if first_enemy:
                    path.first_enemy_frame = frame
                if first_aoe:
                    path.first_aoe_frame = frame
                if first_pattern:
                    path.first_pattern_frame = frame
                if first_near:
                    path.first_near_collision_frame = frame
                force_sample = first_enemy or first_aoe or first_pattern or first_near
                regular_sample = (
                    game_decisions[lane] % config.sample_every == 0
                    and path.sample_count < config.max_samples_per_seed
                )
                if frame not in path.sampled_frames and (
                    regular_sample or force_sample
                ):
                    path.sampled_frames.add(frame)
                    samples.append(
                        _Sample(
                            seed=seed,
                            frame=frame,
                            action_index=action,
                            enemy_count=int(metrics["enemy_count"]),
                            aoe_count=int(metrics["aoe_count"]),
                            pattern_active=bool(metrics["pattern_active"]),
                            nearest_gap=metrics["nearest_gap"],
                            time_to_intersection=metrics["time_to_intersection"],
                            pixel_hash=int(result.pixel_hashes[lane]),
                            snapshot=current_snapshots[lane],
                        )
                    )
                    path.sample_count += 1
                if seed in visual_seeds:
                    _maybe_save_visual(
                        config.output_directory,
                        path,
                        "opening",
                        seed,
                        frame,
                        current_pixels[lane],
                        int(result.pixel_hashes[lane]),
                        True,
                    )
                    _maybe_save_visual(
                        config.output_directory,
                        path,
                        "enemy",
                        seed,
                        frame,
                        current_pixels[lane],
                        int(result.pixel_hashes[lane]),
                        metrics["enemy_count"] > 0,
                    )
                    _maybe_save_visual(
                        config.output_directory,
                        path,
                        "aoe",
                        seed,
                        frame,
                        current_pixels[lane],
                        int(result.pixel_hashes[lane]),
                        metrics["aoe_count"] > 0,
                    )
                    _maybe_save_visual(
                        config.output_directory,
                        path,
                        "pattern",
                        seed,
                        frame,
                        current_pixels[lane],
                        int(result.pixel_hashes[lane]),
                        bool(metrics["pattern_active"]),
                    )
                    _maybe_save_visual(
                        config.output_directory,
                        path,
                        "near-collision",
                        seed,
                        frame,
                        current_pixels[lane],
                        int(result.pixel_hashes[lane]),
                        (
                            metrics["time_to_intersection"] is not None
                            and metrics["time_to_intersection"]
                            <= config.near_collision_horizon_frames
                        ),
                    )
                game_decisions[lane] += 1

            stepped = environment.step_batch(actions)
            next_snapshots = list(stepped.snapshot_bytes)
            next_pixels = _require_pixels(stepped)
            next_pixel_stack = (
                _advance_pixel_stack(current_pixel_stack, next_pixels)
                if current_pixel_stack is not None
                else None
            )
            next_board = _copy_optional(stepped.board) if board_enabled else None
            done_lanes = np.flatnonzero(stepped.done)
            for lane in done_lanes.tolist():
                if not active[lane]:
                    continue
                terminal = _decode_optional_snapshot(next_snapshots[lane])
                if terminal is not None:
                    seed = int(seeds[lane])
                    paths[seed].terminal_frame = int(terminal.frame)
                    if seed in visual_seeds:
                        _maybe_save_visual(
                            config.output_directory,
                            paths[seed],
                            "terminal",
                            seed,
                            int(terminal.frame),
                            next_pixels[lane],
                            int(stepped.pixel_hashes[lane]),
                            True,
                        )
                active[lane] = False

            current_snapshots = next_snapshots
            current_pixels = next_pixels
            current_pixel_stack = next_pixel_stack
            current_board = next_board
            if len(done_lanes):
                reset = environment.reset_lanes(
                    done_lanes.astype(np.uint32),
                    np.zeros(len(done_lanes), dtype=np.uint32),
                )
                reset_pixels = _require_pixels(reset)
                reset_board = _copy_optional(reset.board) if board_enabled else None
                reset_pixel_stack = (
                    _initial_pixel_stack(reset_pixels, policy.info.pixel_stack)
                    if policy.info.observation_mode == "pixels"
                    else None
                )
                for position, lane_value in enumerate(reset.lane_ids.tolist()):
                    lane = int(lane_value)
                    current_snapshots[lane] = reset.snapshot_bytes[position]
                    current_pixels[lane] = reset_pixels[position]
                    if (
                        current_pixel_stack is not None
                        and reset_pixel_stack is not None
                    ):
                        current_pixel_stack[lane] = reset_pixel_stack[position]
                    if current_board is not None and reset_board is not None:
                        current_board[lane] = reset_board[position]
        for lane, is_active in enumerate(active):
            if is_active:
                paths[int(seeds[lane])].truncated = True
    return paths, samples


def _score_samples(
    config: RelevanceConfig,
    samples: Sequence[_Sample],
    paths: Mapping[int, _PathMetrics],
    cache: CounterfactualCache,
) -> tuple[list[dict[str, object]], bool]:
    summaries: list[dict[str, object]] = []
    nonmutation_verified = True
    if not samples:
        return (
            [
                {
                    "lookahead_steps": lookahead_steps,
                    "lookahead_frames": lookahead_steps * config.step_frames,
                    "sample_count": 0,
                    "decisive_count": 0,
                    "decisive_fraction": 0.0,
                    "decisive_seed_count": 0,
                    "action_range": None,
                    "best_second_margin": None,
                    "policy_regret": None,
                    "mean_action_range": 0.0,
                    "mean_policy_regret": 0.0,
                    "max_action_range": 0.0,
                    "best_action_counts": {},
                    "sample_metrics": [],
                }
                for lookahead_steps in config.lookahead_steps
            ],
            nonmutation_verified,
        )
    snapshots = [sample.snapshot for sample in samples]
    with NativeBatchEnvironment(
        step_frames=config.step_frames,
        execution=config.native_execution,
        full_state=False,
        pixels=False,
        board=False,
        difficulty=config.difficulty,
        patterns_enabled=config.patterns_enabled,
        powerups_enabled=config.powerups_enabled,
    ) as environment:
        probe = environment.reset_batch([samples[0].seed])
        before = _result_fingerprint(probe)
        for lookahead_steps in config.lookahead_steps:
            scores = cache.score(environment, snapshots, lookahead_steps)
            sample_metrics: list[dict[str, object]] = []
            for sample, row in zip(samples, scores, strict=True):
                if not np.isfinite(row).all():
                    raise ControlRuntimeError(
                        "relevance scorer returned non-finite scores"
                    )
                ordered = np.sort(row)
                best_index = int(np.argmax(row))
                best_score = float(ordered[-1])
                second_score = float(ordered[-2])
                action_range = best_score - float(ordered[0])
                best_second_margin = best_score - second_score
                policy_regret = best_score - float(row[sample.action_index])
                decisive = action_range > config.margin_threshold
                path = paths[sample.seed]
                if decisive and path.first_decisive_frames.get(lookahead_steps) is None:
                    path.first_decisive_frames[lookahead_steps] = sample.frame
                sample_metrics.append(
                    {
                        "seed": sample.seed,
                        "frame": sample.frame,
                        "action": ACTION_CHOICES[sample.action_index],
                        "best_action": ACTION_CHOICES[best_index],
                        "enemy_count": sample.enemy_count,
                        "aoe_count": sample.aoe_count,
                        "pattern_active": sample.pattern_active,
                        "nearest_gap": sample.nearest_gap,
                        "time_to_intersection": sample.time_to_intersection,
                        "pixel_hash": sample.pixel_hash,
                        "action_range": action_range,
                        "best_second_margin": best_second_margin,
                        "policy_regret": policy_regret,
                        "decisive": decisive,
                    }
                )
                sample.horizons[lookahead_steps] = sample_metrics[-1]
            ranges = [float(row["action_range"]) for row in sample_metrics]
            best_second = [float(row["best_second_margin"]) for row in sample_metrics]
            regrets = [float(row["policy_regret"]) for row in sample_metrics]
            decisive_rows = [row for row in sample_metrics if row["decisive"]]
            best_actions = Counter(str(row["best_action"]) for row in sample_metrics)
            summaries.append(
                {
                    "lookahead_steps": lookahead_steps,
                    "lookahead_frames": lookahead_steps * config.step_frames,
                    "sample_count": len(sample_metrics),
                    "decisive_count": len(decisive_rows),
                    "decisive_fraction": (
                        len(decisive_rows) / len(sample_metrics)
                        if sample_metrics
                        else 0.0
                    ),
                    "decisive_seed_count": len({row["seed"] for row in decisive_rows}),
                    "action_range": _distribution(ranges),
                    "best_second_margin": _distribution(best_second),
                    "policy_regret": _distribution(regrets),
                    "mean_action_range": _mean(ranges),
                    "mean_policy_regret": _mean(regrets),
                    "max_action_range": max(ranges, default=0.0),
                    "best_action_counts": dict(best_actions),
                    "sample_metrics": sample_metrics,
                }
            )
        after = _result_fingerprint(environment.last_result)
        nonmutation_verified = before == after
    return summaries, nonmutation_verified


def _result_fingerprint(result: NativeBatchResult) -> tuple[object, ...]:
    return (
        tuple(np.asarray(result.lane_ids).tolist()),
        tuple(np.asarray(result.frames).tolist()),
        tuple(np.asarray(result.state_hashes).tolist()),
        tuple(np.asarray(result.pixel_hashes).tolist()),
        tuple(result.snapshot_bytes),
    )


def _build_gate(
    config: RelevanceConfig,
    horizons: Sequence[Mapping[str, object]],
    paths: Mapping[int, _PathMetrics],
    sample_count: int,
) -> dict[str, object]:
    primary_steps = config.gate_lookahead_steps
    primary = next(
        (row for row in horizons if row["lookahead_steps"] == primary_steps),
        None,
    )
    if primary is None:
        raise ControlRuntimeError("relevance primary horizon is missing")
    decisive_count = int(primary["decisive_count"])
    decisive_fraction = float(primary["decisive_fraction"])
    decisive_seed_count = int(primary["decisive_seed_count"])
    max_range = float(primary["max_action_range"])
    if config.partition != "training":
        reason = "holdout-only report cannot set selection gate"
        passed = False
    elif sample_count == 0:
        reason = "no game-state samples were collected"
        passed = False
    elif max_range <= config.margin_threshold or decisive_count == 0:
        reason = "all sampled action scores remain within margin threshold"
        passed = False
    elif decisive_fraction < config.min_decisive_fraction:
        reason = "decisive-state fraction is below configured minimum"
        passed = False
    elif decisive_seed_count < config.min_decisive_seeds:
        reason = "too few seeds contain a decisive state"
        passed = False
    else:
        reason = "sampled states contain decision-relevant action differences"
        passed = True
    return {
        "passed": passed,
        "selection_eligible": passed and config.partition == "training",
        "partition": config.partition,
        "primary_lookahead_steps": primary_steps,
        "primary_lookahead_frames": primary_steps * config.step_frames,
        "margin_threshold": config.margin_threshold,
        "min_decisive_fraction": config.min_decisive_fraction,
        "min_decisive_seeds": config.min_decisive_seeds,
        "sample_count": sample_count,
        "decisive_count": decisive_count,
        "decisive_fraction": decisive_fraction,
        "decisive_seed_count": decisive_seed_count,
        "max_action_range": max_range,
        "path_count": len(paths),
        "reason": reason,
    }


def _hazard_metrics(snapshot: NativeSnapshot) -> dict[str, object]:
    player_x, player_y, player_vx, player_vy, player_size = (
        _fixed(value) for value in snapshot.player
    )
    hazards: list[tuple[float, float, float, float, float, float]] = []
    normal_enemies = [enemy for enemy in snapshot.enemies if enemy.personality != -1]
    aoe_enemies = [enemy for enemy in snapshot.enemies if enemy.personality == -1]
    for enemy in snapshot.enemies:
        size = 8.0 if enemy.personality >= 2 else _fixed(enemy.size)
        hazards.append(
            (
                _fixed(enemy.x),
                _fixed(enemy.y),
                _fixed(enemy.vx),
                _fixed(enemy.vy),
                size,
                size,
            )
        )
    pattern_active = snapshot.active_pattern is not None or snapshot.pattern_active
    pattern_rects = ()
    if snapshot.active_pattern is not None:
        pattern = snapshot.patterns[snapshot.active_pattern]
        pattern_rects = tuple(rect for rect in pattern.rects if rect.shown)
        for rect in pattern_rects:
            width = _fixed(rect.width)
            height = _fixed(rect.height)
            hazards.append(
                (
                    _fixed(rect.x) + width / 2,
                    _fixed(rect.y) + height / 2,
                    _fixed(rect.dx),
                    _fixed(rect.dy),
                    width,
                    height,
                )
            )
    gaps: list[float] = []
    times: list[float] = []
    for hazard_x, hazard_y, hazard_vx, hazard_vy, width, height in hazards:
        combined_x = (player_size + width) / 2
        combined_y = (player_size + height) / 2
        gap_x = max(abs(hazard_x - player_x) - combined_x, 0.0)
        gap_y = max(abs(hazard_y - player_y) - combined_y, 0.0)
        gaps.append(math.hypot(gap_x, gap_y))
        times.append(
            max(
                _axis_time_to_intersection(
                    hazard_x - player_x,
                    hazard_vx - player_vx,
                    combined_x,
                ),
                _axis_time_to_intersection(
                    hazard_y - player_y,
                    hazard_vy - player_vy,
                    combined_y,
                ),
            )
        )
    nearest_time = min(times, default=math.inf)
    return {
        "enemy_count": len(normal_enemies),
        "aoe_count": len(aoe_enemies) + len(pattern_rects),
        "pattern_active": pattern_active,
        "nearest_gap": min(gaps) if gaps else None,
        "time_to_intersection": None if math.isinf(nearest_time) else nearest_time,
    }


def _axis_time_to_intersection(
    distance: float, velocity: float, combined_half: float
) -> float:
    if abs(distance) <= combined_half:
        return 0.0
    if velocity == 0.0 or distance * velocity >= 0.0:
        return math.inf
    return max(0.0, (abs(distance) - combined_half) / abs(velocity))


def _fixed(value: int) -> float:
    return value / float(1 << 16)


def _decode_optional_snapshot(value: bytes | None) -> NativeSnapshot | None:
    return None if value is None else decode_native_snapshot(value.hex())


def _require_pixels(result: object) -> np.ndarray:
    pixels = getattr(result, "pixels", None)
    if not isinstance(pixels, np.ndarray):
        raise ControlRuntimeError("relevance audit requires native pixels")
    if pixels.ndim != 3 or pixels.shape[1:] != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ControlRuntimeError("native relevance pixels have invalid shape")
    if pixels.size and (int(pixels.min()) < 0 or int(pixels.max()) > 15):
        raise ControlRuntimeError(
            "native relevance pixels contain invalid palette indexes"
        )
    return np.array(pixels, dtype=np.uint8, copy=True)


def _copy_optional(value: object) -> np.ndarray | None:
    return None if value is None else np.array(value, dtype=np.float32, copy=True)


def _maybe_save_visual(
    output_directory: Path,
    path: _PathMetrics,
    label: str,
    seed: int,
    frame: int,
    pixels: np.ndarray,
    pixel_hash: int,
    condition: bool,
) -> None:
    if not condition or label in path.visual_checkpoints:
        return
    visual_directory = output_directory / "visuals"
    visual_directory.mkdir(parents=True, exist_ok=True)
    relative = Path("visuals") / f"seed-{seed}-{label}-frame-{frame:05d}.png"
    target = output_directory / relative
    _write_visual(target, pixels)
    path.visual_checkpoints[label] = {
        "path": str(relative),
        "frame": frame,
        "pixel_hash": pixel_hash,
    }


def _write_visual(path: Path, pixels: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    palette = np.asarray(PICO8_PALETTE, dtype=np.uint8)
    indexed = np.asarray(pixels, dtype=np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH)
    pyplot.imsave(path, palette[indexed], format="png")
    pyplot.close("all")


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    quantiles = np.percentile(
        np.asarray(values, dtype=np.float64), [0, 10, 50, 90, 100]
    )
    return {
        key: float(value)
        for key, value in zip(
            ("min", "p10", "median", "p90", "max"), quantiles, strict=True
        )
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_plots(output_directory: Path, diagnostic: Mapping[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    horizons = diagnostic.get("horizons")
    if not isinstance(horizons, Sequence) or not horizons:
        return
    frame_values = [int(row["lookahead_frames"]) for row in horizons]
    ranges = [float(row["mean_action_range"]) for row in horizons]
    decisive = [100 * float(row["decisive_fraction"]) for row in horizons]
    regrets = [float(row["mean_policy_regret"]) for row in horizons]
    figure, axis = pyplot.subplots(figsize=(8, 4.5))
    axis.plot(frame_values, ranges, marker="o", label="mean action range")
    axis.plot(frame_values, regrets, marker="o", label="mean policy regret")
    axis.set_xlabel("Additional game frames")
    axis.set_ylabel("Frames survived")
    axis.grid(alpha=0.25)
    secondary = axis.twinx()
    secondary.plot(
        frame_values,
        decisive,
        color="tab:green",
        marker="s",
        label="decisive samples (%)",
    )
    secondary.set_ylabel("Decisive samples (%)")
    lines, labels = axis.get_legend_handles_labels()
    lines2, labels2 = secondary.get_legend_handles_labels()
    axis.legend(lines + lines2, labels + labels2, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_directory / "relevance_horizons.png", dpi=140)
    pyplot.close(figure)

    paths = diagnostic.get("paths")
    if not isinstance(paths, Sequence) or not paths:
        return
    terminal = [
        row["terminal_frame"]
        for row in paths
        if isinstance(row, Mapping) and row.get("terminal_frame") is not None
    ]
    near = [
        row["first_near_collision_frame"]
        for row in paths
        if isinstance(row, Mapping)
        and row.get("first_near_collision_frame") is not None
    ]
    gate = diagnostic.get("gate")
    decisive_steps = (
        gate["primary_lookahead_steps"]
        if isinstance(gate, Mapping)
        else horizons[-1]["lookahead_steps"]
    )
    decisive_key = str(decisive_steps)
    first_decisive = []
    for row in paths:
        if not isinstance(row, Mapping):
            continue
        values = row.get("first_decisive_frames")
        if isinstance(values, Mapping) and values.get(decisive_key) is not None:
            first_decisive.append(values[decisive_key])
    figure, axis = pyplot.subplots(figsize=(8, 4.5))
    labels = ["near collision", "first decisive", "terminal"]
    distributions = [near, first_decisive, terminal]
    axis.boxplot(distributions, tick_labels=labels, showfliers=False)
    axis.set_ylabel("Game frame")
    axis.set_title("Cross-seed hazard and decision timeline")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_directory / "relevance_timeline.png", dpi=140)
    pyplot.close(figure)


def _markdown(diagnostic: Mapping[str, object]) -> str:
    gate = diagnostic["gate"]
    if not isinstance(gate, Mapping):
        raise ValueError("relevance gate must be an object")
    horizons = diagnostic["horizons"]
    if not isinstance(horizons, Sequence):
        raise ValueError("relevance horizons must be a sequence")
    lines = [
        "# Dodge NG decision-relevance audit",
        "",
        f"Manifest: `{diagnostic['manifest_id']}`  ",
        f"Manifest SHA-256: `{diagnostic['manifest_sha256']}`  ",
        f"Partition: `{diagnostic['partition']}`  ",
        f"Policy: `{diagnostic['policy']['label']}`  ",
        f"Samples: {diagnostic['sample_count']}  ",
        f"Gate: **{'PASS' if gate['passed'] else 'FAIL'}** — {gate['reason']}",
        "",
        "This audit scores every action from the same canonical snapshot at "
        "multiple lookahead horizons. It is diagnostic only; it never trains "
        "or selects a checkpoint.",
        "",
        "## Horizon evidence",
        "",
        "| Frames | Samples | Decisive | Decisive fraction | Mean range | "
        "Mean regret |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in horizons:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"| {row['lookahead_frames']} | {row['sample_count']} | "
            f"{row['decisive_count']} | {float(row['decisive_fraction']):.3f} | "
            f"{float(row['mean_action_range']):.2f} | "
            f"{float(row['mean_policy_regret']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed timeline and policy dynamics",
            "",
            "| Seed | Enemy | AOE | Pattern | Near collision | First decisive | "
            "First change | Max run | Terminal |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    primary_steps = gate["primary_lookahead_steps"]
    paths = diagnostic["paths"]
    if isinstance(paths, Sequence):
        for row in paths:
            if not isinstance(row, Mapping):
                continue
            first_decisive = row.get("first_decisive_frames")
            first_decisive_frame = (
                first_decisive.get(str(primary_steps))
                if isinstance(first_decisive, Mapping)
                else None
            )
            lines.append(
                f"| {row['seed']} | {row['first_enemy_frame'] or '-'} | "
                f"{row['first_aoe_frame'] or '-'} | "
                f"{row['first_pattern_frame'] or '-'} | "
                f"{row['first_near_collision_frame'] or '-'} | "
                f"{first_decisive_frame or '-'} | "
                f"{row['first_action_change_frame'] or '-'} | "
                f"{row['max_action_run']} | {row['terminal_frame'] or '-'} |"
            )
    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            f"Primary horizon: `{gate['primary_lookahead_frames']}` game frames.  ",
            f"Action-range threshold: `{gate['margin_threshold']}` frames.  ",
            f"Decisive seeds: `{gate['decisive_seed_count']}`.  ",
            f"Selection eligible: **{gate['selection_eligible']}**.",
            "",
            "Visual checkpoints, when requested, live under `visuals/` and are "
            "indexed-palette PNGs from the same native replay.",
            "",
        ]
    )
    return "\n".join(lines)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-diagnose-relevance")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--action", choices=ACTION_CHOICES)
    parser.add_argument(
        "--partition", choices=("training", "holdout"), default="training"
    )
    parser.add_argument(
        "--lookahead-steps",
        type=_positive_int,
        nargs="+",
        default=list(DEFAULT_LOOKAHEAD_STEPS),
    )
    parser.add_argument(
        "--gate-lookahead-steps",
        type=_positive_int,
        default=DEFAULT_LOOKAHEAD_STEPS[0],
    )
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--sample-every", type=_positive_int, default=4)
    parser.add_argument("--max-samples-per-seed", type=_positive_int, default=64)
    parser.add_argument("--max-episode-steps", type=_positive_int, default=2_000)
    parser.add_argument("--margin-threshold", type=float, default=4.0)
    parser.add_argument("--min-decisive-fraction", type=float, default=0.10)
    parser.add_argument("--min-decisive-seeds", type=_positive_int, default=1)
    parser.add_argument("--near-collision-horizon-frames", type=float, default=32.0)
    parser.add_argument("--native-lanes", type=_positive_int, default=32)
    parser.add_argument(
        "--execution", choices=("serial", "parallel"), default="parallel"
    )
    parser.add_argument("--visual-seed", type=int, action="append", default=[])
    arguments = parser.parse_args(argv)
    config = RelevanceConfig(
        manifest_path=arguments.manifest,
        output_directory=arguments.output_dir,
        checkpoint=arguments.checkpoint,
        action=arguments.action,
        partition=arguments.partition,
        lookahead_steps=tuple(arguments.lookahead_steps),
        gate_lookahead_steps=arguments.gate_lookahead_steps,
        step_frames=arguments.step_frames,
        sample_every=arguments.sample_every,
        max_samples_per_seed=arguments.max_samples_per_seed,
        max_episode_steps=arguments.max_episode_steps,
        margin_threshold=arguments.margin_threshold,
        min_decisive_fraction=arguments.min_decisive_fraction,
        min_decisive_seeds=arguments.min_decisive_seeds,
        near_collision_horizon_frames=arguments.near_collision_horizon_frames,
        native_lanes=arguments.native_lanes,
        native_execution=arguments.execution,
        visual_seeds=tuple(arguments.visual_seed),
    )
    try:
        diagnostic = build_decision_relevance_audit(config)
    except (ControlRuntimeError, OSError, RuntimeError, ValueError) as error:
        print(f"dodge-ng-diagnose-relevance: {error}", file=sys.stderr)
        return 1
    gate = diagnostic["gate"]
    print(
        json.dumps(
            {
                "output_directory": str(config.output_directory),
                "manifest_sha256": diagnostic["manifest_sha256"],
                "sample_count": diagnostic["sample_count"],
                "gate_passed": gate["passed"] if isinstance(gate, Mapping) else False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
