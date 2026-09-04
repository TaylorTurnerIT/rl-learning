from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from dodge.control import PROJECT_ROOT, ControlInputError, ControlRuntimeError
from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import NativeBatchEnvironment
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, SeedManifest, load_manifest

TEACHER_SCHEMA_VERSION = 1
TEACHER_DATA_VERSION = 1
BOARD_SHAPE = (19, 16, 16)
ACTION_COUNT = len(ACTION_CHOICES)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "history" / "dodge" / "ng" / "teacher-p2"
DEFAULT_COLLECTOR_SEED = 2_026_0910
DEFAULT_STATES_PER_SEED = 64
DEFAULT_LOOKAHEAD_STEPS = 8
DEFAULT_NATIVE_LANES = 32
DEFAULT_MAX_COLLECTOR_STEPS = 8_000
Execution = Literal["serial", "parallel"]
CollectionPolicy = Literal["uniform", "planner"]


@dataclass(frozen=True, slots=True)
class TeacherConfig:
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
    states_per_seed: int = DEFAULT_STATES_PER_SEED
    lookahead_steps: int = DEFAULT_LOOKAHEAD_STEPS
    step_frames: int = 4
    native_lanes: int = DEFAULT_NATIVE_LANES
    native_execution: Execution = "parallel"
    collector_seed: int = DEFAULT_COLLECTOR_SEED
    max_collector_steps: int = DEFAULT_MAX_COLLECTOR_STEPS
    collection_policy: CollectionPolicy = "planner"
    planner_epsilon: float = 0.15
    difficulty: int = 2
    patterns_enabled: bool = True
    powerups_enabled: bool = True

    def validate(self, manifest: SeedManifest) -> None:
        if self.states_per_seed < 1:
            raise ValueError("states per seed must be positive")
        if self.lookahead_steps < 1:
            raise ValueError("lookahead steps must be positive")
        if not 3 <= self.step_frames <= 5:
            raise ValueError("step frames must be between 3 and 5")
        if self.native_lanes < 1:
            raise ValueError("native lanes must be positive")
        if self.native_lanes > len(manifest.training_seeds):
            raise ValueError("native lanes must not exceed training seed count")
        if self.native_execution not in {"serial", "parallel"}:
            raise ValueError("native execution must be serial or parallel")
        if self.max_collector_steps < 1:
            raise ValueError("maximum collector steps must be positive")
        if self.collection_policy not in {"uniform", "planner"}:
            raise ValueError("collection policy must be uniform or planner")
        if not 0 <= self.planner_epsilon <= 1:
            raise ValueError("planner epsilon must be between 0 and 1")
        if not 1 <= self.difficulty <= 3:
            raise ValueError("difficulty must be between 1 and 3")

    def to_json(self) -> dict[str, object]:
        value = asdict(self)
        value["manifest_path"] = str(self.manifest_path)
        value["output_directory"] = str(self.output_directory)
        return value


@dataclass(frozen=True, slots=True)
class TeacherDataset:
    boards: np.ndarray
    actions: np.ndarray
    scores: np.ndarray
    margins: np.ndarray
    seeds: np.ndarray
    frames: np.ndarray
    state_hashes: np.ndarray
    pixel_hashes: np.ndarray
    metadata: dict[str, object]

    @property
    def count(self) -> int:
        return int(self.actions.shape[0])

    @property
    def decisive_mask(self) -> np.ndarray:
        return self.margins > 0

    @property
    def decisive_count(self) -> int:
        return int(np.count_nonzero(self.decisive_mask))

    def training_subset(self, seeds: Sequence[int]) -> TeacherDataset:
        allowed = np.asarray(tuple(seeds), dtype=np.uint32)
        mask = np.isin(self.seeds, allowed)
        return _subset(self, mask)


class CounterfactualCache:
    """Memoize exact state/config scores within one data-generation run."""

    def __init__(self) -> None:
        self._values: dict[tuple[bytes, int], np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def score(
        self,
        environment: NativeBatchEnvironment,
        snapshots: Sequence[bytes],
        lookahead_steps: int,
    ) -> np.ndarray:
        values: list[np.ndarray | None] = [None] * len(snapshots)
        missing: list[bytes] = []
        missing_positions: dict[tuple[bytes, int], list[int]] = {}
        for position, snapshot in enumerate(snapshots):
            key = (snapshot, lookahead_steps)
            cached = self._values.get(key)
            if cached is not None:
                values[position] = cached
                self.hits += 1
                continue
            if key not in missing_positions:
                missing.append(snapshot)
                missing_positions[key] = []
            missing_positions[key].append(position)

        if missing:
            self.misses += len(missing)
            computed = _score_snapshots_uncached(environment, missing, lookahead_steps)
            for snapshot, score in zip(missing, computed, strict=True):
                self._values[(snapshot, lookahead_steps)] = np.array(
                    score, dtype=np.float32, copy=True
                )
            for key, positions in missing_positions.items():
                score = self._values[key]
                for position in positions:
                    values[position] = score
        if any(value is None for value in values):
            raise ControlRuntimeError("counterfactual cache lost a score")
        return np.asarray(values, dtype=np.float32)

    def to_json(self) -> dict[str, int]:
        return {
            "entries": len(self._values),
            "hits": self.hits,
            "misses": self.misses,
        }


def collect_teacher_dataset(config: TeacherConfig) -> TeacherDataset:
    manifest = load_manifest(config.manifest_path)
    config.validate(manifest)
    training_seeds = tuple(manifest.training_seeds)
    counts = {seed: 0 for seed in training_seeds}
    next_seed_index = config.native_lanes
    rng = np.random.default_rng(config.collector_seed)
    boards: list[np.ndarray] = []
    snapshots: list[bytes] = []
    seeds: list[int] = []
    frames: list[int] = []
    state_hashes: list[int] = []
    pixel_hashes: list[int] = []
    collected_scores: list[np.ndarray] = []
    score_cache = CounterfactualCache()
    current_seeds = list(training_seeds[: config.native_lanes])

    with NativeBatchEnvironment(
        step_frames=config.step_frames,
        execution=config.native_execution,
        full_state=True,
        pixels=False,
        board=True,
        difficulty=config.difficulty,
        patterns_enabled=config.patterns_enabled,
        powerups_enabled=config.powerups_enabled,
    ) as environment:
        result = environment.reset_batch(np.asarray(current_seeds, dtype=np.uint32))
        for _collector_step in range(config.max_collector_steps):
            current_boards = result.board
            if current_boards is None:
                raise ControlRuntimeError("teacher collection requires board buffers")
            online_scores: np.ndarray | None = None
            if config.collection_policy == "planner":
                current_snapshots = [
                    snapshot
                    for snapshot in result.snapshot_bytes
                    if snapshot is not None
                ]
                if len(current_snapshots) != config.native_lanes:
                    raise ControlRuntimeError(
                        "planner collection requires a snapshot for every lane"
                    )
                online_scores = _score_snapshots(
                    environment,
                    current_snapshots,
                    config.lookahead_steps,
                    cache=score_cache,
                )
            for lane, seed in enumerate(current_seeds):
                snapshot = result.snapshot_bytes[lane]
                if (
                    result.modes[lane] == 2
                    and snapshot is not None
                    and counts[seed] < config.states_per_seed
                ):
                    boards.append(np.array(current_boards[lane], copy=True))
                    snapshots.append(snapshot)
                    seeds.append(seed)
                    frames.append(int(result.frames[lane]))
                    state_hashes.append(int(result.state_hashes[lane]))
                    pixel_hashes.append(int(result.pixel_hashes[lane]))
                    if online_scores is not None:
                        collected_scores.append(
                            np.array(online_scores[lane], dtype=np.float32, copy=True)
                        )
                    counts[seed] += 1

            if all(count == config.states_per_seed for count in counts.values()):
                break

            if online_scores is None:
                actions = rng.integers(0, ACTION_COUNT, size=config.native_lanes)
            else:
                actions = np.argmax(online_scores, axis=1).astype(np.int64)
                for lane in range(config.native_lanes):
                    if rng.random() < config.planner_epsilon:
                        actions[lane] = rng.integers(0, ACTION_COUNT)
            done_result = environment.step_batch(actions.astype(np.uint8, copy=False))
            next_result = done_result
            reset_lanes: list[int] = []
            reset_seeds: list[int] = []
            for lane, seed in enumerate(current_seeds):
                needs_reset = bool(done_result.done[lane]) or (
                    counts[seed] >= config.states_per_seed
                )
                if not needs_reset:
                    continue
                reset_lanes.append(lane)
                if counts[seed] < config.states_per_seed:
                    reset_seeds.append(seed)
                elif next_seed_index < len(training_seeds):
                    replacement = training_seeds[next_seed_index]
                    next_seed_index += 1
                    current_seeds[lane] = replacement
                    reset_seeds.append(replacement)
                else:
                    # This lane has finished its assigned seed, but the
                    # collection loop may still need other lanes. Replaying
                    # its final seed keeps the batch callable until completion.
                    reset_seeds.append(seed)
            if reset_lanes:
                reset_result = environment.reset_lanes(
                    np.asarray(reset_lanes, dtype=np.uint32),
                    np.asarray(reset_seeds, dtype=np.uint32),
                )
                # The partial reset result replaces those lanes in the next
                # observation; all unselected lanes remain in done_result.
                next_result = _merge_results(done_result, reset_result)
            result = next_result
        else:
            incomplete = [
                seed for seed, count in counts.items() if count < config.states_per_seed
            ]
            raise ControlRuntimeError(
                "teacher collection reached its step limit; incomplete seeds: "
                f"{incomplete[:8]}"
            )

        scores = _score_snapshots(
            environment, snapshots, config.lookahead_steps, cache=score_cache
        )

    boards_array = np.asarray(boards, dtype=np.float32)
    if config.collection_policy == "planner":
        scores_array = np.asarray(collected_scores, dtype=np.float32)
    else:
        scores_array = np.asarray(scores, dtype=np.float32)
    actions_array = np.argmax(scores_array, axis=1).astype(np.int64)
    margins_array = _score_margins(scores_array)
    dataset = TeacherDataset(
        boards=boards_array,
        actions=actions_array,
        scores=scores_array,
        margins=margins_array,
        seeds=np.asarray(seeds, dtype=np.uint32),
        frames=np.asarray(frames, dtype=np.uint32),
        state_hashes=np.asarray(state_hashes, dtype=np.uint64),
        pixel_hashes=np.asarray(pixel_hashes, dtype=np.uint64),
        metadata={
            "schema_version": TEACHER_SCHEMA_VERSION,
            "data_version": TEACHER_DATA_VERSION,
            "kind": "dodge_ng_teacher_dataset",
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": manifest.sha256,
            "seed_scope": "training_only",
            "training_seeds": list(manifest.training_seeds),
            "holdout_examples": 0,
            "legacy_inputs": "none",
            "board_shape": list(BOARD_SHAPE),
            "actions": list(ACTION_CHOICES),
            "examples": len(actions_array),
            "decisive_examples": int(np.count_nonzero(margins_array > 0)),
            "action_counts": np.bincount(
                actions_array, minlength=ACTION_COUNT
            ).tolist(),
            "lookahead_steps": config.lookahead_steps,
            "step_frames": config.step_frames,
            "collection_policy": config.collection_policy,
            "planner_epsilon": config.planner_epsilon,
            "score_cache": score_cache.to_json(),
            "native_config": {
                "difficulty": config.difficulty,
                "patterns_enabled": config.patterns_enabled,
                "powerups_enabled": config.powerups_enabled,
            },
            "collector_config": config.to_json(),
        },
    )
    save_teacher_dataset(dataset, config.output_directory)
    return dataset


def save_teacher_dataset(dataset: TeacherDataset, output_directory: Path) -> None:
    _validate_arrays(dataset)
    output_directory.mkdir(parents=True, exist_ok=True)
    data_path = output_directory / "teacher-data.npz"
    metadata_path = output_directory / "metadata.json"
    temporary_data = output_directory / ".teacher-data.tmp.npz"
    try:
        np.savez_compressed(
            temporary_data,
            boards=dataset.boards,
            actions=dataset.actions,
            scores=dataset.scores,
            margins=dataset.margins,
            seeds=dataset.seeds,
            frames=dataset.frames,
            state_hashes=dataset.state_hashes,
            pixel_hashes=dataset.pixel_hashes,
        )
        temporary_data.replace(data_path)
    finally:
        temporary_data.unlink(missing_ok=True)
    metadata = dict(dataset.metadata)
    metadata["data_path"] = data_path.name
    metadata["examples"] = dataset.count
    metadata["decisive_examples"] = dataset.decisive_count
    metadata["action_counts"] = np.bincount(
        dataset.actions, minlength=ACTION_COUNT
    ).tolist()
    metadata["data_sha256"] = _sha256(data_path)
    _write_json(metadata_path, metadata)


def load_teacher_dataset(
    data_path: Path, manifest: SeedManifest, *, metadata_path: Path | None = None
) -> TeacherDataset:
    metadata_path = metadata_path or data_path.with_name("metadata.json")
    try:
        metadata_value = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(data_path, allow_pickle=False) as values:
            dataset = TeacherDataset(
                boards=np.asarray(values["boards"]),
                actions=np.asarray(values["actions"]),
                scores=np.asarray(values["scores"]),
                margins=np.asarray(values["margins"]),
                seeds=np.asarray(values["seeds"]),
                frames=np.asarray(values["frames"]),
                state_hashes=np.asarray(values["state_hashes"]),
                pixel_hashes=np.asarray(values["pixel_hashes"]),
                metadata=metadata_value,
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid teacher dataset: {error}") from error
    if not isinstance(metadata_value, dict):
        raise ValueError("teacher metadata must be a JSON object")
    if metadata_value.get("examples") != dataset.count:
        raise ValueError("teacher metadata example count does not match arrays")
    if metadata_value.get("decisive_examples") != dataset.decisive_count:
        raise ValueError("teacher metadata decisive count does not match arrays")
    expected_action_counts = np.bincount(
        dataset.actions, minlength=ACTION_COUNT
    ).tolist()
    if metadata_value.get("action_counts") != expected_action_counts:
        raise ValueError("teacher metadata action counts do not match arrays")
    expected_hash = metadata_value.get("data_sha256")
    if expected_hash != _sha256(data_path):
        raise ValueError("teacher dataset hash is missing or invalid")
    _validate_metadata(metadata_value, manifest)
    _validate_arrays(dataset, manifest=manifest)
    return dataset


def _score_snapshots(
    environment: NativeBatchEnvironment,
    snapshots: Sequence[bytes],
    lookahead_steps: int,
    *,
    cache: CounterfactualCache | None = None,
) -> np.ndarray:
    if cache is not None:
        return cache.score(environment, snapshots, lookahead_steps)
    return _score_snapshots_uncached(environment, snapshots, lookahead_steps)


def _score_snapshots_uncached(
    environment: NativeBatchEnvironment,
    snapshots: Sequence[bytes],
    lookahead_steps: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, len(snapshots), 256):
        scores.append(
            environment.score_actions(snapshots[start : start + 256], lookahead_steps)
        )
    if not scores:
        raise ControlRuntimeError("teacher collection produced no snapshots")
    return np.concatenate(scores, axis=0)


def _score_margins(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(scores, axis=1)
    return (ordered[:, -1] - ordered[:, -2]).astype(np.float32)


def _subset(dataset: TeacherDataset, mask: np.ndarray) -> TeacherDataset:
    return TeacherDataset(
        boards=dataset.boards[mask],
        actions=dataset.actions[mask],
        scores=dataset.scores[mask],
        margins=dataset.margins[mask],
        seeds=dataset.seeds[mask],
        frames=dataset.frames[mask],
        state_hashes=dataset.state_hashes[mask],
        pixel_hashes=dataset.pixel_hashes[mask],
        metadata=dict(dataset.metadata),
    )


def _validate_metadata(metadata: dict[str, object], manifest: SeedManifest) -> None:
    if metadata.get("schema_version") != TEACHER_SCHEMA_VERSION:
        raise ValueError("unsupported teacher dataset schema")
    if metadata.get("data_version") != TEACHER_DATA_VERSION:
        raise ValueError("unsupported teacher data version")
    if metadata.get("kind") != "dodge_ng_teacher_dataset":
        raise ValueError("teacher metadata kind is invalid")
    if metadata.get("manifest_sha256") != manifest.sha256:
        raise ValueError("teacher dataset manifest does not match")
    if metadata.get("seed_scope") != "training_only":
        raise ValueError("teacher dataset seed scope is not training-only")
    if metadata.get("legacy_inputs") != "none":
        raise ValueError("teacher dataset declares legacy inputs")
    if metadata.get("holdout_examples") != 0:
        raise ValueError("teacher dataset contains holdout examples")
    if tuple(metadata.get("board_shape", ())) != BOARD_SHAPE:
        raise ValueError("teacher board shape is invalid")
    if tuple(metadata.get("actions", ())) != ACTION_CHOICES:
        raise ValueError("teacher action ordering is invalid")


def _validate_arrays(
    dataset: TeacherDataset, *, manifest: SeedManifest | None = None
) -> None:
    count = dataset.actions.shape[0]
    if dataset.boards.shape != (count, *BOARD_SHAPE):
        raise ValueError("teacher boards have an invalid shape")
    if dataset.scores.shape != (count, ACTION_COUNT):
        raise ValueError("teacher scores have an invalid shape")
    for name, value in (
        ("actions", dataset.actions),
        ("margins", dataset.margins),
        ("seeds", dataset.seeds),
        ("frames", dataset.frames),
        ("state_hashes", dataset.state_hashes),
        ("pixel_hashes", dataset.pixel_hashes),
    ):
        if value.shape != (count,):
            raise ValueError(f"teacher {name} has an invalid shape")
    if not all(
        np.isfinite(value).all()
        for value in (dataset.boards, dataset.scores, dataset.margins)
    ):
        raise ValueError("teacher data contains non-finite values")
    if np.any(dataset.actions < 0) or np.any(dataset.actions >= ACTION_COUNT):
        raise ValueError("teacher actions are outside the native action space")
    expected_actions = np.argmax(dataset.scores, axis=1)
    if not np.array_equal(dataset.actions, expected_actions):
        raise ValueError("teacher actions do not match score maxima")
    expected_margins = _score_margins(dataset.scores)
    if not np.allclose(dataset.margins, expected_margins, atol=0, rtol=0):
        raise ValueError("teacher margins do not match action scores")
    if manifest is not None:
        training = np.asarray(manifest.training_seeds, dtype=np.uint32)
        holdout = np.asarray(manifest.holdout_seeds, dtype=np.uint32)
        if np.any(np.isin(dataset.seeds, holdout)):
            raise ValueError("teacher data contains a holdout seed")
        if np.any(~np.isin(dataset.seeds, training)):
            raise ValueError("teacher data contains a non-training seed")


def _merge_results(full: object, partial: object) -> object:
    """Merge a partial reset into a batch result without borrowing buffers."""
    # Kept local to the collector so the public batch result remains a simple
    # ownership boundary. NativeBatchResult is intentionally reconstructed by
    # field, including every optional buffer and snapshot.
    from dodge.native.batch import NativeBatchResult

    if not isinstance(full, NativeBatchResult) or not isinstance(
        partial, NativeBatchResult
    ):
        raise ControlRuntimeError("native teacher reset returned an invalid result")
    lane_ids = full.lane_ids.copy()
    frames = full.frames.copy()
    frames_advanced = full.frames_advanced.copy()
    rewards = full.rewards.copy()
    done = full.done.copy()
    seeds = full.seeds.copy()
    state_hashes = full.state_hashes.copy()
    pixel_hashes = full.pixel_hashes.copy()
    modes = full.modes.copy()
    event_flags = full.event_flags.copy()
    pixels = None if full.pixels is None else full.pixels.copy()
    board = None if full.board is None else full.board.copy()
    ml_observation = None if full.ml_observation is None else full.ml_observation.copy()
    player_positions = (
        None if full.player_positions is None else full.player_positions.copy()
    )
    snapshots = list(full.snapshot_bytes)
    for position, lane_value in enumerate(partial.lane_ids.tolist()):
        lane = int(lane_value)
        frames[lane] = partial.frames[position]
        frames_advanced[lane] = partial.frames_advanced[position]
        rewards[lane] = partial.rewards[position]
        done[lane] = partial.done[position]
        seeds[lane] = partial.seeds[position]
        state_hashes[lane] = partial.state_hashes[position]
        pixel_hashes[lane] = partial.pixel_hashes[position]
        modes[lane] = partial.modes[position]
        event_flags[lane] = partial.event_flags[position]
        snapshots[lane] = partial.snapshot_bytes[position]
        if pixels is not None and partial.pixels is not None:
            pixels[lane] = partial.pixels[position]
        if board is not None and partial.board is not None:
            board[lane] = partial.board[position]
        if ml_observation is not None and partial.ml_observation is not None:
            ml_observation[lane] = partial.ml_observation[position]
        if player_positions is not None and partial.player_positions is not None:
            player_positions[lane] = partial.player_positions[position]
    return NativeBatchResult(
        lane_ids=lane_ids,
        frames=frames,
        frames_advanced=frames_advanced,
        rewards=rewards,
        done=done,
        seeds=seeds,
        state_hashes=state_hashes,
        pixel_hashes=pixel_hashes,
        modes=modes,
        event_flags=event_flags,
        pixels=pixels,
        board=board,
        ml_observation=ml_observation,
        player_positions=player_positions,
        snapshot_bytes=tuple(snapshots),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-teacher")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--states-per-seed", type=_positive_int, default=64)
    parser.add_argument("--lookahead-steps", type=_positive_int, default=8)
    parser.add_argument("--step-frames", type=int, default=4)
    parser.add_argument("--native-lanes", type=_positive_int, default=32)
    parser.add_argument(
        "--native-execution", choices=("serial", "parallel"), default="parallel"
    )
    parser.add_argument("--collector-seed", type=int, default=DEFAULT_COLLECTOR_SEED)
    parser.add_argument(
        "--max-collector-steps", type=_positive_int, default=DEFAULT_MAX_COLLECTOR_STEPS
    )
    parser.add_argument(
        "--collection-policy", choices=("uniform", "planner"), default="planner"
    )
    parser.add_argument("--planner-epsilon", type=float, default=0.15)
    parser.add_argument("--difficulty", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--no-patterns", action="store_true")
    parser.add_argument("--no-powerups", action="store_true")
    arguments = parser.parse_args(argv)
    config = TeacherConfig(
        manifest_path=arguments.manifest,
        output_directory=arguments.output_dir,
        states_per_seed=arguments.states_per_seed,
        lookahead_steps=arguments.lookahead_steps,
        step_frames=arguments.step_frames,
        native_lanes=arguments.native_lanes,
        native_execution=arguments.native_execution,
        collector_seed=arguments.collector_seed,
        max_collector_steps=arguments.max_collector_steps,
        collection_policy=arguments.collection_policy,
        planner_epsilon=arguments.planner_epsilon,
        difficulty=arguments.difficulty,
        patterns_enabled=not arguments.no_patterns,
        powerups_enabled=not arguments.no_powerups,
    )
    try:
        dataset = collect_teacher_dataset(config)
    except (
        ControlInputError,
        ControlRuntimeError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"dodge-ng-teacher: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output_directory": str(config.output_directory),
                "examples": dataset.count,
                "decisive_examples": dataset.decisive_count,
                "manifest_sha256": dataset.metadata["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
