from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sqlite3
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from dodge.control import ControlInputError, ControlRuntimeError, MovementCommand
from dodge.headless import run_headless, run_headless_trace
from dodge.neat.state import RawState, project_state

DATASET_VERSION = 1
DEFAULT_DATABASE = Path("history/dodge/dataset.sqlite3")
STEP_FRAMES = 4
TARGET_SURVIVAL_FRAMES = 1_800
TRACES_PER_SEED = 5
PILOT_SEED_COUNT = 5
PILOT_GENERATIONS_PER_SEED = 100
DEFAULT_POPULATION = 50
ELITE_COUNT = 5
MUTATION_RATE = 0.05
EVALUATION_SEEDS = tuple(range(30_001, 30_011))
TRAINING_SEED_MAX = 30_000
BOOTSTRAP = (("x", 3), ("neutral", 18), ("up", 6), ("down", 6))
ACTION_CHOICES = (
    "neutral",
    "left",
    "right",
    "up",
    "down",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
)


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    database: Path = DEFAULT_DATABASE
    train_seeds: tuple[int, ...] = tuple(range(PILOT_SEED_COUNT))
    generations_per_seed: int = PILOT_GENERATIONS_PER_SEED
    population: int = DEFAULT_POPULATION
    workers: int = 8
    evolution_seed: int = 42

    @property
    def genome_length(self) -> int:
        return (TARGET_SURVIVAL_FRAMES + STEP_FRAMES - 1) // STEP_FRAMES

    def to_json(self) -> dict[str, object]:
        return {
            "dataset_version": DATASET_VERSION,
            "step_frames": STEP_FRAMES,
            "target_survival_frames": TARGET_SURVIVAL_FRAMES,
            "traces_per_seed": TRACES_PER_SEED,
            "bootstrap": list(BOOTSTRAP),
            "train_seeds": list(self.train_seeds),
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "generations_per_seed": self.generations_per_seed,
            "population": self.population,
            "workers": self.workers,
            "evolution_seed": self.evolution_seed,
        }


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    completed_seeds: int
    accepted_episodes: int
    unsolved_seeds: tuple[int, ...]


def collect(config: CollectorConfig, *, resume: bool = False) -> CollectionSummary:
    _validate_config(config)
    connection = _open_database(config.database)
    try:
        _initialize_database(connection, config, resume=resume)
        checkpoint = _load_checkpoint(connection)
        random_source, seed_index, generation, population = _restore_or_initialize(
            checkpoint, config
        )
        unsolved: list[int] = []
        while seed_index < len(config.train_seeds):
            seed = config.train_seeds[seed_index]
            accepted_hashes = _accepted_hashes(connection, seed)
            if len(accepted_hashes) >= TRACES_PER_SEED:
                seed_index += 1
                generation = 0
                population = _new_population(random_source, config)
                _checkpoint(
                    connection, random_source, seed_index, generation, population
                )
                continue
            for next_generation in range(
                generation + 1, config.generations_per_seed + 1
            ):
                generation = next_generation
                results = _evaluate_population(seed, population, config)
                ranked = sorted(zip(results, population, strict=True), reverse=True)
                for fitness, genome in ranked:
                    if fitness < TARGET_SURVIVAL_FRAMES:
                        break
                    if len(accepted_hashes) >= TRACES_PER_SEED:
                        break
                    digest = _genome_hash(genome)
                    if digest in accepted_hashes:
                        continue
                    _accept_episode(connection, seed, genome, digest, config)
                    accepted_hashes.add(digest)
                if len(accepted_hashes) >= TRACES_PER_SEED:
                    break
                population = _breed_population(ranked, random_source, config)
                _checkpoint(
                    connection, random_source, seed_index, generation, population
                )
            else:
                unsolved.append(seed)
            seed_index += 1
            generation = 0
            population = _new_population(random_source, config)
            _checkpoint(connection, random_source, seed_index, generation, population)
        return CollectionSummary(
            completed_seeds=len(config.train_seeds),
            accepted_episodes=_episode_count(connection),
            unsolved_seeds=tuple(unsolved),
        )
    finally:
        connection.close()


def _validate_config(config: CollectorConfig) -> None:
    if not config.train_seeds:
        raise ValueError("at least one training seed is required")
    if any(not 0 <= seed <= TRAINING_SEED_MAX for seed in config.train_seeds):
        raise ValueError("training seeds must be from 0 to 30000")
    if len(set(config.train_seeds)) != len(config.train_seeds):
        raise ValueError("training seeds must be unique")
    if len(EVALUATION_SEEDS) != 10 or any(
        seed <= TRAINING_SEED_MAX for seed in EVALUATION_SEEDS
    ):
        raise AssertionError("evaluation seed policy is invalid")
    if set(config.train_seeds) & set(EVALUATION_SEEDS):
        raise ValueError("training and evaluation seeds overlap")
    if config.generations_per_seed < 1 or config.population < ELITE_COUNT:
        raise ValueError("generation and population sizes are too small")
    if config.workers < 1:
        raise ValueError("workers must be positive")


def _open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _initialize_database(
    connection: sqlite3.Connection, config: CollectorConfig, *, resume: bool
) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS seeds (seed INTEGER PRIMARY KEY, role TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS episodes (
          id INTEGER PRIMARY KEY,
          seed INTEGER NOT NULL REFERENCES seeds(seed),
          action_hash TEXT NOT NULL,
          result_json TEXT NOT NULL,
          config_json TEXT NOT NULL,
          UNIQUE(seed, action_hash)
        );
        CREATE TABLE IF NOT EXISTS steps (
          episode_id INTEGER NOT NULL REFERENCES episodes(id),
          action_index INTEGER NOT NULL,
          frame INTEGER NOT NULL,
          action TEXT NOT NULL,
          bootstrap INTEGER NOT NULL,
          observation_f32 BLOB NOT NULL,
          raw_state_json TEXT NOT NULL,
          PRIMARY KEY (episode_id, action_index)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS checkpoints (
          id INTEGER PRIMARY KEY CHECK(id=1),
          state BLOB NOT NULL
        );
        """
    )
    stored = connection.execute(
        "SELECT value FROM metadata WHERE key='config'"
    ).fetchone()
    encoded = json.dumps(config.to_json(), sort_keys=True, separators=(",", ":"))
    if stored is None:
        connection.execute("INSERT INTO metadata VALUES ('config', ?)", (encoded,))
        connection.executemany(
            "INSERT INTO seeds(seed, role) VALUES (?, ?)",
            [(seed, "training") for seed in config.train_seeds]
            + [(seed, "evaluation") for seed in EVALUATION_SEEDS],
        )
        connection.commit()
    elif not resume:
        raise ValueError("dataset exists; use --resume")
    elif stored[0] != encoded:
        raise ValueError("collector configuration differs from existing dataset")


def _restore_or_initialize(
    checkpoint: bytes | None, config: CollectorConfig
) -> tuple[random.Random, int, int, list[tuple[str, ...]]]:
    random_source = random.Random(config.evolution_seed)
    if checkpoint is None:
        return random_source, 0, 0, _new_population(random_source, config)
    state = pickle.loads(checkpoint)
    random_source.setstate(state["rng"])
    return random_source, state["seed_index"], state["generation"], state["population"]


def _new_population(
    random_source: random.Random, config: CollectorConfig
) -> list[tuple[str, ...]]:
    return [
        tuple(random_source.choice(ACTION_CHOICES) for _ in range(config.genome_length))
        for _ in range(config.population)
    ]


def _evaluate_population(
    seed: int, population: list[tuple[str, ...]], config: CollectorConfig
) -> list[int]:
    with ThreadPoolExecutor(
        max_workers=min(config.workers, len(population))
    ) as executor:
        return list(executor.map(lambda genome: _fitness(seed, genome), population))


def _fitness(seed: int, genome: tuple[str, ...]) -> int:
    return int(
        run_headless(_commands(genome), seed=seed, wait_for_game_start=True)[
            "survival_frames"
        ]
    )


def _commands(genome: tuple[str, ...]) -> list[MovementCommand]:
    return [
        *[_command(move, frames) for move, frames in BOOTSTRAP],
        *[_command(action, STEP_FRAMES) for action in genome],
    ]


def _command(action: str, frames: int) -> MovementCommand:
    return MovementCommand(action, (frames * 1_000) // 60)


def _breed_population(
    ranked: list[tuple[int, tuple[str, ...]]],
    random_source: random.Random,
    config: CollectorConfig,
) -> list[tuple[str, ...]]:
    parents = [genome for _, genome in ranked[:ELITE_COUNT]]
    population = parents.copy()
    while len(population) < config.population:
        parent = parents[len(population) % len(parents)]
        population.append(
            tuple(
                random_source.choice(ACTION_CHOICES)
                if random_source.random() < MUTATION_RATE
                else action
                for action in parent
            )
        )
    return population


def _accept_episode(
    connection: sqlite3.Connection,
    seed: int,
    genome: tuple[str, ...],
    digest: str,
    config: CollectorConfig,
) -> None:
    commands = _commands(genome)
    expected = run_headless(commands, seed=seed, wait_for_game_start=True)
    trace = run_headless_trace(commands, seed=seed)
    if (
        trace.result != expected
        or trace.result["survival_frames"] < TARGET_SURVIVAL_FRAMES
    ):
        raise ControlRuntimeError("accepted trace failed deterministic replay")
    actions = ("neutral", "up", "down", *genome)
    states = trace.states[:-1]
    if not states or len(states) > len(actions):
        raise ControlRuntimeError("headless state trace does not match action trace")
    with connection:
        cursor = connection.execute(
            "INSERT INTO episodes(seed, action_hash, result_json, config_json) "
            "VALUES (?, ?, ?, ?)",
            (
                seed,
                digest,
                json.dumps(trace.result, sort_keys=True, separators=(",", ":")),
                json.dumps(config.to_json(), sort_keys=True, separators=(",", ":")),
            ),
        )
        episode_id = cursor.lastrowid
        connection.executemany(
            "INSERT INTO steps VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    episode_id,
                    index,
                    state.frame,
                    action,
                    int(index < 3),
                    _packed_observation(state),
                    json.dumps(state.to_json(), sort_keys=True, separators=(",", ":")),
                )
                for index, (state, action) in enumerate(
                    zip(states, actions[: len(states)], strict=True)
                )
            ],
        )


def _packed_observation(state: RawState) -> bytes:
    projected = project_state(state, include_time_to_intersection=True)
    return struct.pack(f"<{len(projected.values)}f", *projected.values)


def _genome_hash(genome: tuple[str, ...]) -> str:
    encoded = json.dumps(genome, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _accepted_hashes(connection: sqlite3.Connection, seed: int) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT action_hash FROM episodes WHERE seed=?", (seed,)
        )
    }


def _episode_count(connection: sqlite3.Connection) -> int:
    return int(connection.execute("SELECT count(*) FROM episodes").fetchone()[0])


def _checkpoint(
    connection: sqlite3.Connection,
    random_source: random.Random,
    seed_index: int,
    generation: int,
    population: list[tuple[str, ...]],
) -> None:
    state = pickle.dumps(
        {
            "rng": random_source.getstate(),
            "seed_index": seed_index,
            "generation": generation,
            "population": population,
        }
    )
    with connection:
        connection.execute("INSERT OR REPLACE INTO checkpoints VALUES (1, ?)", (state,))


def _load_checkpoint(connection: sqlite3.Connection) -> bytes | None:
    row = connection.execute("SELECT state FROM checkpoints WHERE id=1").fetchone()
    return None if row is None else row[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-dataset-collect")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=PILOT_SEED_COUNT)
    parser.add_argument(
        "--generations-per-seed", type=int, default=PILOT_GENERATIONS_PER_SEED
    )
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--evolution-seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        config = CollectorConfig(
            database=arguments.database,
            train_seeds=tuple(
                range(arguments.seed_start, arguments.seed_start + arguments.seed_count)
            ),
            generations_per_seed=arguments.generations_per_seed,
            population=arguments.population,
            workers=arguments.workers,
            evolution_seed=arguments.evolution_seed,
        )
        print(
            json.dumps(
                asdict(collect(config, resume=arguments.resume)), separators=(",", ":")
            )
        )
    except (ControlInputError, ControlRuntimeError, ValueError) as error:
        print(f"dodge-dataset-collect: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
