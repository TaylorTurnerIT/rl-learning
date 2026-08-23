from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import random
import sqlite3
import statistics
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from dodge.control import ControlInputError, ControlRuntimeError, MovementCommand
from dodge.headless import (
    HeadlessResult,
    replay_commands,
    run_headless,
    run_headless_trace,
)
from dodge.neat.bridge import Direction
from dodge.neat.state import RawState, project_state

DATASET_VERSION = 3
DEFAULT_DATABASE = Path("history/dodge/dataset.sqlite3")
RESET_TABLES = ("steps", "episodes", "champions", "checkpoints", "seeds", "metadata")
STEP_FRAMES = 8
TARGET_SURVIVAL_FRAMES = 1_800
TRACES_PER_SEED = 5
PILOT_SEED_COUNT = 5
PILOT_GENERATIONS_PER_SEED = 100
DEFAULT_POPULATION = 50
ELITE_COUNT = 5
EARLY_MUTATION_RATE = 0.02
TAIL_MUTATION_RATE = 0.20
TAIL_MUTATION_FRACTION = 0.25
EVALUATION_SEEDS = tuple(range(30_001, 30_011))
TRAINING_SEED_MAX = 30_000
BootstrapAction = Direction | Literal["x"]
Genome = tuple[Direction, ...]
Population = list[Genome]

BOOTSTRAP: tuple[tuple[BootstrapAction, int], ...] = (
    ("x", 3),
    ("neutral", 18),
    ("up", 6),
    ("down", 6),
    ("neutral", 31),
)
ACTION_CHOICES: tuple[Direction, ...] = (
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
LOGGER = logging.getLogger(__name__)


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
            "early_mutation_rate": EARLY_MUTATION_RATE,
            "tail_mutation_rate": TAIL_MUTATION_RATE,
            "tail_mutation_fraction": TAIL_MUTATION_FRACTION,
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


@dataclass(frozen=True, slots=True)
class Champion:
    seed: int
    generation: int
    survival_frames: int
    genome: Genome

    @property
    def action_hash(self) -> str:
        return _genome_hash(self.genome)


def collect(config: CollectorConfig, *, resume: bool = False) -> CollectionSummary:
    _validate_config(config)
    LOGGER.info(
        "collect start seeds=%d population=%d generations_per_seed=%d target=%d "
        "mutation=%.0f%%/%.0f%% tail=%.0f%%",
        len(config.train_seeds),
        config.population,
        config.generations_per_seed,
        TARGET_SURVIVAL_FRAMES,
        EARLY_MUTATION_RATE * 100,
        TAIL_MUTATION_RATE * 100,
        TAIL_MUTATION_FRACTION * 100,
    )
    connection = _open_database(config.database)
    try:
        _initialize_database(connection, config, resume=resume)
        checkpoint = _load_checkpoint(connection)
        random_source, seed_index, generation, population = _restore_or_initialize(
            checkpoint, config
        )
        if checkpoint is not None:
            LOGGER.info(
                "collect resume seed_index=%d generation=%d", seed_index, generation
            )
        unsolved: list[int] = []
        while seed_index < len(config.train_seeds):
            seed = config.train_seeds[seed_index]
            accepted_hashes = _accepted_hashes(connection, seed)
            champion: Champion | None = None
            LOGGER.info(
                "seed=%d start accepted=%d/%d generation=%d/%d",
                seed,
                len(accepted_hashes),
                TRACES_PER_SEED,
                generation,
                config.generations_per_seed,
            )
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
                champion = Champion(seed, generation, ranked[0][0], ranked[0][1])
                LOGGER.info(
                    "seed=%d generation=%d/%d best=%d median=%d target=%d "
                    "accepted=%d/%d",
                    seed,
                    generation,
                    config.generations_per_seed,
                    ranked[0][0],
                    int(statistics.median(results)),
                    TARGET_SURVIVAL_FRAMES,
                    len(accepted_hashes),
                    TRACES_PER_SEED,
                )
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
                    LOGGER.info(
                        "seed=%d accepted=%d/%d survival=%d hash=%s",
                        seed,
                        len(accepted_hashes),
                        TRACES_PER_SEED,
                        fitness,
                        digest[:12],
                    )
                if len(accepted_hashes) >= TRACES_PER_SEED:
                    break
                population = _breed_population(ranked, random_source, config)
                _checkpoint(
                    connection,
                    random_source,
                    seed_index,
                    generation,
                    population,
                    champion=champion,
                )
            else:
                unsolved.append(seed)
                LOGGER.info(
                    "seed=%d deferred after %d generations best_below_target",
                    seed,
                    config.generations_per_seed,
                )
            seed_index += 1
            generation = 0
            population = _new_population(random_source, config)
            _checkpoint(
                connection,
                random_source,
                seed_index,
                generation,
                population,
                champion=champion,
            )
        summary = CollectionSummary(
            completed_seeds=len(config.train_seeds),
            accepted_episodes=_episode_count(connection),
            unsolved_seeds=tuple(unsolved),
        )
        LOGGER.info(
            "collect complete seeds=%d accepted_episodes=%d deferred=%d",
            summary.completed_seeds,
            summary.accepted_episodes,
            len(summary.unsolved_seeds),
        )
        return summary
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
        CREATE TABLE IF NOT EXISTS champions (
          seed INTEGER PRIMARY KEY REFERENCES seeds(seed),
          generation INTEGER NOT NULL,
          survival_frames INTEGER NOT NULL,
          action_hash TEXT NOT NULL,
          genome_json TEXT NOT NULL
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


def _load_resume_config(database: Path) -> CollectorConfig:
    if not database.is_file():
        raise ControlInputError(f"dataset database does not exist: {database}")
    connection = _open_database(database)
    try:
        stored = connection.execute(
            "SELECT value FROM metadata WHERE key='config'"
        ).fetchone()
    finally:
        connection.close()
    if stored is None:
        raise ControlInputError("dataset has no stored collector configuration")
    try:
        values = json.loads(stored[0])
        if not isinstance(values, dict):
            raise ValueError
        config = CollectorConfig(
            database=database,
            train_seeds=tuple(values["train_seeds"]),
            generations_per_seed=values["generations_per_seed"],
            population=values["population"],
            workers=values["workers"],
            evolution_seed=values["evolution_seed"],
        )
        _validate_config(config)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ControlInputError("stored collector configuration is invalid") from error
    encoded = json.dumps(config.to_json(), sort_keys=True, separators=(",", ":"))
    if stored[0] != encoded:
        raise ControlInputError("stored collector configuration is incompatible")
    return config


def append_training_seeds(database: Path, count: int) -> CollectorConfig:
    if count < 1:
        raise ControlInputError("--append-seeds must be positive")
    previous = _load_resume_config(database)
    first_seed = max(previous.train_seeds, default=-1) + 1
    added_seeds = tuple(range(first_seed, first_seed + count))
    config = CollectorConfig(
        database=database,
        train_seeds=(*previous.train_seeds, *added_seeds),
        generations_per_seed=previous.generations_per_seed,
        population=previous.population,
        workers=previous.workers,
        evolution_seed=previous.evolution_seed,
    )
    try:
        _validate_config(config)
    except ValueError as error:
        raise ControlInputError(
            "appended seeds exceed the training seed range"
        ) from error
    previous_encoded = json.dumps(
        previous.to_json(), sort_keys=True, separators=(",", ":")
    )
    encoded = json.dumps(config.to_json(), sort_keys=True, separators=(",", ":"))
    connection = _open_database(database)
    try:
        with connection:
            stored = connection.execute(
                "SELECT value FROM metadata WHERE key='config'"
            ).fetchone()
            if stored is None or stored[0] != previous_encoded:
                raise ControlRuntimeError(
                    "collector configuration changed while appending"
                )
            connection.execute(
                "UPDATE metadata SET value=? WHERE key='config'", (encoded,)
            )
            connection.executemany(
                "INSERT INTO seeds(seed, role) VALUES (?, 'training')",
                [(seed,) for seed in added_seeds],
            )
    finally:
        connection.close()
    LOGGER.info(
        "collect append seeds=%d..%d total=%d",
        added_seeds[0],
        added_seeds[-1],
        len(config.train_seeds),
    )
    return config


def reset_database(database: Path) -> dict[str, int]:
    if not database.is_file():
        raise ControlInputError(f"dataset database does not exist: {database}")
    connection = _open_database(database)
    try:
        known_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = set(RESET_TABLES) - known_tables
        if missing:
            names = ", ".join(sorted(missing))
            raise ControlInputError(f"dataset database is missing tables: {names}")
        removed = {
            table: int(
                connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            )
            for table in RESET_TABLES
        }
        with connection:
            for table in RESET_TABLES:
                connection.execute(f"DELETE FROM {table}")
        return removed
    finally:
        connection.close()


def _restore_or_initialize(
    checkpoint: bytes | None, config: CollectorConfig
) -> tuple[random.Random, int, int, Population]:
    random_source = random.Random(config.evolution_seed)
    if checkpoint is None:
        return random_source, 0, 0, _new_population(random_source, config)
    state = pickle.loads(checkpoint)
    random_source.setstate(state["rng"])
    return random_source, state["seed_index"], state["generation"], state["population"]


def _new_population(
    random_source: random.Random, config: CollectorConfig
) -> Population:
    return [
        tuple(random_source.choice(ACTION_CHOICES) for _ in range(config.genome_length))
        for _ in range(config.population)
    ]


def _evaluate_population(
    seed: int, population: Population, config: CollectorConfig
) -> list[int]:
    with ThreadPoolExecutor(
        max_workers=min(config.workers, len(population))
    ) as executor:
        return list(executor.map(lambda genome: _fitness(seed, genome), population))


def _fitness(seed: int, genome: Genome) -> int:
    return int(
        run_headless(_commands(genome), seed=seed, wait_for_game_start=True)[
            "survival_frames"
        ]
    )


def _commands(genome: Genome) -> list[MovementCommand]:
    return [
        *[_command(move, frames) for move, frames in BOOTSTRAP],
        *[_command(action, STEP_FRAMES) for action in genome],
    ]


def _command(action: BootstrapAction, frames: int) -> MovementCommand:
    return MovementCommand(action, (frames * 1_000) // 60)


def _breed_population(
    ranked: list[tuple[int, Genome]],
    random_source: random.Random,
    config: CollectorConfig,
) -> Population:
    parents = [genome for _, genome in ranked[:ELITE_COUNT]]
    population = parents.copy()
    while len(population) < config.population:
        parent = parents[len(population) % len(parents)]
        population.append(
            tuple(
                random_source.choice(ACTION_CHOICES)
                if random_source.random() < _mutation_rate(index, len(parent))
                else action
                for index, action in enumerate(parent)
            )
        )
    return population


def _mutation_rate(index: int, genome_length: int) -> float:
    if index < 0 or index >= genome_length:
        raise ValueError("mutation index must be within genome")
    tail_start = int(genome_length * (1 - TAIL_MUTATION_FRACTION))
    return TAIL_MUTATION_RATE if index >= tail_start else EARLY_MUTATION_RATE


def _accept_episode(
    connection: sqlite3.Connection,
    seed: int,
    genome: Genome,
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
    actions: tuple[Direction, ...] = (
        "neutral",
        "up",
        "down",
        "neutral",
        *genome,
    )
    state_count = min(len(trace.states) - 1, len(actions))
    states = trace.states[:state_count]
    if not states:
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
                    int(index < len(BOOTSTRAP) - 1),
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


def _genome_hash(genome: Genome) -> str:
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
    population: Population,
    *,
    champion: Champion | None = None,
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
        if champion is not None:
            _store_champion(connection, champion)
        connection.execute("INSERT OR REPLACE INTO checkpoints VALUES (1, ?)", (state,))


def _load_checkpoint(connection: sqlite3.Connection) -> bytes | None:
    row = connection.execute("SELECT state FROM checkpoints WHERE id=1").fetchone()
    return None if row is None else row[0]


def _store_champion(connection: sqlite3.Connection, champion: Champion) -> None:
    connection.execute(
        "INSERT INTO champions("
        "seed, generation, survival_frames, action_hash, genome_json) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(seed) DO UPDATE SET "
        "generation=excluded.generation, survival_frames=excluded.survival_frames, "
        "action_hash=excluded.action_hash, genome_json=excluded.genome_json "
        "WHERE excluded.survival_frames > champions.survival_frames",
        (
            champion.seed,
            champion.generation,
            champion.survival_frames,
            champion.action_hash,
            json.dumps(champion.genome, separators=(",", ":")),
        ),
    )


def _load_champion(connection: sqlite3.Connection, seed: int) -> Champion:
    row = connection.execute(
        "SELECT generation, survival_frames, genome_json FROM champions WHERE seed=?",
        (seed,),
    ).fetchone()
    if row is None:
        raise ControlInputError(f"no champion recorded for seed {seed}")
    try:
        values = json.loads(row[2])
        if not isinstance(values, list) or any(
            action not in ACTION_CHOICES for action in values
        ):
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ControlRuntimeError(f"invalid champion genome for seed {seed}") from error
    return Champion(seed, int(row[0]), int(row[1]), cast(Genome, tuple(values)))


def replay_champion(database: Path, seed: int) -> HeadlessResult:
    connection = _open_database(database)
    try:
        champion = _load_champion(connection, seed)
    finally:
        connection.close()
    LOGGER.info(
        "replay seed=%d generation=%d survival=%d hash=%s",
        champion.seed,
        champion.generation,
        champion.survival_frames,
        champion.action_hash[:12],
    )
    return replay_commands(
        _commands(champion.genome), seed=seed, wait_for_game_start=True
    )


def reconstruct_champion(config: CollectorConfig, target_seed: int) -> Champion:
    _validate_config(config)
    if target_seed not in config.train_seeds:
        raise ControlInputError(f"seed {target_seed} is not configured for training")
    connection = _open_database(config.database)
    try:
        _initialize_database(connection, config, resume=True)
        random_source = random.Random(config.evolution_seed)
        population = _new_population(random_source, config)
        for seed in config.train_seeds:
            accepted_hashes = _accepted_hashes(connection, seed)
            for generation in range(1, config.generations_per_seed + 1):
                results = _evaluate_population(seed, population, config)
                ranked = sorted(zip(results, population, strict=True), reverse=True)
                champion = Champion(seed, generation, ranked[0][0], ranked[0][1])
                LOGGER.info(
                    "reconstruct seed=%d generation=%d/%d best=%d",
                    seed,
                    generation,
                    config.generations_per_seed,
                    champion.survival_frames,
                )
                if seed == target_seed:
                    with connection:
                        _store_champion(connection, champion)
                for fitness, genome in ranked:
                    if fitness < TARGET_SURVIVAL_FRAMES:
                        break
                    accepted_hashes.add(_genome_hash(genome))
                if len(accepted_hashes) >= TRACES_PER_SEED:
                    break
                population = _breed_population(ranked, random_source, config)
            if seed == target_seed:
                champion = _load_champion(connection, seed)
                return champion
            population = _new_population(random_source, config)
    finally:
        connection.close()
    raise AssertionError("configured target seed was not reconstructed")


def _add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=PILOT_SEED_COUNT)
    parser.add_argument(
        "--generations-per-seed", type=int, default=PILOT_GENERATIONS_PER_SEED
    )
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--evolution-seed", type=int, default=42)


def _config_from_arguments(arguments: argparse.Namespace) -> CollectorConfig:
    return CollectorConfig(
        database=arguments.database,
        train_seeds=tuple(
            range(arguments.seed_start, arguments.seed_start + arguments.seed_count)
        ),
        generations_per_seed=arguments.generations_per_seed,
        population=arguments.population,
        workers=arguments.workers,
        evolution_seed=arguments.evolution_seed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-dataset-collect")
    _add_collection_arguments(parser)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--append-seeds", type=int)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = (
            _load_resume_config(arguments.database)
            if arguments.resume
            else _config_from_arguments(arguments)
        )
        if arguments.append_seeds is not None:
            if not arguments.resume:
                raise ControlInputError("--append-seeds requires --resume")
            config = append_training_seeds(arguments.database, arguments.append_seeds)
        print(
            json.dumps(
                asdict(collect(config, resume=arguments.resume)), separators=(",", ":")
            )
        )
    except (ControlInputError, ControlRuntimeError, ValueError) as error:
        print(f"dodge-dataset-collect: {error}", file=sys.stderr)
        return 1
    return 0


def replay_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-dataset-replay")
    parser.add_argument("database", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        print(json.dumps(replay_champion(arguments.database, arguments.seed)))
    except (ControlInputError, ControlRuntimeError, OSError, sqlite3.Error) as error:
        print(f"dodge-dataset-replay: {error}", file=sys.stderr)
        return 1
    return 0


def reconstruct_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-dataset-reconstruct")
    _add_collection_arguments(parser)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        champion = reconstruct_champion(
            _config_from_arguments(arguments), arguments.seed
        )
        print(
            json.dumps(
                {
                    "seed": champion.seed,
                    "generation": champion.generation,
                    "survival_frames": champion.survival_frames,
                    "action_hash": champion.action_hash,
                },
                separators=(",", ":"),
            )
        )
    except (ControlInputError, ControlRuntimeError, ValueError, sqlite3.Error) as error:
        print(f"dodge-dataset-reconstruct: {error}", file=sys.stderr)
        return 1
    return 0


def reset_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-dataset-reset")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--yes", action="store_true", help="delete collector data")
    arguments = parser.parse_args(argv)
    if not arguments.yes:
        print(
            "dodge-dataset-reset: pass --yes to delete collector data", file=sys.stderr
        )
        return 1
    try:
        removed = reset_database(arguments.database)
    except (ControlInputError, OSError, sqlite3.Error) as error:
        print(f"dodge-dataset-reset: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"database": str(arguments.database), "removed": removed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
