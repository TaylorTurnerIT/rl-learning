from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dodge.control import ControlInputError, ControlRuntimeError
from dodge.dataset import ACTION_CHOICES, DEFAULT_DATABASE
from dodge.neat.state import OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION

ACTION_INDEX = {action: index for index, action in enumerate(ACTION_CHOICES)}


@dataclass(frozen=True, slots=True)
class Demonstrations:
    """Feature matrix and categorical labels read from accepted GA episodes."""

    observations: np.ndarray
    actions: np.ndarray
    seeds: np.ndarray

    @property
    def count(self) -> int:
        return len(self.actions)


@dataclass(frozen=True, slots=True)
class DemonstrationSplit:
    """Seed-held training and validation examples."""

    training: Demonstrations
    validation: Demonstrations
    validation_seeds: tuple[int, ...]


def load_demonstrations(database: Path = DEFAULT_DATABASE) -> Demonstrations:
    """Read a consistent, non-mutating snapshot of learned decision rows."""
    if not database.is_file():
        raise ControlInputError(f"dataset database does not exist: {database}")
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            "SELECT episodes.seed, steps.observation_f32, steps.action FROM steps "
            "JOIN episodes ON episodes.id=steps.episode_id "
            "JOIN seeds ON seeds.seed=episodes.seed "
            "WHERE steps.bootstrap=0 AND seeds.role='training' "
            "ORDER BY steps.episode_id, steps.action_index"
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ControlInputError("dataset contains no learned decision rows")

    observations = np.empty(
        (len(rows), OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION), dtype=np.float32
    )
    actions = np.empty(len(rows), dtype=np.int64)
    seeds = np.empty(len(rows), dtype=np.int64)
    expected_bytes = OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION * 4
    for index, (seed, packed_observation, action) in enumerate(rows):
        if len(packed_observation) != expected_bytes:
            raise ControlRuntimeError("collector observation has unexpected width")
        if action not in ACTION_INDEX:
            raise ControlRuntimeError(f"collector action is not recognized: {action}")
        observations[index] = np.frombuffer(packed_observation, dtype="<f4")
        actions[index] = ACTION_INDEX[action]
        seeds[index] = seed
    return Demonstrations(observations, actions, seeds)


def split_demonstrations(
    demonstrations: Demonstrations, validation_seed_count: int = 10
) -> DemonstrationSplit:
    """Hold out the highest accepted training seed IDs for validation."""
    seed_values = np.unique(demonstrations.seeds)
    if validation_seed_count < 1:
        raise ValueError("validation seed count must be positive")
    if len(seed_values) <= validation_seed_count:
        raise ControlInputError(
            "dataset needs more accepted training seeds than validation seed count"
        )
    validation_seeds = tuple(int(seed) for seed in seed_values[-validation_seed_count:])
    validation_mask = np.isin(demonstrations.seeds, validation_seeds)
    return DemonstrationSplit(
        training=_subset(demonstrations, ~validation_mask),
        validation=_subset(demonstrations, validation_mask),
        validation_seeds=validation_seeds,
    )


def _subset(demonstrations: Demonstrations, mask: np.ndarray) -> Demonstrations:
    return Demonstrations(
        demonstrations.observations[mask],
        demonstrations.actions[mask],
        demonstrations.seeds[mask],
    )
