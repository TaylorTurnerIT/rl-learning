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

    @property
    def count(self) -> int:
        return len(self.actions)


def load_demonstrations(database: Path = DEFAULT_DATABASE) -> Demonstrations:
    """Read a consistent, non-mutating snapshot of learned decision rows."""
    if not database.is_file():
        raise ControlInputError(f"dataset database does not exist: {database}")
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            "SELECT observation_f32, action FROM steps "
            "WHERE bootstrap=0 ORDER BY episode_id, action_index"
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ControlInputError("dataset contains no learned decision rows")

    observations = np.empty(
        (len(rows), OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION), dtype=np.float32
    )
    actions = np.empty(len(rows), dtype=np.int64)
    expected_bytes = OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION * 4
    for index, (packed_observation, action) in enumerate(rows):
        if len(packed_observation) != expected_bytes:
            raise ControlRuntimeError("collector observation has unexpected width")
        if action not in ACTION_INDEX:
            raise ControlRuntimeError(f"collector action is not recognized: {action}")
        observations[index] = np.frombuffer(packed_observation, dtype="<f4")
        actions[index] = ACTION_INDEX[action]
    return Demonstrations(observations, actions)
