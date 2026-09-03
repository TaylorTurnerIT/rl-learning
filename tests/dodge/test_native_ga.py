from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from dodge.control import ControlInputError
from dodge.dataset import BOOTSTRAP, STEP_FRAMES
from dodge.native.ga import load_longest_episode


def _create_database(
    path: Path,
    episodes: list[tuple[int, int, list[str], int, int]],
) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE episodes (
              id INTEGER PRIMARY KEY,
              seed INTEGER NOT NULL,
              action_hash TEXT NOT NULL,
              result_json TEXT NOT NULL,
              config_json TEXT NOT NULL
            );
            CREATE TABLE steps (
              episode_id INTEGER NOT NULL,
              action_index INTEGER NOT NULL,
              frame INTEGER NOT NULL,
              action TEXT NOT NULL,
              bootstrap INTEGER NOT NULL,
              observation_f32 BLOB NOT NULL,
              raw_state_json TEXT NOT NULL,
              PRIMARY KEY (episode_id, action_index)
            );
            """
        )
        for episode_id, seed, genome, survival_frames, total_frames in episodes:
            action_hash = hashlib.sha256(
                json.dumps(genome, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            result = {
                "died": True,
                "frames": total_frames,
                "score": 0,
                "seed": seed,
                "started": True,
                "survival_frames": survival_frames,
            }
            connection.execute(
                "INSERT INTO episodes VALUES (?, ?, ?, ?, '{}')",
                (
                    episode_id,
                    seed,
                    action_hash,
                    json.dumps(result, sort_keys=True, separators=(",", ":")),
                ),
            )
            actions = [action for action, _ in BOOTSTRAP[1:]] + genome
            connection.executemany(
                "INSERT INTO steps VALUES (?, ?, ?, ?, ?, x'', '{}')",
                [
                    (
                        episode_id,
                        index,
                        20 + index,
                        action,
                        int(index < len(BOOTSTRAP) - 1),
                    )
                    for index, action in enumerate(actions)
                ],
            )


def test_load_longest_ga_episode_reconstructs_full_schedule(tmp_path: Path) -> None:
    database = tmp_path / "dataset.sqlite3"
    _create_database(
        database,
        [
            (1, 7, ["left"], 12, 20),
            (2, 9, ["up_right", "neutral"], 24, 36),
        ],
    )

    episode = load_longest_episode(database)

    assert episode.episode_id == 2
    assert episode.seed == 9
    assert episode.survival_frames == 24
    assert episode.total_frames == 36
    assert episode.recorded_steps == 6
    assert episode.genome_actions == ("up_right", "neutral")
    assert [(command.move, command.duration_ms) for command in episode.commands] == [
        ("x", 50),
        ("neutral", 300),
        ("up", 100),
        ("down", 100),
        ("neutral", 516),
        ("up_right", (STEP_FRAMES * 1_000) // 60),
        ("neutral", (STEP_FRAMES * 1_000) // 60),
    ]


def test_load_ga_episode_can_select_explicit_id(tmp_path: Path) -> None:
    database = tmp_path / "dataset.sqlite3"
    _create_database(database, [(1, 7, ["left"], 12, 20)])

    episode = load_longest_episode(database, episode_id=1)

    assert episode.episode_id == 1
    assert episode.genome_actions == ("left",)


def test_v12_load_longest_ga_episode_skips_incomplete_higher_ranked_row(
    tmp_path: Path,
) -> None:
    database = tmp_path / "dataset.sqlite3"
    _create_database(
        database,
        [
            (1, 7, ["left"], 12, 20),
            (2, 9, ["right"], 24, 36),
        ],
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "DELETE FROM steps WHERE episode_id=? AND action_index=?", (2, 4)
        )

    episode = load_longest_episode(database)

    assert episode.episode_id == 1
    assert episode.skipped_episode_ids == (2,)


def test_load_ga_episode_rejects_a_tampered_genome_hash(tmp_path: Path) -> None:
    database = tmp_path / "dataset.sqlite3"
    _create_database(database, [(1, 7, ["left"], 12, 20)])
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE episodes SET action_hash='tampered'")

    with pytest.raises(ControlInputError, match="action_hash"):
        load_longest_episode(database, episode_id=1)
