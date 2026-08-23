from __future__ import annotations

import random
import sqlite3

import pytest

import dodge.dataset as dataset
from dodge.dataset import (
    EVALUATION_SEEDS,
    CollectorConfig,
    _checkpoint,
    _initialize_database,
    _load_checkpoint,
    _restore_or_initialize,
    _validate_config,
)
from dodge.headless import HeadlessTrace
from dodge.neat.state import PlayerState, RawState


def test_collector_reserves_exactly_ten_high_evaluation_seeds(tmp_path) -> None:
    config = CollectorConfig(database=tmp_path / "dataset.sqlite3", train_seeds=(0, 1))

    _validate_config(config)
    assert len(EVALUATION_SEEDS) == 10
    assert min(EVALUATION_SEEDS) > 30_000


def test_collector_rejects_training_seed_outside_reserved_range(tmp_path) -> None:
    config = CollectorConfig(
        database=tmp_path / "dataset.sqlite3", train_seeds=(30_001,)
    )

    with pytest.raises(ValueError, match="0 to 30000"):
        _validate_config(config)


def test_database_refuses_different_resume_configuration(tmp_path) -> None:
    path = tmp_path / "dataset.sqlite3"
    first = CollectorConfig(database=path, train_seeds=(0,))
    second = CollectorConfig(database=path, train_seeds=(1,))
    connection = sqlite3.connect(path)
    try:
        _initialize_database(connection, first, resume=False)
        with pytest.raises(ValueError, match="configuration differs"):
            _initialize_database(connection, second, resume=True)
    finally:
        connection.close()


def test_v43_allows_matching_action_hashes_on_different_training_seeds(
    tmp_path,
) -> None:
    config = CollectorConfig(database=tmp_path / "dataset.sqlite3", train_seeds=(0, 1))
    connection = sqlite3.connect(config.database)
    try:
        _initialize_database(connection, config, resume=False)
        connection.executemany(
            "INSERT INTO episodes(seed, action_hash, result_json, config_json) "
            "VALUES (?, 'same', '{}', '{}')",
            [(0,), (1,)],
        )
        assert connection.execute("SELECT count(*) FROM episodes").fetchone()[0] == 2
    finally:
        connection.close()


def test_v45_checkpoint_restores_pending_population_and_rng(tmp_path) -> None:
    config = CollectorConfig(database=tmp_path / "dataset.sqlite3", train_seeds=(0,))
    connection = sqlite3.connect(config.database)
    random_source = random.Random(9)
    population = [("left",), ("right",), ("up",), ("down",), ("neutral",)]
    try:
        _initialize_database(connection, config, resume=False)
        _checkpoint(connection, random_source, 1, 7, population)
        restored, seed_index, generation, restored_population = _restore_or_initialize(
            _load_checkpoint(connection), config
        )
        assert (seed_index, generation, restored_population) == (1, 7, population)
        assert restored.random() == random_source.random()
    finally:
        connection.close()


def test_v46_accepted_episode_writes_221_float_bootstrap_rows_in_one_transaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = CollectorConfig(database=tmp_path / "dataset.sqlite3", train_seeds=(0,))
    connection = sqlite3.connect(config.database)
    states = tuple(
        RawState(frame, PlayerState(64, 64, 0, 0, 4), (), ())
        for frame in (20, 24, 28, 32)
    )
    result = {
        "score": 0,
        "frames": 32,
        "survival_frames": 4,
        "seed": 0,
        "started": True,
        "died": True,
    }
    monkeypatch.setattr(dataset, "TARGET_SURVIVAL_FRAMES", 4)
    monkeypatch.setattr(dataset, "run_headless", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(
        dataset,
        "run_headless_trace",
        lambda *_args, **_kwargs: HeadlessTrace(result, states),
    )
    try:
        _initialize_database(connection, config, resume=False)
        dataset._accept_episode(connection, 0, ("left",), "digest", config)
        assert connection.execute("SELECT count(*) FROM episodes").fetchone()[0] == 1
        rows = connection.execute(
            "SELECT action, bootstrap, length(observation_f32) "
            "FROM steps ORDER BY action_index"
        ).fetchall()
        assert rows == [
            ("neutral", 1, 884),
            ("up", 1, 884),
            ("down", 1, 884),
        ]
    finally:
        connection.close()
