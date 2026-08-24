from __future__ import annotations

import logging
import random
import sqlite3

import pytest

import dodge.dataset as dataset
from dodge.dataset import (
    DEVELOPMENT_VALIDATION_SEEDS,
    EVALUATION_SEEDS,
    Champion,
    CollectorConfig,
    Genome,
    _checkpoint,
    _initialize_database,
    _load_champion,
    _load_checkpoint,
    _restore_or_initialize,
    _validate_config,
    export_database,
    reset_database,
)
from dodge.headless import HeadlessResult, HeadlessTrace
from dodge.neat.state import PlayerState, RawState


def test_collector_reserves_exactly_ten_high_evaluation_seeds(tmp_path) -> None:
    config = CollectorConfig(database=tmp_path / "dataset.sqlite3", train_seeds=(0, 1))

    _validate_config(config)
    assert len(EVALUATION_SEEDS) == 10
    assert min(EVALUATION_SEEDS) > 30_000


def test_v66_collector_reserves_development_validation_seeds(tmp_path) -> None:
    path = tmp_path / "dataset.sqlite3"
    config = CollectorConfig(database=path, train_seeds=(0,))
    connection = sqlite3.connect(path)
    try:
        _initialize_database(connection, config, resume=False)
        rows = connection.execute(
            "SELECT seed, role FROM seeds WHERE role='validation' ORDER BY seed"
        ).fetchall()
        connection.execute("DELETE FROM seeds WHERE role='validation'")
        connection.commit()
        _initialize_database(connection, config, resume=True)
        restored = connection.execute(
            "SELECT seed, role FROM seeds WHERE role='validation' ORDER BY seed"
        ).fetchall()
    finally:
        connection.close()

    assert (
        rows
        == restored
        == [(seed, "validation") for seed in DEVELOPMENT_VALIDATION_SEEDS]
    )
    with pytest.raises(ValueError, match="training and validation"):
        _validate_config(
            CollectorConfig(
                database=path, train_seeds=(DEVELOPMENT_VALIDATION_SEEDS[0],)
            )
        )


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


def test_v55_resume_loads_saved_collector_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "dataset.sqlite3"
    expected = CollectorConfig(
        database=path,
        train_seeds=(0,),
        generations_per_seed=500,
        population=5,
        workers=2,
        evolution_seed=9,
    )
    connection = sqlite3.connect(path)
    try:
        _initialize_database(connection, expected, resume=False)
    finally:
        connection.close()
    received: list[tuple[CollectorConfig, bool]] = []

    def capture(config: CollectorConfig, *, resume: bool) -> dataset.CollectionSummary:
        received.append((config, resume))
        return dataset.CollectionSummary(0, 0, ())

    monkeypatch.setattr(dataset, "collect", capture)

    assert dataset.main(["--database", str(path), "--resume"]) == 0
    assert received == [(expected, True)]


def test_v64_resume_worker_override_preserves_stored_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "dataset.sqlite3"
    expected = CollectorConfig(database=path, train_seeds=(0,), workers=8)
    connection = sqlite3.connect(path)
    try:
        _initialize_database(connection, expected, resume=False)
    finally:
        connection.close()
    received: list[tuple[CollectorConfig, bool, int | None]] = []

    def capture(
        config: CollectorConfig, *, resume: bool, workers: int | None = None
    ) -> dataset.CollectionSummary:
        received.append((config, resume, workers))
        return dataset.CollectionSummary(0, 0, ())

    monkeypatch.setattr(dataset, "collect", capture)

    assert dataset.main(["--database", str(path), "--resume", "--workers", "1"]) == 0
    assert received == [(expected, True, 1)]
    assert dataset._load_resume_config(path) == expected


def test_v56_resume_appends_new_seeds_with_stored_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "dataset.sqlite3"
    previous = CollectorConfig(
        database=path,
        train_seeds=(0,),
        generations_per_seed=500,
        population=5,
        workers=2,
        evolution_seed=9,
    )
    population: list[Genome] = [
        ("left",),
        ("right",),
        ("up",),
        ("down",),
        ("neutral",),
    ]
    connection = sqlite3.connect(path)
    try:
        _initialize_database(connection, previous, resume=False)
        _checkpoint(connection, random.Random(9), 1, 0, population)
    finally:
        connection.close()
    received: list[CollectorConfig] = []

    def capture(config: CollectorConfig, *, resume: bool) -> dataset.CollectionSummary:
        assert resume
        received.append(config)
        return dataset.CollectionSummary(0, 0, ())

    monkeypatch.setattr(dataset, "collect", capture)

    assert (
        dataset.main(["--database", str(path), "--resume", "--append-seeds", "2"]) == 0
    )
    expected = CollectorConfig(
        database=path,
        train_seeds=(0, 1, 2),
        generations_per_seed=500,
        population=5,
        workers=2,
        evolution_seed=9,
    )
    assert received == [expected]
    assert dataset._load_resume_config(path) == expected
    connection = sqlite3.connect(path)
    try:
        _, seed_index, generation, restored_population = dataset._restore_or_initialize(
            _load_checkpoint(connection), expected
        )
        assert (seed_index, generation, restored_population) == (1, 0, population)
        assert connection.execute(
            "SELECT seed FROM seeds ORDER BY seed"
        ).fetchall() == [
            (0,),
            (1,),
            (2,),
            *[(seed,) for seed in DEVELOPMENT_VALIDATION_SEEDS],
            *[(seed,) for seed in EVALUATION_SEEDS],
        ]
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
    population: list[Genome] = [
        ("left",),
        ("right",),
        ("up",),
        ("down",),
        ("neutral",),
    ]
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


def test_champion_persists_best_genome_with_checkpoint(tmp_path) -> None:
    config = CollectorConfig(database=tmp_path / "dataset.sqlite3", train_seeds=(0,))
    connection = sqlite3.connect(config.database)
    champion = Champion(0, 7, 1_213, ("left",))
    try:
        _initialize_database(connection, config, resume=False)
        _checkpoint(
            connection,
            random.Random(9),
            0,
            7,
            [("left",), ("right",), ("up",), ("down",), ("neutral",)],
            champion=champion,
        )
        assert _load_champion(connection, 0) == champion
    finally:
        connection.close()


def test_v53_reset_requires_confirmation_and_clears_collector_data(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "dataset.sqlite3"
    config = CollectorConfig(database=path, train_seeds=(0,))
    connection = sqlite3.connect(path)
    try:
        _initialize_database(connection, config, resume=False)
        _checkpoint(
            connection,
            random.Random(9),
            0,
            1,
            [("left",), ("right",), ("up",), ("down",), ("neutral",)],
            champion=Champion(0, 1, 100, ("left",)),
        )
    finally:
        connection.close()

    assert dataset.reset_main(["--database", str(path)]) == 1
    assert "pass --yes" in capsys.readouterr().err
    assert reset_database(path) == {
        "steps": 0,
        "episodes": 0,
        "champions": 1,
        "checkpoints": 1,
        "seeds": 21,
        "metadata": 1,
    }
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT count(*) FROM metadata").fetchone()[0] == 0
    finally:
        connection.close()


def test_v60_export_includes_committed_wal_data_without_mutating_source(
    tmp_path,
) -> None:
    source = tmp_path / "dataset.sqlite3"
    snapshot = tmp_path / "snapshot.sqlite3"
    connection = sqlite3.connect(source)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        _initialize_database(connection, CollectorConfig(database=source), resume=False)
        connection.execute(
            "INSERT INTO episodes(seed, action_hash, result_json, config_json) "
            "VALUES (0, 'hash', '{}', '{}')"
        )
        connection.commit()
        before = int(connection.execute("SELECT count(*) FROM episodes").fetchone()[0])
        result = export_database(source, snapshot)
        after = int(connection.execute("SELECT count(*) FROM episodes").fetchone()[0])
    finally:
        connection.close()

    exported = sqlite3.connect(snapshot)
    try:
        snapshot_episodes = int(
            exported.execute("SELECT count(*) FROM episodes").fetchone()[0]
        )
    finally:
        exported.close()

    assert result == {"database": str(snapshot), "episodes": 1}
    assert before == after == snapshot_episodes == 1


def test_reconstruct_champion_restores_recorded_high_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = CollectorConfig(
        database=tmp_path / "dataset.sqlite3",
        train_seeds=(0, 1),
        generations_per_seed=1,
        population=5,
    )
    connection = sqlite3.connect(config.database)
    try:
        _initialize_database(connection, config, resume=False)
    finally:
        connection.close()
    monkeypatch.setattr(
        dataset,
        "_evaluate_population",
        lambda seed, *_args: [10, 9, 8, 7, 6] if seed == 0 else [99, 8, 7, 6, 5],
    )

    champion = dataset.reconstruct_champion(config, 1)

    assert champion.seed == 1
    assert champion.generation == 1
    assert champion.survival_frames == 99


def test_v51_bootstrap_neutral_wait_ends_when_first_enemy_spawns() -> None:
    commands = dataset._commands(("neutral",))
    trace = dataset.run_headless_trace(commands, seed=1)

    assert [command.duration_ms for command in commands] == [
        50,
        300,
        100,
        100,
        516,
        133,
    ]
    assert len(trace.states[4].enemies) == 1


def test_collect_logs_generation_scores_and_seed_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    config = CollectorConfig(
        database=tmp_path / "dataset.sqlite3",
        train_seeds=(0,),
        generations_per_seed=1,
        population=5,
    )
    monkeypatch.setattr(
        dataset, "_evaluate_population", lambda *_args: [40, 30, 20, 10, 5]
    )

    with caplog.at_level(logging.INFO, logger="dodge.dataset"):
        summary = dataset.collect(config)

    assert summary.unsolved_seeds == (0,)
    assert (
        "seed=0 generation=1/1 best=40 median=20 target=1800 accepted=0/5"
        in caplog.text
    )
    assert "seed=0 deferred after 1 generations best_below_target" in caplog.text
    assert "collect complete seeds=1 accepted_episodes=0 deferred=1" in caplog.text


def test_v52_breeding_concentrates_mutation_in_genome_tail() -> None:
    class TailMutationRandom:
        def random(self) -> float:
            return 0.03

        def choice(self, _choices: tuple[str, ...]) -> str:
            return "right"

    config = CollectorConfig(population=6)
    parent: Genome = ("left", "left", "left", "left")
    ranked = [(10 - index, parent) for index in range(5)]

    population = dataset._breed_population(ranked, TailMutationRandom(), config)  # type: ignore[arg-type]

    assert population[:5] == [parent] * 5
    assert population[5] == ("left", "left", "left", "right")


def test_v54_accepted_episode_excludes_terminal_and_idle_trace_states(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    config = CollectorConfig(database=tmp_path / "dataset.sqlite3", train_seeds=(0,))
    connection = sqlite3.connect(config.database)
    states = tuple(
        RawState(frame, PlayerState(64, 64, 0, 0, 4), (), ())
        for frame in (20, 38, 44, 50, 81, 89, 90)
    )
    result: HeadlessResult = {
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
            ("neutral", 1, 884),
            ("left", 0, 884),
        ]
    finally:
        connection.close()
