# ruff: noqa: E402
from __future__ import annotations

import sqlite3
import struct

import pytest

torch = pytest.importorskip("torch")

from dodge.dataset import CollectorConfig, _initialize_database
from dodge.imitation.data import ACTION_INDEX, load_demonstrations
from dodge.imitation.model import BehaviorCloningMLP
from dodge.imitation.train import main as train_main
from dodge.imitation.train import train_behavior_cloning
from dodge.neat.state import OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION


def test_v57_loader_reads_only_learned_rows(tmp_path) -> None:
    path = tmp_path / "dataset.sqlite3"
    connection = sqlite3.connect(path)
    packed = struct.pack(f"<{OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION}f", *range(221))
    try:
        _initialize_database(connection, CollectorConfig(database=path), resume=False)
        episode_id = connection.execute(
            "INSERT INTO episodes(seed, action_hash, result_json, config_json) "
            "VALUES (0, 'hash', '{}', '{}')"
        ).lastrowid
        connection.executemany(
            "INSERT INTO steps VALUES (?, ?, ?, ?, ?, ?, '{}')",
            [
                (episode_id, 0, 0, "neutral", 1, packed),
                (episode_id, 1, 8, "up_left", 0, packed),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    demonstrations = load_demonstrations(path)

    assert demonstrations.observations.shape == (1, 221)
    assert demonstrations.actions.tolist() == [ACTION_INDEX["up_left"]]
    assert demonstrations.observations[0, -1] == 220


def test_v58_mlp_produces_nine_trainable_direction_logits(tmp_path) -> None:
    model = BehaviorCloningMLP(hidden_size=8)
    observations = torch.zeros((2, OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION))

    logits = model(observations)
    loss = torch.nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
    loss.backward()

    assert logits.shape == (2, 9)
    assert all(parameter.grad is not None for parameter in model.parameters())
    with pytest.raises(ValueError, match="observations must have shape"):
        model(torch.zeros((2, 220)))


def test_behavior_cloning_trains_on_loaded_demonstrations(tmp_path) -> None:
    path = tmp_path / "dataset.sqlite3"
    connection = sqlite3.connect(path)
    packed = struct.pack(f"<{OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION}f", *range(221))
    try:
        _initialize_database(connection, CollectorConfig(database=path), resume=False)
        episode_id = connection.execute(
            "INSERT INTO episodes(seed, action_hash, result_json, config_json) "
            "VALUES (0, 'hash', '{}', '{}')"
        ).lastrowid
        connection.executemany(
            "INSERT INTO steps VALUES (?, ?, ?, ?, ?, ?, '{}')",
            [(episode_id, index, index * 8, "left", 0, packed) for index in range(4)],
        )
        connection.commit()
    finally:
        connection.close()

    result = train_behavior_cloning(
        load_demonstrations(path), epochs=1, batch_size=2, learning_rate=1e-3
    )

    assert result.examples == 4
    assert result.final_loss > 0


def test_v61_cli_defaults_to_cuda_and_fails_without_gpu(monkeypatch, tmp_path) -> None:
    path = tmp_path / "dataset.sqlite3"
    connection = sqlite3.connect(path)
    packed = struct.pack(f"<{OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION}f", *range(221))
    try:
        _initialize_database(connection, CollectorConfig(database=path), resume=False)
        episode_id = connection.execute(
            "INSERT INTO episodes(seed, action_hash, result_json, config_json) "
            "VALUES (0, 'hash', '{}', '{}')"
        ).lastrowid
        connection.execute(
            "INSERT INTO steps VALUES (?, 0, 0, 'left', 0, ?, '{}')",
            (episode_id, packed),
        )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert train_main(["--database", str(path), "--epochs", "1"]) == 1
