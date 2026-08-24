from __future__ import annotations

import json
import sqlite3
import struct

import pytest
import torch

from dodge.control import ControlInputError
from dodge.dataset import CollectorConfig, _initialize_database
from dodge.imitation.data import ACTION_INDEX, load_demonstrations, split_demonstrations
from dodge.imitation.model import BehaviorCloningMLP
from dodge.imitation.plot import plot_training_history
from dodge.imitation.train import (
    main as train_main,
)
from dodge.imitation.train import (
    save_training_history,
    train_behavior_cloning,
)
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
    assert demonstrations.seeds.tolist() == [0]
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


def test_v61_cli_auto_falls_back_to_cpu_without_cuda(
    monkeypatch, tmp_path, capsys
) -> None:
    path = tmp_path / "dataset.sqlite3"
    connection = sqlite3.connect(path)
    packed = struct.pack(f"<{OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION}f", *range(221))
    try:
        _initialize_database(
            connection, CollectorConfig(database=path, train_seeds=(0, 1)), resume=False
        )
        for seed in (0, 1):
            episode_id = connection.execute(
                "INSERT INTO episodes(seed, action_hash, result_json, config_json) "
                "VALUES (?, 'hash', '{}', '{}')",
                (seed,),
            ).lastrowid
            connection.execute(
                "INSERT INTO steps VALUES (?, 0, 0, 'left', 0, ?, '{}')",
                (episode_id, packed),
            )
        connection.commit()
    finally:
        connection.close()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    output = tmp_path / "model.pt"
    history = tmp_path / "history.json"
    assert (
        train_main(
            [
                "--database",
                str(path),
                "--epochs",
                "1",
                "--output",
                str(output),
                "--history",
                str(history),
                "--validation-seed-count",
                "1",
            ]
        )
        == 0
    )
    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines[0].startswith("epoch=1/1 train_loss=")
    assert "validation_loss=" in output_lines[0]
    payload = json.loads(output_lines[-1])
    assert payload["device"] == "cpu"
    assert payload["final_validation_loss"] is not None
    assert output.is_file()
    assert json.loads(history.read_text())["validation_seeds"] == [1]
    artifact = torch.load(output, weights_only=True)
    assert torch.isfinite(artifact["standard_deviation"]).all()
    with pytest.raises(ControlInputError, match="CUDA is unavailable"):
        train_behavior_cloning(load_demonstrations(path), device="cuda")


def test_v69_v70_hold_out_highest_seed_and_plot_losses(tmp_path) -> None:
    path = tmp_path / "dataset.sqlite3"
    connection = sqlite3.connect(path)
    packed = struct.pack(f"<{OBSERVATION_SIZE_WITH_TIME_TO_INTERSECTION}f", *range(221))
    try:
        _initialize_database(
            connection,
            CollectorConfig(database=path, train_seeds=(3, 4, 5)),
            resume=False,
        )
        for seed in (3, 4, 5):
            episode_id = connection.execute(
                "INSERT INTO episodes(seed, action_hash, result_json, config_json) "
                "VALUES (?, ?, '{}', '{}')",
                (seed, f"hash-{seed}"),
            ).lastrowid
            connection.execute(
                "INSERT INTO steps VALUES (?, 0, 0, 'left', 0, ?, '{}')",
                (episode_id, packed),
            )
        connection.commit()
    finally:
        connection.close()

    split = split_demonstrations(load_demonstrations(path), validation_seed_count=1)
    result = train_behavior_cloning(
        split.training,
        validation_demonstrations=split.validation,
        epochs=2,
        batch_size=2,
    )
    history = tmp_path / "history.json"
    plot = tmp_path / "losses.png"
    save_training_history(result, history, split.validation_seeds)
    plot_training_history(history, plot)

    assert split.validation_seeds == (5,)
    assert split.training.seeds.tolist() == [3, 4]
    assert split.validation.seeds.tolist() == [5]
    assert result.final_validation_loss is not None
    assert len(result.history) == 2
    assert plot.is_file()
    assert plot.stat().st_size > 0
