from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dodge.dataset import ACTION_CHOICES
from dodge.neat.state import PlayerState, RawState
from dodge.ng.dqn import (
    WAYPOINT_OBSERVATION_SIZE,
    DQNConfig,
    DuelingWaypointDQN,
    NStepAccumulator,
    ReplayBuffer,
    _checkpoint_payload,
    _load_checkpoint,
    encode_waypoint_observation,
)
from dodge.ng.manifest import SeedManifest
from dodge.ng.waypoint import WaypointGrid

pytest.importorskip("dodge_native")


def _state() -> RawState:
    return RawState(
        frame=17,
        player=PlayerState(66.0, 66.0, 0.0, 0.0, 4.0),
        enemies=(),
        aoes=(),
    )


def _transition(
    value: float,
    *,
    terminated: bool = False,
    truncated: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    observation = np.full(WAYPOINT_OBSERVATION_SIZE, value, dtype=np.float32)
    next_observation = np.full(
        WAYPOINT_OBSERVATION_SIZE,
        value + 1,
        dtype=np.float32,
    )
    return observation, next_observation


def test_waypoint_observation_declares_structured_state_and_grid_cell() -> None:
    observation = encode_waypoint_observation(_state(), WaypointGrid(32))

    assert observation.shape == (WAYPOINT_OBSERVATION_SIZE,)
    assert np.isfinite(observation).all()
    assert observation[-4:-2].tolist() == [0.5, 0.5]
    assert observation[-2:].tolist() == [0.0, 0.0]


def test_dueling_waypoint_dqn_returns_one_value_per_native_action() -> None:
    model = DuelingWaypointDQN(hidden_size=16)
    values = model(torch.zeros((4, WAYPOINT_OBSERVATION_SIZE)))

    assert values.shape == (4, len(ACTION_CHOICES))
    assert torch.isfinite(values).all()
    with pytest.raises(ValueError, match="observations must have shape"):
        model(torch.zeros((4, WAYPOINT_OBSERVATION_SIZE + 1)))


def test_n_step_replay_preserves_target_and_stops_at_episode_boundary() -> None:
    replay = ReplayBuffer(capacity=8, observation_size=WAYPOINT_OBSERVATION_SIZE)
    accumulator = NStepAccumulator(
        lane_count=1,
        n_step=3,
        gamma=0.9,
        replay=replay,
    )

    for index in range(3):
        observation, next_observation = _transition(float(index))
        accumulator.append(
            0,
            observation,
            index,
            (index + 1, index + 2),
            float(index + 1),
            next_observation,
            False,
            False,
        )

    assert replay.size == 1
    assert replay.rewards[0] == pytest.approx(1.0 + 0.9 * 2.0 + 0.9**2 * 3.0)
    assert replay.discounts[0] == pytest.approx(0.9**3)
    assert replay.target_columns[0] == 1
    assert replay.target_rows[0] == 2
    assert not replay.terminated[0]
    assert not replay.truncated[0]

    observation, next_observation = _transition(3.0)
    accumulator.append(
        0,
        observation,
        3,
        (4, 5),
        4.0,
        next_observation,
        True,
        False,
    )

    assert replay.size == 4
    assert np.all(replay.discounts[1:4] == 0.0)
    assert np.all(replay.terminated[1:4])
    assert replay.rewards[1] == pytest.approx(2.0 + 0.9 * 3.0 + 0.9**2 * 4.0)

    observation, next_observation = _transition(10.0)
    accumulator.append(
        0,
        observation,
        0,
        (0, 0),
        10.0,
        next_observation,
        False,
        False,
    )
    assert replay.size == 4


def test_dqn_config_rejects_zero_evaluation_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        DQNConfig(eval_every=0).validate()


def test_checkpoint_round_trip_restores_contract_and_replay(
    tmp_path: Path,
) -> None:
    manifest = SeedManifest.fresh_default(
        seed_start=30_200,
        seed_count=10,
        split_seed=17,
    )
    config = DQNConfig(
        total_steps=4,
        batch_size=2,
        replay_capacity=8,
        warmup_steps=1,
        target_update_interval=2,
        checkpoint_every=2,
        eval_every=2,
        hidden_size=16,
        native_lanes=2,
    )
    config.validate()
    model = DuelingWaypointDQN(hidden_size=config.hidden_size)
    target_model = DuelingWaypointDQN(hidden_size=config.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity, WAYPOINT_OBSERVATION_SIZE)
    observation, next_observation = _transition(1.0)
    replay.add(
        observation,
        2,
        (3, 4),
        3.0,
        next_observation,
        config.gamma,
        False,
        False,
        1,
    )
    accumulator = NStepAccumulator(
        config.native_lanes,
        config.n_step,
        config.gamma,
        replay,
    )
    rng = np.random.default_rng(9)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        _checkpoint_payload(
            model,
            target_model,
            optimizer,
            config,
            manifest,
            step=2,
            seed_cursor=5,
            best_inner={"mean_survival_frames": 12.0, "step": 2},
            replay=replay,
            replay_accumulator=accumulator,
            rng=rng,
            total_native_steps=64,
            metrics=[{"step": 1}],
        ),
        checkpoint,
    )

    restored_model = DuelingWaypointDQN(hidden_size=config.hidden_size)
    restored_target = DuelingWaypointDQN(hidden_size=config.hidden_size)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=config.learning_rate
    )
    restored_replay = ReplayBuffer(config.replay_capacity, WAYPOINT_OBSERVATION_SIZE)
    restored_accumulator = NStepAccumulator(
        config.native_lanes,
        config.n_step,
        config.gamma,
        restored_replay,
    )
    restored_rng = np.random.default_rng(99)
    result = _load_checkpoint(
        checkpoint,
        restored_model,
        restored_target,
        restored_optimizer,
        config,
        manifest,
        restored_replay,
        restored_accumulator,
        restored_rng,
    )

    assert result[:3] == (2, 5, {"mean_survival_frames": 12.0, "step": 2})
    assert result[3] is None
    assert result[4] == 64
    assert result[5] == [{"step": 1}]
    assert restored_replay.size == 1
    assert restored_replay.target_columns[0] == 3
    np.testing.assert_array_equal(
        restored_replay.observations[0], replay.observations[0]
    )


def test_checkpoint_rejects_different_manifest(tmp_path: Path) -> None:
    manifest = SeedManifest.fresh_default(
        seed_start=30_200,
        seed_count=10,
        split_seed=17,
    )
    other_manifest = SeedManifest.fresh_default(
        seed_start=30_300,
        seed_count=10,
        split_seed=17,
    )
    config = DQNConfig(
        total_steps=2,
        batch_size=1,
        replay_capacity=2,
        warmup_steps=1,
        checkpoint_every=1,
        eval_every=1,
        hidden_size=8,
        native_lanes=1,
    )
    model = DuelingWaypointDQN(hidden_size=config.hidden_size)
    target = DuelingWaypointDQN(hidden_size=config.hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity, WAYPOINT_OBSERVATION_SIZE)
    accumulator = NStepAccumulator(1, config.n_step, config.gamma, replay)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        _checkpoint_payload(
            model,
            target,
            optimizer,
            config,
            manifest,
            step=0,
            seed_cursor=0,
            best_inner=None,
            replay=replay,
            replay_accumulator=accumulator,
            rng=np.random.default_rng(1),
        ),
        checkpoint,
    )

    with pytest.raises(ValueError, match="manifest"):
        _load_checkpoint(
            checkpoint,
            DuelingWaypointDQN(hidden_size=config.hidden_size),
            DuelingWaypointDQN(hidden_size=config.hidden_size),
            torch.optim.AdamW(
                DuelingWaypointDQN(hidden_size=config.hidden_size).parameters(),
                lr=config.learning_rate,
            ),
            config,
            other_manifest,
            ReplayBuffer(config.replay_capacity, WAYPOINT_OBSERVATION_SIZE),
            NStepAccumulator(
                1,
                config.n_step,
                config.gamma,
                ReplayBuffer(config.replay_capacity, WAYPOINT_OBSERVATION_SIZE),
            ),
            np.random.default_rng(2),
        )
