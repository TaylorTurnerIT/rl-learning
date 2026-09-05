from __future__ import annotations

import signal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import NativeBatchEnvironment, _decode_snapshot
from dodge.native.differential import NativeDifferentialError
from dodge.neat.state import PlayerState, RawState
from dodge.ng.dqn import (
    RELEVANCE_GATE_FRAMES,
    WAYPOINT_OBSERVATION_SIZE,
    DQNConfig,
    DuelingWaypointDQN,
    NStepAccumulator,
    ReplayBuffer,
    _checkpoint_config_matches,
    _checkpoint_contract,
    _checkpoint_payload,
    _collect_macro_transition,
    _consume_training_life,
    _epsilon,
    _install_stop_signal_handlers,
    _load_checkpoint,
    _native_ml_state,
    _player_position,
    _restore_stop_signal_handlers,
    encode_waypoint_observation,
)
from dodge.ng.manifest import SeedManifest
from dodge.ng.waypoint import WaypointController, WaypointGrid

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


def test_fast_player_position_reader_matches_full_snapshot_decoder() -> None:
    with NativeBatchEnvironment(
        step_frames=4,
        execution="serial",
        full_state=True,
        board=False,
    ) as environment:
        result = environment.reset_batch([30_200])
        snapshot = result.snapshot_bytes[0]
        assert snapshot is not None
        full_player = _decode_snapshot(snapshot).player
        fast_position = _player_position(snapshot)

    assert fast_position == pytest.approx(
        (full_player[0] / 65_536, full_player[1] / 65_536)
    )
    with pytest.raises(NativeDifferentialError, match="too short"):
        _player_position(b"DGSN")


def test_native_ml_result_satisfies_waypoint_dqn_contract_without_snapshots() -> None:
    with NativeBatchEnvironment(
        step_frames=4,
        full_state=False,
        pixels=False,
        board=False,
        ml=True,
        ml_grid_spacing=32,
    ) as environment:
        result = environment.reset_batch([30_200, 30_201])
        observations, positions = _native_ml_state(result)

    assert observations.shape == (2, WAYPOINT_OBSERVATION_SIZE)
    assert positions.shape == (2, 2)
    assert result.snapshot_bytes == (None, None)


def test_dqn_relevance_gate_is_not_the_episode_safety_horizon() -> None:
    config = DQNConfig()

    assert config.max_episode_steps * config.step_frames > RELEVANCE_GATE_FRAMES


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


def test_dqn_controller_controls_are_validated_and_provenanced() -> None:
    config = DQNConfig(
        grid_spacing=24,
        hold_decisions=12,
        steering_tolerance=6.0,
        arrival_latching=True,
        ban_corner_nodes=True,
        corner_node_penalty=-8.0,
    )

    config.validate()
    contract = _checkpoint_contract(config)

    assert contract["grid_spacing"] == 24
    assert contract["corner_nodes"] == "banned"
    assert contract["controller"] == {
        "tolerance": 6.0,
        "arrival_latching": True,
        "corner_node_penalty": -8.0,
        "steering": "sign(target_position-current_position)",
    }
    assert contract["cadence"] == {
        "step_frames": 4,
        "hold_decisions": 12,
        "decision_interval": 12,
    }


def test_epsilon_uses_explicit_decay_schedule_when_configured() -> None:
    config = DQNConfig(epsilon_decay_steps=20, epsilon_final=0.2)

    assert _epsilon(config, 0) == pytest.approx(1.0)
    assert _epsilon(config, 10) == pytest.approx(0.6)
    assert _epsilon(config, 20) == pytest.approx(0.2)
    assert _epsilon(config, 100) == pytest.approx(0.2)


def test_training_life_consumption_resets_life_steps_and_marks_final_loss() -> None:
    lives = np.asarray([3], dtype=np.int64)
    episode_steps = np.asarray([17], dtype=np.int64)

    assert not _consume_training_life(lives, episode_steps, 0, 3)
    assert lives.tolist() == [2]
    assert episode_steps.tolist() == [0]

    assert _consume_training_life(lives, episode_steps, 0, 3) is False
    assert lives.tolist() == [1]
    assert _consume_training_life(lives, episode_steps, 0, 3) is True
    assert lives.tolist() == [3]


def test_training_stop_signal_handler_sets_flag_and_restores() -> None:
    requested, previous = _install_stop_signal_handlers()
    try:
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(int(signal.SIGTERM), None)
        assert requested[0]
    finally:
        _restore_stop_signal_handlers(previous)


def test_training_lives_penalize_nonfinal_death_and_advance_seed_on_final_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnvironment:
        def __init__(self) -> None:
            self.reset_calls: list[tuple[np.ndarray, np.ndarray]] = []

        def step_ml_batch(self, _actions: np.ndarray) -> SimpleNamespace:
            return SimpleNamespace(
                lane_count=1,
                ml_observation=np.full(
                    (1, WAYPOINT_OBSERVATION_SIZE),
                    2.0,
                    dtype=np.float32,
                ),
                player_positions=np.full((1, 2), 66.0, dtype=np.float32),
                rewards=np.asarray([4.0], dtype=np.float32),
                done=np.asarray([True], dtype=bool),
            )

        def reset_ml_lanes_with_startup(
            self,
            lanes: np.ndarray,
            seeds: np.ndarray,
        ) -> SimpleNamespace:
            self.reset_calls.append((lanes.copy(), seeds.copy()))
            return SimpleNamespace(
                lane_count=1,
                ml_observation=np.full(
                    (1, WAYPOINT_OBSERVATION_SIZE),
                    9.0,
                    dtype=np.float32,
                ),
                player_positions=np.full((1, 2), 66.0, dtype=np.float32),
                rewards=np.asarray([0.0], dtype=np.float32),
                done=np.asarray([False], dtype=bool),
            )

    monkeypatch.setattr(
        "dodge.ng.dqn._choose_actions",
        lambda *_args: np.asarray([0], dtype=np.uint8),
    )
    config = DQNConfig(
        total_steps=1,
        batch_size=1,
        replay_capacity=4,
        n_step=1,
        warmup_steps=1,
        target_update_interval=1,
        hidden_size=8,
        hold_decisions=1,
        native_lanes=1,
        training_lives=3,
        life_loss_penalty=-64.0,
    )
    environment = FakeEnvironment()
    replay = ReplayBuffer(config.replay_capacity, WAYPOINT_OBSERVATION_SIZE)
    accumulator = NStepAccumulator(1, config.n_step, config.gamma, replay)
    model = DuelingWaypointDQN(hidden_size=config.hidden_size)
    observations = np.zeros((1, WAYPOINT_OBSERVATION_SIZE), dtype=np.float32)
    positions = np.full((1, 2), 66.0, dtype=np.float32)
    episode_steps = np.asarray([17], dtype=np.int64)
    episode_seeds = np.asarray([123], dtype=np.uint32)
    lives_remaining = np.asarray([3], dtype=np.int64)

    result = _collect_macro_transition(
        environment,
        observations,
        positions,
        WaypointController(WaypointGrid(config.grid_spacing)),
        model,
        config,
        episode_steps,
        episode_seeds,
        lives_remaining,
        (456, 789),
        0,
        accumulator,
        np.random.default_rng(7),
        torch.device("cpu"),
        0,
    )

    assert result[2] == 0
    assert lives_remaining.tolist() == [2]
    assert episode_seeds.tolist() == [123]
    assert environment.reset_calls[0][1].tolist() == [123]
    assert replay.size == 1
    assert replay.rewards[0] == pytest.approx(-60.0)
    assert not bool(replay.terminated[0])
    assert replay.next_observations[0, 0] == pytest.approx(9.0)
    assert result[4]["life_loss_count"] == 1.0
    assert result[4]["final_death_count"] == 0.0

    lives_remaining[0] = 1
    result = _collect_macro_transition(
        environment,
        result[0],
        result[1],
        WaypointController(WaypointGrid(config.grid_spacing)),
        model,
        config,
        episode_steps,
        episode_seeds,
        lives_remaining,
        (456, 789),
        result[2],
        accumulator,
        np.random.default_rng(8),
        torch.device("cpu"),
        1,
    )

    assert result[2] == 1
    assert lives_remaining.tolist() == [3]
    assert episode_seeds.tolist() == [456]
    assert environment.reset_calls[1][1].tolist() == [456]
    assert replay.size == 2
    assert replay.rewards[1] == pytest.approx(-60.0)
    assert bool(replay.terminated[1])
    assert replay.next_observations[1, 0] == pytest.approx(2.0)
    assert result[4]["life_loss_count"] == 1.0
    assert result[4]["final_death_count"] == 1.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("weight_decay", -1.0, "weight decay"),
        ("training_lives", 0, "training lives"),
        ("life_loss_penalty", 1.0, "life-loss penalty"),
        ("epsilon_decay_steps", -1, "epsilon decay"),
        ("epsilon_final", 1.0, "final epsilon"),
    ],
)
def test_dqn_config_rejects_invalid_regularization_or_epsilon(
    field: str,
    value: float | int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DQNConfig(**{field: value}).validate()


def test_old_checkpoint_config_is_compatible_with_new_optional_fields() -> None:
    config = DQNConfig(total_steps=20, reset_mode="legacy")
    saved = config.to_json()
    saved["total_steps"] = 10
    saved.pop("weight_decay")
    saved.pop("epsilon_decay_steps")
    saved.pop("epsilon_final")
    saved.pop("training_lives")
    saved.pop("life_loss_penalty")
    saved.pop("reset_mode")

    assert _checkpoint_config_matches(saved, config)


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
