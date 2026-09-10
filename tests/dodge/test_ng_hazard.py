from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from dodge.native.batch import NativeBatchEnvironment
from dodge.ng.dqn import (
    DQNConfig,
    NStepAccumulator,
    ReplayBuffer,
    _checkpoint_config_matches,
    _checkpoint_contract,
    _checkpoint_contract_matches,
    _collect_macro_transition,
    _native_observation_state,
    _reset_ml_batch,
    model_for_config,
    observation_size_for_config,
    waypoint_controller_for_config,
)
from dodge.ng.hazard import (
    HAZARD_PLAYER_PRESENCE,
    HazardObservationSpec,
    decode_hazard_observation,
    hazard_observation_size,
    spawn_corner_bits,
    validate_native_hazard_result,
)
from dodge.ng.waypoint import WaypointGrid


def test_centered_waypoint_grid_scales_points_to_cell_centers() -> None:
    grid = WaypointGrid.centered(2)

    assert grid.axis_points == pytest.approx((32.75, 94.25))
    assert grid.shape == (2, 2)
    assert grid.point((0, 0)) == pytest.approx((32.75, 32.75))
    assert grid.point((1, 1)) == pytest.approx((94.25, 94.25))
    assert grid.nearest_cell(64.0, 64.0) == (1, 1)


def test_hazard_config_uses_fixed_width_centered_observation() -> None:
    config = DQNConfig(
        observation_mode="hazard",
        hazard_grid_size=4,
        prediction_horizon_frames=32,
        spawn_halo_radius=2,
        steering_tolerance=2.0,
    )

    config.validate()
    assert observation_size_for_config(config) == hazard_observation_size(4)
    assert model_for_config(config).input_size == hazard_observation_size(4)
    assert waypoint_controller_for_config(config).grid.axis_points == pytest.approx(
        (17.375, 48.125, 78.875, 109.625)
    )


def test_hazard_config_rejects_tolerance_larger_than_centered_cell_window() -> None:
    config = replace(
        DQNConfig(observation_mode="hazard", hazard_grid_size=32),
        steering_tolerance=2.0,
    )

    with pytest.raises(ValueError, match="below half grid spacing"):
        config.validate()


def test_hazard_startup_uses_centered_grid_geometry() -> None:
    with NativeBatchEnvironment(
        step_frames=4,
        full_state=False,
        pixels=False,
        board=False,
        ml=True,
    ) as environment:
        result = environment.reset_ml_batch_with_centered_startup([42], 1)

    assert result.frames[0] > 13
    assert result.player_positions[0, 1] > 58.0


def test_hazard_checkpoint_contract_freezes_field_and_waypoint_schema() -> None:
    config = DQNConfig(
        observation_mode="hazard",
        hazard_grid_size=4,
        prediction_horizon_frames=24,
        spawn_halo_radius=2,
    )
    config.validate()

    contract = _checkpoint_contract(config)

    assert contract["observation_mode"] == "hazard"
    assert contract["grid_size"] == 4
    assert contract["grid_geometry"] == "centered_equal_cells"
    assert contract["hazard"]["prediction_horizon_frames"] == 24  # type: ignore[index]
    assert contract["hazard"]["spawn_halo_radius"] == 2  # type: ignore[index]
    assert contract["target"]["algorithm"] == "double_dqn"  # type: ignore[index]
    assert len(contract["actions"]) == 9  # type: ignore[arg-type]
    assert _checkpoint_contract_matches(contract, config)
    assert _checkpoint_config_matches(config.to_json(), config)


def test_hazard_ddqn_path_chooses_waypoint_actions_from_reference_field() -> None:
    config = DQNConfig(
        total_steps=1,
        batch_size=1,
        replay_capacity=4,
        n_step=1,
        warmup_steps=1,
        hidden_size=8,
        native_lanes=2,
        hold_decisions=1,
        max_episode_steps=1,
        observation_mode="hazard",
        hazard_grid_size=4,
    )
    config.validate()
    with NativeBatchEnvironment(
        step_frames=config.step_frames,
        execution="serial",
        full_state=False,
        pixels=False,
        board=False,
        ml=True,
    ) as environment:
        reset = _reset_ml_batch(
            environment,
            np.asarray([13, 27], dtype=np.uint32),
            config,
        )
        observations, positions = _native_observation_state(reset, config)
        replay = ReplayBuffer(
            config.replay_capacity, observation_size_for_config(config)
        )
        accumulator = NStepAccumulator(2, config.n_step, config.gamma, replay)
        episode_steps = np.zeros(2, dtype=np.int64)
        episode_seeds = np.asarray([13, 27], dtype=np.uint32)
        lives_remaining = np.ones(2, dtype=np.int64)
        result = _collect_macro_transition(
            environment,
            observations.copy(),
            positions.copy(),
            waypoint_controller_for_config(config),
            model_for_config(config),
            config,
            episode_steps,
            episode_seeds,
            lives_remaining,
            (13, 27),
            0,
            accumulator,
            np.random.default_rng(7),
            torch.device("cpu"),
            0,
        )

    assert result[0].shape == (2, observation_size_for_config(config))
    assert result[1].shape == (2, 2)
    assert replay.size == 2


pytest.importorskip("dodge_native")


@pytest.mark.parametrize("execution", ["serial", "parallel"])
def test_native_hazard_result_is_finite_and_contains_four_spawn_bits(
    execution: str,
) -> None:
    with NativeBatchEnvironment(
        step_frames=4,
        execution=execution,
        full_state=False,
        pixels=False,
        board=False,
        ml=True,
    ) as environment:
        environment.reset_ml_batch([13, 27])
        result = environment.hazard_observations(
            4,
            prediction_horizon_frames=32,
            spawn_halo_radius=1,
        )

    spec = HazardObservationSpec(4, 32, 1)
    observations, positions = validate_native_hazard_result(result, spec)
    assert observations.shape == (2, hazard_observation_size(4))
    assert positions.shape == (2, 2)
    assert result.ttc_reference.shape == (2, 16)
    assert np.isinf(result.ttc_reference).all()
    assert np.isfinite(observations).all()
    view = decode_hazard_observation(observations[0], spec)
    np.testing.assert_array_equal(
        spawn_corner_bits(view),
        np.asarray(
            [
                [1, 0, 0, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [4, 0, 0, 8],
            ],
            dtype=np.uint8,
        ),
    )
    assert np.count_nonzero(view.fields[HAZARD_PLAYER_PRESENCE]) == 1


def test_native_hazard_serial_parallel_and_live_lane_trace_match() -> None:
    options = dict(
        step_frames=4,
        full_state=False,
        pixels=False,
        board=False,
        ml=True,
    )
    with (
        NativeBatchEnvironment(execution="serial", **options) as serial,
        NativeBatchEnvironment(execution="parallel", **options) as parallel,
        NativeBatchEnvironment(execution="serial", **options) as control,
    ):
        seeds = np.asarray([13, 27], dtype=np.uint32)
        serial.reset_ml_batch(seeds)
        parallel.reset_ml_batch(seeds)
        control.reset_ml_batch(seeds)
        serial_field = serial.hazard_observations(4)
        parallel_field = parallel.hazard_observations(4)
        np.testing.assert_array_equal(
            serial_field.hazard_observation,
            parallel_field.hazard_observation,
        )
        np.testing.assert_array_equal(serial_field.frames, parallel_field.frames)
        np.testing.assert_array_equal(
            serial_field.player_positions, parallel_field.player_positions
        )

        actual = serial.step_ml_batch([0, 1])
        expected = control.step_ml_batch([0, 1])
        np.testing.assert_array_equal(actual.ml_observation, expected.ml_observation)
        np.testing.assert_array_equal(
            actual.player_positions, expected.player_positions
        )
