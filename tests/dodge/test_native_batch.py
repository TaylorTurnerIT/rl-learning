from __future__ import annotations

import numpy as np
import pytest

from dodge.imitation.board import encode_board
from dodge.native.batch import (
    ML_OBSERVATION_SIZE,
    NativeBatchEnvironment,
    NativeDodgeEnv,
    NativeMlBatchResult,
    _raw_state_from_snapshot,
)
from dodge.neat.state import EntityState, PlayerState, RawState
from dodge.ng.dqn import encode_waypoint_observation
from dodge.ng.waypoint import WaypointGrid

pytest.importorskip("dodge_native")


def test_native_batch_returns_owned_arrays_with_documented_contract() -> None:
    environment = NativeBatchEnvironment(
        step_frames=4,
        full_state=True,
        pixels=True,
        board=True,
    )
    reset = environment.reset_batch([42, 13, 27])

    assert reset.frames.shape == (3,)
    np.testing.assert_array_equal(reset.lane_ids, [0, 1, 2])
    assert reset.frames.dtype == np.uint32
    assert reset.rewards.dtype == np.float32
    assert reset.done.dtype == np.bool_
    assert reset.pixels is not None
    assert reset.pixels.shape == (3, 128, 128)
    assert reset.pixels.dtype == np.uint8
    assert reset.board is not None
    assert reset.board.shape == (3, 19, 16, 16)
    assert reset.board.dtype == np.float32
    assert np.isfinite(reset.board).all()
    assert len(reset.snapshot_bytes) == 3
    assert all(isinstance(value, bytes) for value in reset.snapshot_bytes)
    assert all(state.frame == 13 for state in environment.observe_full_state())

    before = reset.pixels.copy()
    step = environment.step_batch([0, 1, 8])
    assert step.frames.tolist() == [17, 17, 17]
    assert step.rewards.tolist() == [4.0, 4.0, 4.0]
    assert not np.array_equal(before, step.pixels)
    before[...] = 0
    assert np.any(step.pixels != 0)

    environment.close()


def test_native_batch_can_omit_optional_buffers_without_changing_state_hashes() -> None:
    full = NativeBatchEnvironment(
        step_frames=4,
        full_state=True,
        pixels=True,
        board=True,
    )
    compact = NativeBatchEnvironment(
        step_frames=4,
        full_state=False,
        pixels=False,
        board=True,
    )
    full_reset = full.reset_batch([42, 13])
    compact_reset = compact.reset_batch([42, 13])

    np.testing.assert_array_equal(full_reset.state_hashes, compact_reset.state_hashes)
    np.testing.assert_array_equal(full_reset.pixel_hashes, compact_reset.pixel_hashes)
    assert compact_reset.pixels is None
    assert compact_reset.snapshot_bytes == (None, None)
    assert compact_reset.board is not None
    np.testing.assert_array_equal(full_reset.board, compact_reset.board)

    full.close()
    compact.close()


@pytest.mark.parametrize("spacing", [8, 16, 32])
def test_native_ml_observation_matches_python_reference_for_grid_resolutions(
    spacing: int,
) -> None:
    with NativeBatchEnvironment(
        step_frames=4,
        full_state=True,
        pixels=False,
        board=False,
        ml=True,
        ml_grid_spacing=spacing,
    ) as environment:
        result = environment.reset_batch([42, 30100])
        for actions in ([0, 8], [1, 4], [8, 0], [3, 5], [2, 6]):
            assert result.ml_observation is not None
            assert result.ml_observation.shape == (2, ML_OBSERVATION_SIZE)
            states = environment.observe_full_state()
            expected = np.stack(
                [
                    encode_waypoint_observation(
                        _raw_state_from_snapshot(state),
                        WaypointGrid(spacing),
                    )
                    for state in states
                ]
            )
            np.testing.assert_array_equal(result.ml_observation, expected)
            assert result.player_positions is not None
            expected_positions = np.asarray(
                [
                    [state.player[0] / 65_536, state.player[1] / 65_536]
                    for state in states
                ],
                dtype=np.float32,
            )
            np.testing.assert_array_equal(result.player_positions, expected_positions)
            result = environment.step_batch(actions)
            if bool(np.any(result.done)):
                break


def test_native_ml_path_omits_snapshots_without_changing_game_results() -> None:
    with (
        NativeBatchEnvironment(
            step_frames=4,
            full_state=True,
            pixels=True,
            board=True,
        ) as reference,
        NativeBatchEnvironment(
            step_frames=4,
            full_state=False,
            pixels=False,
            board=False,
            ml=True,
        ) as native_ml,
    ):
        reference_result = reference.reset_batch([42, 13])
        ml_result = native_ml.reset_batch([42, 13])
        for actions in ([0, 1], [2, 3], [8, 0], [4, 5]):
            np.testing.assert_array_equal(
                reference_result.frames,
                ml_result.frames,
            )
            np.testing.assert_array_equal(
                reference_result.rewards,
                ml_result.rewards,
            )
            np.testing.assert_array_equal(
                reference_result.done,
                ml_result.done,
            )
            np.testing.assert_array_equal(
                reference_result.state_hashes,
                ml_result.state_hashes,
            )
            np.testing.assert_array_equal(
                reference_result.pixel_hashes,
                ml_result.pixel_hashes,
            )
            assert ml_result.snapshot_bytes == (None, None)
            assert ml_result.ml_observation is not None
            assert ml_result.player_positions is not None
            ml_result = native_ml.step_batch(actions)
            reference_result = reference.step_batch(actions)
            if bool(np.any(reference_result.done)):
                break


def test_native_ml_fast_boundary_matches_full_batch_features_and_metadata() -> None:
    with (
        NativeBatchEnvironment(
            step_frames=4,
            full_state=True,
            pixels=False,
            board=False,
            ml=True,
        ) as reference,
        NativeBatchEnvironment(
            step_frames=4,
            full_state=False,
            pixels=False,
            board=False,
            ml=True,
        ) as fast,
    ):
        reference_result = reference.reset_batch([42, 13, 30100])
        fast_result = fast.reset_ml_batch([42, 13, 30100])
        assert isinstance(fast_result, NativeMlBatchResult)
        assert not hasattr(fast_result, "snapshot_bytes")
        assert fast_result.ml_observation.shape == (3, ML_OBSERVATION_SIZE)
        assert fast_result.ml_observation.dtype == np.float32
        assert fast_result.player_positions.shape == (3, 2)
        assert fast_result.player_positions.dtype == np.float32
        np.testing.assert_array_equal(fast_result.frames, reference_result.frames)
        np.testing.assert_array_equal(fast_result.seeds, reference_result.seeds)
        np.testing.assert_array_equal(
            fast_result.ml_observation, reference_result.ml_observation
        )
        np.testing.assert_array_equal(
            fast_result.player_positions, reference_result.player_positions
        )

        for step in range(90):
            actions = [
                step % 9,
                (step + 2) % 9,
                (step + 5) % 9,
            ]
            reference_result = reference.step_batch(actions)
            fast_result = fast.step_ml_batch(actions)
            np.testing.assert_array_equal(
                fast_result.lane_ids, reference_result.lane_ids
            )
            np.testing.assert_array_equal(fast_result.frames, reference_result.frames)
            np.testing.assert_array_equal(
                fast_result.frames_advanced, reference_result.frames_advanced
            )
            np.testing.assert_array_equal(fast_result.rewards, reference_result.rewards)
            np.testing.assert_array_equal(fast_result.done, reference_result.done)
            np.testing.assert_array_equal(fast_result.seeds, reference_result.seeds)
            np.testing.assert_array_equal(fast_result.modes, reference_result.modes)
            np.testing.assert_array_equal(
                fast_result.ml_observation, reference_result.ml_observation
            )
            np.testing.assert_array_equal(
                fast_result.player_positions, reference_result.player_positions
            )
            if bool(np.any(fast_result.done)):
                break


def test_native_ml_lane_reset_preserves_unselected_progress() -> None:
    with NativeBatchEnvironment(
        step_frames=4,
        full_state=False,
        pixels=False,
        board=False,
        ml=True,
    ) as environment:
        environment.reset_ml_batch([13, 27])
        environment.step_ml_batch([0, 1])
        reset = environment.reset_ml_lanes([1], [99])
        np.testing.assert_array_equal(reset.lane_ids, [1])
        np.testing.assert_array_equal(reset.frames, [13])
        np.testing.assert_array_equal(reset.seeds, [99])
        mixed = environment.step_ml_batch([0, 0])
        np.testing.assert_array_equal(mixed.lane_ids, [0, 1])
        np.testing.assert_array_equal(mixed.frames, [21, 17])
        np.testing.assert_array_equal(mixed.seeds, [13, 99])


def test_native_ai_startup_uses_up_waypoint_and_waits_for_visible_enemy() -> None:
    with (
        NativeBatchEnvironment(
            step_frames=4,
            full_state=False,
            pixels=False,
            board=False,
            ml=True,
            ml_grid_spacing=32,
        ) as ready,
        NativeBatchEnvironment(
            step_frames=4,
            full_state=False,
            pixels=False,
            board=False,
            ml=True,
            ml_grid_spacing=32,
        ) as started,
    ):
        ready_result = ready.reset_ml_batch([42])
        started_result = started.reset_ml_batch_with_startup([42])

    assert started_result.frames[0] > ready_result.frames[0]
    assert started_result.player_positions[0, 1] < 58.0
    assert started_result.ml_observation[0, 5] == 1.0


def test_enabling_ml_preserves_existing_optional_observations() -> None:
    with (
        NativeBatchEnvironment(
            step_frames=4,
            full_state=True,
            pixels=True,
            board=True,
        ) as reference,
        NativeBatchEnvironment(
            step_frames=4,
            full_state=True,
            pixels=True,
            board=True,
            ml=True,
        ) as augmented,
    ):
        reference_result = reference.reset_batch([42, 13])
        augmented_result = augmented.reset_batch([42, 13])
        for actions in ([0, 1], [2, 3], [8, 0]):
            np.testing.assert_array_equal(
                reference_result.state_hashes,
                augmented_result.state_hashes,
            )
            np.testing.assert_array_equal(
                reference_result.pixel_hashes,
                augmented_result.pixel_hashes,
            )
            np.testing.assert_array_equal(
                reference_result.pixels, augmented_result.pixels
            )
            np.testing.assert_array_equal(
                reference_result.board, augmented_result.board
            )
            assert reference_result.snapshot_bytes == augmented_result.snapshot_bytes
            assert augmented_result.ml_observation is not None
            reference_result = reference.step_batch(actions)
            augmented_result = augmented.step_batch(actions)
            if bool(np.any(reference_result.done)):
                break


def test_native_ml_requires_positive_grid_spacing() -> None:
    with pytest.raises(ValueError, match="positive"):
        NativeBatchEnvironment(ml=True, ml_grid_spacing=0)


def test_board_full_preserves_offscreen_hazards_and_legacy_board_does_not() -> None:
    legacy = NativeBatchEnvironment(
        step_frames=4,
        full_state=True,
        board=True,
    )
    board_full = NativeBatchEnvironment(
        step_frames=4,
        full_state=True,
        board=True,
        include_offscreen_board=True,
    )
    legacy.reset_batch([30100])
    board_full.reset_batch([30100])

    found_offscreen = False
    for _ in range(20):
        legacy_step = legacy.step_batch([0])
        board_full.step_batch([0])
        state = board_full.observe_full_state()[0]
        found_offscreen = any(
            enemy.x + enemy.size / 2 < 0
            or enemy.x - enemy.size / 2 >= 128
            or enemy.y + enemy.size / 2 < 0
            or enemy.y - enemy.size / 2 >= 128
            for enemy in state.enemies
        )
        if found_offscreen:
            break
        if bool(legacy_step.done[0]):
            break

    assert found_offscreen
    assert legacy.last_result.board is not None
    assert board_full.last_result.board is not None
    legacy_enemy_cells = legacy.last_result.board[0, 5].sum()
    full_enemy_cells = board_full.last_result.board[0, 5].sum()
    assert full_enemy_cells > legacy_enemy_cells
    assert not np.array_equal(legacy.last_result.board, board_full.last_result.board)
    assert board_full.include_offscreen_board is True

    legacy.close()
    board_full.close()


def test_board_full_matches_python_reference_encoder_for_native_state() -> None:
    environment = NativeBatchEnvironment(
        step_frames=4,
        full_state=True,
        board=True,
        include_offscreen_board=True,
    )
    environment.reset_batch([30100])
    for _ in range(11):
        environment.step_batch([0])

    snapshot = environment.observe_full_state()[0]
    assert all(enemy.personality == 0 for enemy in snapshot.enemies)
    fixed_scale = 1 / 65_536
    player_x, player_y, player_vx, player_vy, player_size = snapshot.player
    reference = RawState(
        frame=snapshot.frame,
        player=PlayerState(
            player_x * fixed_scale,
            player_y * fixed_scale,
            player_vx * fixed_scale,
            player_vy * fixed_scale,
            player_size * fixed_scale,
        ),
        enemies=tuple(
            EntityState(
                enemy.x * fixed_scale,
                enemy.y * fixed_scale,
                enemy.vx * fixed_scale,
                enemy.vy * fixed_scale,
                enemy.size * fixed_scale,
                enemy.size * fixed_scale,
                "enemy",
                0,
            )
            for enemy in snapshot.enemies
        ),
        aoes=(),
    )

    assert environment.last_result.board is not None
    np.testing.assert_array_equal(
        environment.last_result.board[0],
        encode_board(reference, include_offscreen=True),
    )
    environment.close()


def test_board_full_coords_preserves_canonical_positions_and_matches_reference() -> (
    None
):
    environment = NativeBatchEnvironment(
        step_frames=4,
        full_state=True,
        board=True,
        include_offscreen_board=True,
        preserve_offscreen_coordinates=True,
    )
    environment.reset_batch([30100])
    for _ in range(15):
        environment.step_batch([0])

    snapshot = environment.observe_full_state()[0]
    fixed_scale = 1 / 65_536
    player_x, player_y, player_vx, player_vy, player_size = snapshot.player
    reference = RawState(
        frame=snapshot.frame,
        player=PlayerState(
            player_x * fixed_scale,
            player_y * fixed_scale,
            player_vx * fixed_scale,
            player_vy * fixed_scale,
            player_size * fixed_scale,
        ),
        enemies=tuple(
            EntityState(
                enemy.x * fixed_scale,
                enemy.y * fixed_scale,
                enemy.vx * fixed_scale,
                enemy.vy * fixed_scale,
                enemy.size * fixed_scale,
                enemy.size * fixed_scale,
                "enemy",
                0,
            )
            for enemy in snapshot.enemies
        ),
        aoes=(),
    )

    assert environment.last_result.board is not None
    assert environment.last_result.board.shape == (1, 23, 16, 16)
    np.testing.assert_array_equal(
        environment.last_result.board[0],
        encode_board(
            reference,
            include_offscreen=True,
            preserve_coordinates=True,
        ),
    )
    assert np.any(environment.last_result.board[0, 19:21] != 0)
    assert environment.preserve_offscreen_coordinates is True
    environment.close()


def test_single_lane_compatibility_adapter_preserves_restart_contract() -> None:
    environment = NativeDodgeEnv(step_frames=4)
    observation = environment.reset(seed=42)
    assert observation.raw_state.frame == 13
    assert observation.raw_state.player.x == 64.0
    assert len(observation.projected.values) == 197

    transition = environment.step("neutral")
    assert transition.observation.raw_state.frame == 17
    assert transition.reward == 4.0
    assert transition.done is False

    environment.close()
    restarted = environment.reset(seed=42)
    assert restarted.raw_state.frame == 13
    assert restarted.raw_state.player.x == 64.0
    environment.close()


def test_serial_and_parallel_python_batches_are_lane_identical() -> None:
    kwargs = {
        "step_frames": 4,
        "full_state": True,
        "pixels": True,
        "board": True,
        "include_offscreen_board": True,
        "preserve_offscreen_coordinates": True,
    }
    serial = NativeBatchEnvironment(execution="serial", **kwargs)
    parallel = NativeBatchEnvironment(execution="parallel", **kwargs)
    seeds = [13, 27, 58, 101]
    serial_reset = serial.reset_batch(seeds)
    parallel_reset = parallel.reset_batch(seeds)
    np.testing.assert_array_equal(serial_reset.lane_ids, parallel_reset.lane_ids)

    for left, right in (
        (serial_reset.frames, parallel_reset.frames),
        (serial_reset.state_hashes, parallel_reset.state_hashes),
        (serial_reset.pixel_hashes, parallel_reset.pixel_hashes),
        (serial_reset.board, parallel_reset.board),
        (serial_reset.pixels, parallel_reset.pixels),
    ):
        np.testing.assert_array_equal(left, right)
    assert serial_reset.snapshot_bytes == parallel_reset.snapshot_bytes

    for step in range(90):
        actions = [(step + lane * 3) % 9 for lane in range(len(seeds))]
        serial_step = serial.step_batch(actions)
        parallel_step = parallel.step_batch(actions)
        np.testing.assert_array_equal(serial_step.lane_ids, parallel_step.lane_ids)
        np.testing.assert_array_equal(serial_step.frames, parallel_step.frames)
        np.testing.assert_array_equal(serial_step.rewards, parallel_step.rewards)
        np.testing.assert_array_equal(serial_step.done, parallel_step.done)
        np.testing.assert_array_equal(
            serial_step.state_hashes, parallel_step.state_hashes
        )
        np.testing.assert_array_equal(
            serial_step.pixel_hashes, parallel_step.pixel_hashes
        )
        np.testing.assert_array_equal(serial_step.board, parallel_step.board)
        np.testing.assert_array_equal(serial_step.pixels, parallel_step.pixels)
        assert serial_step.snapshot_bytes == parallel_step.snapshot_bytes
        if bool(np.any(serial_step.done)):
            break

    serial.close()
    parallel.close()


def test_per_lane_reset_preserves_other_lane_state() -> None:
    environment = NativeBatchEnvironment(step_frames=4, board=True)
    environment.reset_batch([13, 27])
    environment.step_batch([0, 1])

    reset = environment.reset_lanes([1], [99])
    np.testing.assert_array_equal(reset.lane_ids, [1])
    np.testing.assert_array_equal(reset.frames, [13])
    np.testing.assert_array_equal(reset.seeds, [99])

    mixed = environment.step_batch([0, 0])
    np.testing.assert_array_equal(mixed.lane_ids, [0, 1])
    np.testing.assert_array_equal(mixed.frames, [21, 17])
    np.testing.assert_array_equal(mixed.seeds, [13, 99])
    environment.close()


def test_counterfactual_scores_are_deterministic_and_non_mutating() -> None:
    environment = NativeBatchEnvironment(
        step_frames=4,
        execution="parallel",
        full_state=True,
        board=True,
    )
    environment.reset_batch([42, 13])
    before = environment.step_batch([0, 1])
    snapshots = [value for value in before.snapshot_bytes if value is not None]

    first = environment.score_actions(snapshots, lookahead_steps=8)
    second = environment.score_actions(snapshots, lookahead_steps=8)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 9)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()

    after = environment.step_batch([2, 3])
    control = NativeBatchEnvironment(step_frames=4, full_state=True, board=True)
    control.reset_batch([42, 13])
    control.step_batch([0, 1])
    expected = control.step_batch([2, 3])
    np.testing.assert_array_equal(after.state_hashes, expected.state_hashes)
    np.testing.assert_array_equal(after.pixel_hashes, expected.pixel_hashes)
    environment.close()
    control.close()


def test_counterfactual_scores_validate_inputs() -> None:
    environment = NativeBatchEnvironment(step_frames=4, full_state=True, board=True)
    environment.reset_batch([42])
    snapshot = environment.last_result.snapshot_bytes[0]
    assert snapshot is not None
    with pytest.raises(ValueError, match="positive"):
        environment.score_actions([snapshot], lookahead_steps=0)
    with pytest.raises(ValueError, match="non-empty"):
        environment.score_actions([], lookahead_steps=8)
    with pytest.raises(ValueError, match="bytes"):
        environment.score_actions(["not-bytes"], lookahead_steps=8)  # type: ignore[list-item]
    environment.close()
