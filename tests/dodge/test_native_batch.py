from __future__ import annotations

import numpy as np
import pytest

from dodge.imitation.board import encode_board
from dodge.native.batch import NativeBatchEnvironment, NativeDodgeEnv
from dodge.neat.state import EntityState, PlayerState, RawState

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
