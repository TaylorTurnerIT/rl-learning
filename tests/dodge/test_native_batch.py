from __future__ import annotations

import numpy as np
import pytest

from dodge.native.batch import NativeBatchEnvironment, NativeDodgeEnv

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


def test_single_lane_compatibility_adapter_preserves_observation_and_restart_contract(
) -> None:
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
