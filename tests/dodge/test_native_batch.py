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
