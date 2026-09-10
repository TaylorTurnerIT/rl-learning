from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from dodge.dataset import ACTION_CHOICES
from dodge.native.batch import NativeBatchEnvironment
from dodge.ng.pixel_dqn import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    DuelingPixelDQN,
    PixelDQNConfig,
    PixelNStepAccumulator,
    PixelReplayBuffer,
    _checkpoint_contract,
    _validate_pixel_result,
)

pytest.importorskip("dodge_native")


def _stack(value: int, stack_size: int = 2) -> np.ndarray:
    return np.full(
        (1, stack_size, FRAME_HEIGHT, FRAME_WIDTH),
        value,
        dtype=np.uint8,
    )


def test_pixel_dqn_accepts_only_indexed_pixel_tensors() -> None:
    model = DuelingPixelDQN(stack_size=2, hidden_size=16, architecture="fast")
    observations = torch.zeros((3, 2, FRAME_HEIGHT, FRAME_WIDTH), dtype=torch.uint8)

    values = model(observations)

    assert values.shape == (3, len(ACTION_CHOICES))
    assert torch.isfinite(values).all()
    with pytest.raises(ValueError, match="uint8"):
        model(observations.float())
    with pytest.raises(ValueError, match="pixel observations must have shape"):
        model(torch.zeros((3, 1, FRAME_HEIGHT, FRAME_WIDTH), dtype=torch.uint8))


def test_pixel_replay_frame_ring_and_n_step_boundary(tmp_path) -> None:
    replay = PixelReplayBuffer(
        capacity=8,
        stack_size=2,
        lane_count=1,
        run_directory=tmp_path,
    )
    accumulator = PixelNStepAccumulator(1, 3, 0.9, replay)
    references = [replay.store_stacks(_stack(value))[0] for value in (1, 2, 3)]

    accumulator.append(0, references[0], 0, (1, 1), 1.0, references[1], False, False)
    accumulator.append(0, references[1], 1, (1, 2), 2.0, references[2], True, False)

    assert replay.size == 2
    assert replay.rewards[0] == pytest.approx(1.0 + 0.9 * 2.0)
    assert replay.rewards[1] == pytest.approx(2.0)
    assert np.all(replay.discounts[:2] == 0.0)
    assert replay.terminated[:2].tolist() == [True, True]
    sample = replay.sample(2, np.random.default_rng(7))
    assert sample.observations.shape == (2, 2, FRAME_HEIGHT, FRAME_WIDTH)
    assert sample.observations.dtype == np.uint8
    replay.close()


def test_pixel_checkpoint_contract_is_pixel_only_and_lives_are_single_ablation() -> (
    None
):
    one_life = PixelDQNConfig()
    three_lives = replace(one_life, training_lives=3)

    assert _checkpoint_contract(one_life)["observation_source"] == (
        "native_indexed_pixels_only"
    )
    assert one_life.to_json() | {"training_lives": 3} == three_lives.to_json()
    left = one_life.to_json()
    right = three_lives.to_json()
    left.pop("training_lives")
    right.pop("training_lives")
    assert left == right


def test_native_pixel_result_has_exact_raster_and_no_snapshot_payload() -> None:
    with NativeBatchEnvironment(
        step_frames=4,
        execution="serial",
        full_state=False,
        pixels=True,
        board=False,
        ml=True,
        ml_grid_spacing=32,
    ) as environment:
        result = environment.reset_batch([30_200, 30_201])
        pixels, positions = _validate_pixel_result(result)

    assert pixels.shape == (2, FRAME_HEIGHT, FRAME_WIDTH)
    assert pixels.dtype == np.uint8
    assert int(pixels.max()) <= 15
    assert positions.shape == (2, 2)
    assert result.snapshot_bytes == (None, None)
