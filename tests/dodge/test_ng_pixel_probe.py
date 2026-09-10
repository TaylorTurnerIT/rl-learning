from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from dodge.dataset import ACTION_CHOICES
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from dodge.ng.pixel_probe import (
    PaletteSpatialClassifier,
    PixelProbeConfig,
    ScalarFastClassifier,
    _split_training_seeds,
)


def test_probe_models_accept_exact_indexed_pixel_stacks() -> None:
    pixels = torch.zeros((3, 4, 128, 128), dtype=torch.uint8)
    for model in (PaletteSpatialClassifier(4, 32), ScalarFastClassifier(4, 32)):
        assert model(pixels).shape == (3, len(ACTION_CHOICES))


def test_probe_split_uses_only_training_side_seeds() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    train, validation = _split_training_seeds(manifest, PixelProbeConfig())
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(manifest.training_seeds)
    assert (set(train) | set(validation)).isdisjoint(manifest.holdout_seeds)
    assert len(validation) == round(len(manifest.training_seeds) * 0.2)


def test_probe_collection_policy_is_explicit() -> None:
    PixelProbeConfig(collection_policy="oracle").validate()
    with pytest.raises(ValueError, match="collection policy"):
        replace(PixelProbeConfig(), collection_policy="unknown").validate()
    with pytest.raises(ValueError, match="random fraction"):
        replace(PixelProbeConfig(), oracle_random_fraction=1.1).validate()


def test_palette_spatial_model_distinguishes_palette_identity() -> None:
    model = PaletteSpatialClassifier(4, 16)
    zero = np.zeros((1, 4, 128, 128), dtype=np.uint8)
    one = zero.copy()
    one[:, :, 40:48, 40:48] = 1
    with torch.inference_mode():
        outputs = model(torch.from_numpy(np.concatenate((zero, one))))
    assert not torch.equal(outputs[0], outputs[1])
