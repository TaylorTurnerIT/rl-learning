from __future__ import annotations

import numpy as np
import pytest

from dodge.ng.dagger import DaggerConfig, collect_learner_dataset
from dodge.ng.manifest import SeedManifest
from dodge.rl.ppo import DodgeActorCriticCNN

pytest.importorskip("dodge_native")


def test_dagger_collects_learner_visited_training_states() -> None:
    manifest = SeedManifest.fresh_default(
        seed_start=30_200, seed_count=10, split_seed=17
    )
    config = DaggerConfig(
        states_per_seed=1,
        lookahead_steps=2,
        native_lanes=2,
        max_collector_steps=100,
        learner_seed=99,
    )

    dataset = collect_learner_dataset(config, manifest, DodgeActorCriticCNN())

    assert dataset.count == len(manifest.training_seeds)
    assert set(dataset.seeds.tolist()) <= set(manifest.training_seeds)
    assert not set(dataset.seeds.tolist()) & set(manifest.holdout_seeds)
    assert dataset.boards.shape == (dataset.count, 19, 16, 16)
    assert dataset.scores.shape == (dataset.count, 9)
    assert np.isfinite(dataset.scores).all()
    assert dataset.metadata["collection_policy"] == "dagger"
    assert dataset.metadata["score_cache"]["entries"] == dataset.count  # type: ignore[index]
