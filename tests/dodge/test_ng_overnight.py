from __future__ import annotations

from pathlib import Path

from dodge.ng.manifest import SeedManifest
from dodge.ng.overnight import (
    HPO_PARAMETERS,
    HPO_SOURCE,
    HPO_TRIAL,
    OvernightConfig,
    _metadata,
)


def test_overnight_config_inherits_hpo_trial_and_declares_large_batch() -> None:
    config = OvernightConfig(
        manifest_path=Path("manifest.json"),
        run_directory=Path("run"),
        total_steps=1_000_000,
        batch_size=1_024,
        training_lives=3,
        life_loss_penalty=-64.0,
    )
    dqn = config.dqn_config()

    assert dqn.learning_rate == HPO_PARAMETERS["learning_rate"]
    assert dqn.weight_decay == HPO_PARAMETERS["weight_decay"]
    assert dqn.target_update_interval == HPO_PARAMETERS["target_update_interval"]
    assert dqn.n_step == HPO_PARAMETERS["n_step"]
    assert dqn.epsilon_decay_steps == HPO_PARAMETERS["epsilon_decay_steps"]
    assert dqn.batch_size == 1_024
    assert dqn.training_lives == 3
    assert dqn.life_loss_penalty == -64.0
    dqn.validate()


def test_overnight_metadata_records_hpo_and_training_only_mode() -> None:
    manifest = SeedManifest.fresh_default(
        seed_start=30_200,
        seed_count=10,
        split_seed=17,
    )
    config = OvernightConfig(
        run_directory=Path("run"),
        total_steps=100,
        batch_size=1_024,
        training_lives=3,
        life_loss_penalty=-64.0,
        native_lanes=2,
    )
    metadata = _metadata(config, manifest)

    assert metadata["hpo_source"] == HPO_SOURCE
    assert metadata["hpo_trial"] == HPO_TRIAL
    assert metadata["inherited_hpo_parameters"] == HPO_PARAMETERS
    assert metadata["manifest_sha256"] == manifest.sha256
    assert metadata["evaluation_mode"] == {
        "training_split": False,
        "holdout_split": False,
        "inner_training_seeds": True,
    }
    assert metadata["config"]["batch_size"] == 1_024
    assert metadata["config"]["training_lives"] == 3
