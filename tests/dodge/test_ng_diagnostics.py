from __future__ import annotations

import json
from pathlib import Path

import pytest

from dodge.dataset import ACTION_CHOICES
from dodge.ng.diagnostics import build_action_diagnostic
from dodge.ng.manifest import SeedManifest


def test_fixed_action_diagnostic_writes_all_action_controls(tmp_path: Path) -> None:
    pytest.importorskip("dodge_native")
    output_directory = tmp_path / "controls"

    diagnostic = build_action_diagnostic(
        output_directory,
        SeedManifest.fresh_default(),
        max_episode_steps=1,
    )

    actions = diagnostic["actions"]
    assert isinstance(actions, list)
    assert [item["action"] for item in actions] == list(ACTION_CHOICES)
    assert all(item["training"]["count"] == 70 for item in actions)
    assert all(item["holdout"]["count"] == 30 for item in actions)
    assert (
        json.loads((output_directory / "action-controls.json").read_text())[
            "manifest_sha256"
        ]
        == SeedManifest.fresh_default().sha256
    )
    assert (output_directory / "ACTION_CONTROLS.md").is_file()
