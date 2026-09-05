from __future__ import annotations

import json
from pathlib import Path

from dodge.ng.dashboard import DASHBOARD_PAGE, RunInspector
from dodge.ng.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from dodge.ng.replay import (
    REPRESENTATIVE_ROLES,
    DQNConfig,
    record_representative_replays,
    select_representative_replays,
)


def test_representative_selection_is_deterministic_and_tie_breaks_by_seed() -> None:
    selected = select_representative_replays(
        {
            "seeds": [30, 20, 10, 40],
            "survival_frames": [100, 200, 200, 400],
            "summary": {"mean_survival_frames": 250.0},
        }
    )

    assert selected["best"]["seed"] == 40
    assert selected["best"]["survival_frames"] == 400
    assert selected["mean"]["seed"] == 10
    assert selected["mean"]["survival_frames"] == 200
    assert selected["bad"]["seed"] == 30
    assert selected["bad"]["survival_frames"] == 100
    assert set(selected) == set(REPRESENTATIVE_ROLES)


def test_representative_set_evaluates_training_seeds_only(
    monkeypatch, tmp_path: Path
) -> None:
    import dodge.ng.replay as replay_module

    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    config = DQNConfig(native_lanes=32)
    run_directory = tmp_path / "run"
    checkpoint = run_directory / "checkpoint-best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    evaluation_calls: list[tuple[int, ...]] = []

    class FakeModel:
        def __init__(self, *, hidden_size: int) -> None:
            assert hidden_size == config.hidden_size

        def load_state_dict(self, state: object) -> None:
            assert state == {"marker": 1}

        def eval(self) -> FakeModel:
            return self

    def evaluate(
        _model: object, seeds: tuple[int, ...], _config: DQNConfig
    ) -> dict[str, object]:
        evaluation_calls.append(tuple(seeds))
        survival = [100 + index for index, _seed in enumerate(seeds)]
        return {
            "seeds": list(seeds),
            "survival_frames": survival,
            "summary": {
                "mean_survival_frames": sum(survival) / len(survival),
            },
        }

    def record(run: Path, seed: int, **_kwargs: object) -> dict[str, object]:
        replay_directory = run / "dashboard" / "replays"
        replay_directory.mkdir(parents=True, exist_ok=True)
        frame_file = f"seed-{seed}.bin"
        (replay_directory / frame_file).write_bytes(bytes(128 * 128))
        return {
            "frame_file": frame_file,
            "checkpoint_step": 7,
            "frame_count": 1,
            "playback_start": 0,
            "playback_frame_count": 1,
        }

    monkeypatch.setattr(replay_module, "DuelingWaypointDQN", FakeModel)
    monkeypatch.setattr(
        replay_module,
        "_load_checkpoint_payload",
        lambda _path: {
            "config": config.to_json(),
            "manifest_sha256": manifest.sha256,
            "best_model_state": {"marker": 1},
            "best_inner": {"step": 7},
        },
    )
    monkeypatch.setattr(replay_module, "evaluate_waypoint_dqn", evaluate)
    monkeypatch.setattr(replay_module, "record_replay", record)

    result = record_representative_replays(run_directory, checkpoint=checkpoint)

    assert evaluation_calls == [manifest.training_seeds]
    assert result["selection_split"] == "training"
    assert set(result["roles"]) == set(REPRESENTATIVE_ROLES)
    for role in REPRESENTATIVE_ROLES:
        metadata_path = (
            run_directory
            / "dashboard"
            / "replays"
            / result["roles"][role]["metadata_file"]
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["replay_role"] == role
        assert metadata["selection_split"] == "training"
        assert metadata["seed"] in manifest.training_seeds

    replay_set = json.loads(
        (run_directory / "dashboard" / "representative-replays.json").read_text(
            encoding="utf-8"
        )
    )
    assert replay_set["manifest_sha256"] == manifest.sha256
    assert replay_set["selection_split"] == "training"


def test_dashboard_exposes_role_replays_and_role_controls() -> None:
    page = DASHBOARD_PAGE.read_text(encoding="utf-8")

    for role in REPRESENTATIVE_ROLES:
        assert f'data-replay-role="{role}"' in page
    assert "Training comparison" in page


def test_dashboard_lists_role_replays_without_accepting_missing_files(
    tmp_path: Path,
) -> None:
    replay_directory = tmp_path / "run" / "dashboard" / "replays"
    replay_directory.mkdir(parents=True)
    for role in REPRESENTATIVE_ROLES:
        frame_file = f"{role}.bin"
        (replay_directory / frame_file).write_bytes(bytes(128 * 128))
        (replay_directory / f"{role}.json").write_text(
            json.dumps(
                {
                    "frame_file": frame_file,
                    "frame_count": 1,
                    "done": False,
                    "replay_role": role,
                    "seed": 30_100,
                    "survival_frames": 800,
                }
            ),
            encoding="utf-8",
        )
    (replay_directory / "missing.json").write_text(
        json.dumps(
            {
                "frame_file": "missing.bin",
                "frame_count": 1,
                "replay_role": "bad",
            }
        ),
        encoding="utf-8",
    )

    replays = RunInspector(tmp_path / "run")._read_replays()

    assert {item["replay_role"] for item in replays} == set(REPRESENTATIVE_ROLES)
    assert all(item["url"].startswith("/replay/") for item in replays)


def test_v69_dashboard_reads_bounded_metrics_tail(tmp_path: Path) -> None:
    metrics_path = tmp_path / "run" / "dashboard" / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(
        "\n".join(json.dumps({"step": step}) for step in range(1_000)),
        encoding="utf-8",
    )

    metrics = RunInspector(tmp_path / "run")._read_metrics()

    assert len(metrics) == 500
    assert metrics[0]["step"] == 500
    assert metrics[-1]["step"] == 999
