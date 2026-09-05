from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dodge.ng.dashboard import DASHBOARD_PAGE, RunInspector, _replay_playback_window
from dodge.ng.replay import _write_frame
from dodge.ng.telemetry import DashboardTelemetry, issue_control


def test_telemetry_keeps_latest_state_and_writes_metrics(tmp_path: Path) -> None:
    telemetry = DashboardTelemetry(tmp_path / "run")
    telemetry.publish({"state": "running", "step": 1, "record": {"step": 1}})
    _wait_for_file(telemetry.status_path)
    telemetry.publish({"state": "running", "step": 2, "record": {"step": 2}})
    telemetry.close()

    status = json.loads(telemetry.status_path.read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in telemetry.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert status["step"] == 2
    assert metrics[-1]["step"] == 2


def test_dashboard_control_is_consumed_once(tmp_path: Path) -> None:
    telemetry = DashboardTelemetry(tmp_path / "run")
    issue_control(telemetry.run_directory, "pause")
    assert telemetry.consume_control() == "pause"
    assert telemetry.consume_control() is None
    issue_control(telemetry.run_directory, "resume")
    assert telemetry.consume_control() == "resume"
    telemetry.close()


def test_run_inspector_rejects_replay_path_escape(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    replay_directory = run_directory / "dashboard" / "replays"
    replay_directory.mkdir(parents=True)
    (replay_directory / "frame.bin").write_bytes(b"frame")
    inspector = RunInspector(run_directory)

    assert (
        inspector.resolve_replay("frame.bin")
        == (replay_directory / "frame.bin").resolve()
    )
    assert inspector.resolve_replay("../frame.bin") is None
    assert inspector.resolve_replay("%2e%2e/frame.bin") is None


def test_run_inspector_normalizes_replay_playback_window(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    replay_directory = run_directory / "dashboard" / "replays"
    replay_directory.mkdir(parents=True)
    (replay_directory / "legacy.bin").write_bytes(b"legacy")
    (replay_directory / "legacy.json").write_text(
        json.dumps(
            {
                "frame_file": "legacy.bin",
                "frame_count": 5,
                "done": True,
            }
        ),
        encoding="utf-8",
    )
    (replay_directory / "current.bin").write_bytes(b"current")
    (replay_directory / "current.json").write_text(
        json.dumps(
            {
                "frame_file": "current.bin",
                "frame_count": 3,
                "playback_start": 0,
                "playback_frame_count": 3,
                "done": True,
            }
        ),
        encoding="utf-8",
    )

    replays = RunInspector(run_directory)._read_replays()

    legacy = next(item for item in replays if item["frame_file"] == "legacy.bin")
    current = next(item for item in replays if item["frame_file"] == "current.bin")
    assert _replay_playback_window(legacy) == (1, 3)
    assert (legacy["playback_start"], legacy["playback_frame_count"]) == (1, 3)
    assert (current["playback_start"], current["playback_frame_count"]) == (0, 3)


def test_dashboard_page_has_latest_replay_autoplay_control() -> None:
    page = DASHBOARD_PAGE.read_text(encoding="utf-8")

    assert 'id="auto-play-latest"' in page
    assert "autoPlayLatest" in page
    assert "pixel_regression" in page
    assert 'id="life-losses"' in page
    assert 'id="lives"' in page


def test_replay_recorder_omits_reset_and_terminal_frames(
    monkeypatch, tmp_path: Path
) -> None:
    import torch

    import dodge.ng.replay as replay_module

    config = replay_module.DQNConfig(
        step_frames=4,
        hold_decisions=1,
        max_episode_steps=3,
        native_lanes=1,
    )
    checkpoint = tmp_path / "run" / "checkpoint-best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")

    class FakeModel:
        def __init__(self, *, hidden_size: int) -> None:
            assert hidden_size == config.hidden_size

        def load_state_dict(self, state: object) -> None:
            assert state == {"marker": torch.tensor(1)}

        def eval(self) -> FakeModel:
            return self

        def __call__(self, observations: torch.Tensor) -> torch.Tensor:
            return torch.zeros((observations.shape[0], 9))

    class FakeEnvironment:
        def __init__(self, **_kwargs: object) -> None:
            self.step_number = 0

        def __enter__(self) -> FakeEnvironment:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def reset_batch_with_startup(self, _seeds: object) -> SimpleNamespace:
            return self._result(0, False)

        def reset_batch(self, _seeds: object) -> SimpleNamespace:
            return self._result(0, False)

        def step_batch(self, _actions: object) -> SimpleNamespace:
            self.step_number += 1
            return self._result(self.step_number, self.step_number == 3)

        def _result(self, value: int, done: bool) -> SimpleNamespace:
            return SimpleNamespace(
                ml_observation=np.zeros(
                    (1, replay_module.WAYPOINT_OBSERVATION_SIZE), dtype=np.float32
                ),
                player_positions=np.zeros((1, 2), dtype=np.float32),
                pixels=np.full((1, 128, 128), value, dtype=np.uint8),
                frames=np.asarray([value], dtype=np.uint32),
                rewards=np.asarray([value], dtype=np.float32),
                done=np.asarray([done], dtype=np.bool_),
            )

    monkeypatch.setattr(
        replay_module,
        "_load_checkpoint_payload",
        lambda _path: {
            "config": config.to_json(),
            "best_model_state": {"marker": torch.tensor(1)},
            "step": 7,
        },
    )
    monkeypatch.setattr(replay_module, "DuelingWaypointDQN", FakeModel)
    monkeypatch.setattr(replay_module, "NativeBatchEnvironment", FakeEnvironment)
    monkeypatch.setattr(
        replay_module,
        "compare_saved_replay",
        lambda _run_directory, _metadata: {"status": "passed"},
    )

    metadata = replay_module.record_replay(checkpoint.parent, 7)
    frame_path = checkpoint.parent / "dashboard" / "replays" / metadata["frame_file"]

    assert metadata["version"] == 3
    assert metadata["frame_count"] == 2
    assert metadata["playback_start"] == 0
    assert metadata["playback_frame_count"] == 2
    assert metadata["playback_start_frame"] == 0
    assert metadata["survival_frames"] == 6
    assert metadata["reset_mode"] == "native-startup"
    assert metadata["native_steps"] == 3
    assert metadata["action_trace"] == {
        "encoding": "native_action_index_u8",
        "actions": [0, 0, 0],
        "saved_frame_numbers": [1, 2],
        "initial_frame": 0,
        "step_frames": 4,
    }
    assert metadata["pixel_regression"] == {"status": "passed"}
    assert frame_path.read_bytes() == bytes([1]) * (128 * 128) + bytes([2]) * (
        128 * 128
    )

    legacy = replay_module.record_replay(checkpoint.parent, 7, reset_mode="legacy")
    assert legacy["reset_mode"] == "legacy"


def test_replay_frame_writer_validates_palette_frame() -> None:
    pixels = np.zeros((1, 128, 128), dtype=np.uint8)
    result = SimpleNamespace(pixels=pixels)
    from io import BytesIO

    stream = BytesIO()
    assert _write_frame(stream, result) == 1
    assert len(stream.getvalue()) == 128 * 128


def _wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not path.is_file():
        time.sleep(0.01)
    assert path.is_file()
