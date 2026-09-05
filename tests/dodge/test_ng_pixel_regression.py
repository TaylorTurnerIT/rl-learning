from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dodge.control import ControlRuntimeError
from dodge.native.differential import FRAME_SIZE
from dodge.ng import pixel_regression


def _metadata(run_directory: Path) -> dict[str, object]:
    replay_directory = run_directory / "dashboard" / "replays"
    replay_directory.mkdir(parents=True)
    (replay_directory / "replay.bin").write_bytes(
        bytes([1]) * FRAME_SIZE + bytes([2]) * FRAME_SIZE
    )
    return {
        "seed": 30160,
        "manifest_sha256": "manifest",
        "checkpoint_file": "checkpoint-best.pt",
        "checkpoint_step": 680000,
        "config": {"grid_spacing": 24},
        "reset_mode": "native-startup",
        "frame_file": "replay.bin",
        "frame_count": 2,
        "playback_start_frame": 0,
        "step_frames": 4,
        "native_steps": 2,
        "done": False,
        "action_trace": {
            "actions": [3, 0],
            "saved_frame_numbers": [4, 8],
        },
    }


def _oracle(*pixels: bytes) -> SimpleNamespace:
    return SimpleNamespace(
        provenance={
            "source": {"sha256": "source"},
            "pemsa": {"sha256": "pemsa"},
            "capture_mode": "full_draw",
        },
        scenario={"wait_for_game_start": True},
        frames=tuple(
            SimpleNamespace(frame_index=index, pixels=frame)
            for index, frame in zip((4, 8), pixels, strict=True)
        ),
        result={"frames": 8, "died": True},
    )


def test_compare_saved_replay_uses_selected_original_frames_and_passes(
    monkeypatch, tmp_path: Path
) -> None:
    metadata = _metadata(tmp_path / "run")
    observed: dict[str, object] = {}

    def fake_oracle(commands, **kwargs):
        observed["commands"] = commands
        observed.update(kwargs)
        return _oracle(bytes([1]) * FRAME_SIZE, bytes([2]) * FRAME_SIZE)

    monkeypatch.setattr(pixel_regression, "run_oracle_trace", fake_oracle)

    report = pixel_regression.compare_saved_replay(tmp_path / "run", metadata)

    assert report["status"] == "passed"
    assert report["frames_compared"] == 2
    assert report["differing_pixels"] == 0
    assert report["first_mismatch"] is None
    assert report["action_trace_sha256"]
    assert observed["capture_frame_indices"] == [4, 8]
    assert observed["native_startup_grid_spacing"] == 24
    assert [command.move for command in observed["commands"]] == ["x", "up", "neutral"]
    assert [command.duration_ms for command in observed["commands"]] == [50, 66, 66]


def test_compare_saved_replay_reports_first_pixel_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    metadata = _metadata(tmp_path / "run")
    expected = bytes([1]) * FRAME_SIZE
    actual = bytearray(expected)
    actual[7] = 9
    monkeypatch.setattr(
        pixel_regression,
        "run_oracle_trace",
        lambda *_args, **_kwargs: _oracle(expected, bytes([2]) * FRAME_SIZE),
    )
    (tmp_path / "run" / "dashboard" / "replays" / "replay.bin").write_bytes(
        bytes(actual) + bytes([2]) * FRAME_SIZE
    )

    report = pixel_regression.compare_saved_replay(tmp_path / "run", metadata)

    assert report["status"] == "mismatch"
    assert report["frames_compared"] == 2
    assert report["differing_pixels"] == 1
    assert report["first_mismatch"] == {
        "saved_frame_index": 0,
        "game_frame": 4,
        "pixel_index": 7,
        "x": 7,
        "y": 0,
        "expected": 1,
        "actual": 9,
        "differing_pixels_in_frame": 1,
    }


def test_compare_saved_replay_keeps_earlier_pixel_mismatch_before_missing_frame(
    monkeypatch, tmp_path: Path
) -> None:
    metadata = _metadata(tmp_path / "run")
    metadata["frame_count"] = 3
    metadata["native_steps"] = 3
    metadata["action_trace"] = {
        "actions": [3, 0, 1],
        "saved_frame_numbers": [4, 8, 12],
    }
    expected = bytes([1]) * FRAME_SIZE
    actual = bytearray(expected)
    actual[7] = 9
    (tmp_path / "run" / "dashboard" / "replays" / "replay.bin").write_bytes(
        bytes(actual) + bytes([2]) * FRAME_SIZE + bytes([3]) * FRAME_SIZE
    )
    monkeypatch.setattr(
        pixel_regression,
        "run_oracle_trace",
        lambda *_args, **_kwargs: _oracle(expected, bytes([2]) * FRAME_SIZE),
    )

    report = pixel_regression.compare_saved_replay(tmp_path / "run", metadata)

    assert report["status"] == "mismatch"
    assert report["frames_compared"] == 2
    assert report["differing_pixels"] == 1
    assert report["first_mismatch"] == {
        "saved_frame_index": 0,
        "game_frame": 4,
        "pixel_index": 7,
        "x": 7,
        "y": 0,
        "expected": 1,
        "actual": 9,
        "differing_pixels_in_frame": 1,
    }


def test_compare_saved_replay_rejects_unaligned_saved_frames(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path / "run")
    metadata["action_trace"] = {
        "actions": [3, 0],
        "saved_frame_numbers": [4, 9],
    }

    with pytest.raises(ControlRuntimeError, match="aligned"):
        pixel_regression.compare_saved_replay(tmp_path / "run", metadata)


def test_unavailable_pixel_regression_is_not_a_pass() -> None:
    report = pixel_regression.unavailable_pixel_regression(
        {
            "seed": 30160,
            "frame_count": 2,
            "step_frames": 4,
            "action_trace": {
                "actions": [0, 1],
                "saved_frame_numbers": [4, 8],
            },
        },
        RuntimeError("Xvfb missing"),
    )

    assert report["status"] == "unavailable"
    assert report["error"] == "Xvfb missing"
    assert report["action_trace_sha256"]
