"""Original-cartridge pixel regression for saved NG replays."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from dodge.control import ControlRuntimeError, MovementCommand
from dodge.dataset import ACTION_CHOICES
from dodge.native.differential import FRAME_SIZE, FRAME_WIDTH
from dodge.native.manifest import canonical_json
from dodge.native.oracle import run_oracle_trace

PIXEL_REGRESSION_VERSION = 1
PIXEL_COMPARISON = "original_cartridge_pemsa_palette_index"
_DURATION_MS_BY_STEP_FRAMES = {3: 50, 4: 66, 5: 83}


def compare_saved_replay(
    run_directory: Path,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Compare every saved replay frame with the original PICO-8 cartridge."""
    replay_directory = (Path(run_directory) / "dashboard" / "replays").resolve()
    seed = _integer(metadata.get("seed"), "replay seed")
    frame_count = _integer(metadata.get("frame_count"), "replay frame count")
    initial_frame = _integer(metadata.get("playback_start_frame"), "initial frame")
    step_frames = _integer(metadata.get("step_frames"), "replay step frames")
    native_steps = _integer(metadata.get("native_steps"), "native step count")
    if frame_count < 0 or initial_frame < 0 or native_steps < 0:
        raise ControlRuntimeError(
            "replay regression metadata contains a negative count"
        )
    if step_frames not in _DURATION_MS_BY_STEP_FRAMES:
        raise ControlRuntimeError("replay regression step frames must be 3, 4, or 5")

    frame_file = metadata.get("frame_file")
    if not isinstance(frame_file, str) or not frame_file:
        raise ControlRuntimeError("replay regression frame file is invalid")
    frame_path = (replay_directory / frame_file).resolve()
    if frame_path.parent != replay_directory or frame_path.name != frame_file:
        raise ControlRuntimeError(
            "replay regression frame file escapes replay directory"
        )

    actions = _action_trace(metadata.get("action_trace"), native_steps)
    saved_frame_numbers = _saved_frame_numbers(
        metadata.get("action_trace"), frame_count, initial_frame, step_frames
    )
    try:
        replay_bytes = frame_path.read_bytes()
    except OSError as error:
        raise ControlRuntimeError(
            f"could not read saved replay frames: {error}"
        ) from error
    expected_bytes = frame_count * FRAME_SIZE
    if len(replay_bytes) != expected_bytes:
        raise ControlRuntimeError(
            "saved replay frame data has unexpected size: "
            f"expected {expected_bytes}, got {len(replay_bytes)}"
        )

    commands = _commands_for_action_trace(actions, step_frames)
    reset_mode = metadata.get("reset_mode")
    if reset_mode not in {"legacy", "native-startup"}:
        raise ControlRuntimeError("replay regression reset mode is invalid")
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise ControlRuntimeError("replay regression configuration is missing")
    grid_spacing = _integer(config.get("grid_spacing"), "grid spacing")

    oracle = run_oracle_trace(
        commands,
        seed=seed,
        timeout=_oracle_timeout(
            saved_frame_numbers[-1] if saved_frame_numbers else initial_frame
        ),
        capture_frame_limit=(
            None
            if metadata.get("done") is True
            else (saved_frame_numbers[-1] if saved_frame_numbers else initial_frame)
        ),
        native_startup_grid_spacing=(
            grid_spacing if reset_mode == "native-startup" else None
        ),
        capture_frame_indices=saved_frame_numbers,
    )
    oracle_frames = {frame.frame_index: frame.pixels for frame in oracle.frames}
    first_mismatch: dict[str, object] | None = None
    differing_pixels = 0
    frames_compared = 0
    for index, frame_number in enumerate(saved_frame_numbers):
        expected = oracle_frames.get(frame_number)
        actual = replay_bytes[index * FRAME_SIZE : (index + 1) * FRAME_SIZE]
        if expected is None:
            if first_mismatch is None:
                first_mismatch = {
                    "saved_frame_index": index,
                    "game_frame": frame_number,
                    "reason": "oracle_frame_missing",
                }
            break
        frames_compared += 1
        frame_difference = _count_differences(actual, expected)
        differing_pixels += frame_difference
        if frame_difference and first_mismatch is None:
            pixel_index = _first_difference(actual, expected)
            first_mismatch = {
                "saved_frame_index": index,
                "game_frame": frame_number,
                "pixel_index": pixel_index,
                "x": pixel_index % FRAME_WIDTH,
                "y": pixel_index // FRAME_WIDTH,
                "expected": expected[pixel_index],
                "actual": actual[pixel_index],
                "differing_pixels_in_frame": frame_difference,
            }

    return {
        "version": PIXEL_REGRESSION_VERSION,
        "status": "passed" if first_mismatch is None else "mismatch",
        "comparison": PIXEL_COMPARISON,
        "seed": seed,
        "manifest_sha256": metadata.get("manifest_sha256"),
        "checkpoint_file": metadata.get("checkpoint_file"),
        "checkpoint_step": metadata.get("checkpoint_step"),
        "config": dict(config),
        "reset_mode": reset_mode,
        "step_frames": step_frames,
        "grid_spacing": grid_spacing,
        "frames_compared": frames_compared,
        "saved_frame_count": frame_count,
        "native_steps": native_steps,
        "differing_pixels": differing_pixels,
        "first_mismatch": first_mismatch,
        "action_trace_sha256": _action_trace_sha256(
            actions, saved_frame_numbers, step_frames
        ),
        "oracle": {
            "source": oracle.provenance.get("source"),
            "pemsa": oracle.provenance.get("pemsa"),
            "capture_mode": oracle.provenance.get("capture_mode"),
            "scenario": oracle.scenario,
            "trace_frame_count": len(oracle.frames),
            "result": oracle.result,
        },
    }


def unavailable_pixel_regression(
    metadata: Mapping[str, object], error: Exception
) -> dict[str, object]:
    """Build a non-passing report when the original oracle cannot run."""
    action_trace = metadata.get("action_trace")
    actions: list[int] = []
    saved_frame_numbers: list[int] = []
    step_frames = metadata.get("step_frames")
    if isinstance(action_trace, Mapping):
        raw_actions = action_trace.get("actions")
        if isinstance(raw_actions, Sequence) and not isinstance(
            raw_actions, (str, bytes)
        ):
            actions = [value for value in raw_actions if isinstance(value, int)]
        raw_frames = action_trace.get("saved_frame_numbers")
        if isinstance(raw_frames, Sequence) and not isinstance(
            raw_frames, (str, bytes)
        ):
            saved_frame_numbers = [
                value for value in raw_frames if isinstance(value, int)
            ]
    trace_hash = None
    if isinstance(step_frames, int) and actions and saved_frame_numbers:
        trace_hash = _action_trace_sha256(actions, saved_frame_numbers, step_frames)
    return {
        "version": PIXEL_REGRESSION_VERSION,
        "status": "unavailable",
        "comparison": PIXEL_COMPARISON,
        "seed": metadata.get("seed"),
        "manifest_sha256": metadata.get("manifest_sha256"),
        "checkpoint_file": metadata.get("checkpoint_file"),
        "checkpoint_step": metadata.get("checkpoint_step"),
        "config": metadata.get("config"),
        "reset_mode": metadata.get("reset_mode"),
        "step_frames": step_frames,
        "frames_compared": 0,
        "saved_frame_count": metadata.get("frame_count"),
        "native_steps": metadata.get("native_steps"),
        "differing_pixels": None,
        "first_mismatch": None,
        "action_trace_sha256": trace_hash,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _commands_for_action_trace(
    actions: Sequence[int], step_frames: int
) -> list[MovementCommand]:
    duration_ms = _DURATION_MS_BY_STEP_FRAMES[step_frames]
    return [
        MovementCommand("x", 50),
        *(MovementCommand(ACTION_CHOICES[action], duration_ms) for action in actions),
    ]


def _action_trace(metadata: object, native_steps: int) -> list[int]:
    if not isinstance(metadata, Mapping):
        raise ControlRuntimeError("replay action trace is missing")
    raw_actions = metadata.get("actions")
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes)):
        raise ControlRuntimeError("replay action trace actions are invalid")
    if len(raw_actions) != native_steps:
        raise ControlRuntimeError(
            "replay action trace length does not match native steps"
        )
    actions: list[int] = []
    for action in raw_actions:
        if isinstance(action, bool) or not isinstance(action, int):
            raise ControlRuntimeError(
                "replay action trace contains a non-integer action"
            )
        if not 0 <= action < len(ACTION_CHOICES):
            raise ControlRuntimeError("replay action trace contains an invalid action")
        actions.append(action)
    return actions


def _saved_frame_numbers(
    metadata: object,
    frame_count: int,
    initial_frame: int,
    step_frames: int,
) -> list[int]:
    if not isinstance(metadata, Mapping):
        raise ControlRuntimeError("replay action trace is missing")
    raw_frames = metadata.get("saved_frame_numbers")
    if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)):
        raise ControlRuntimeError("replay saved frame numbers are invalid")
    if len(raw_frames) != frame_count:
        raise ControlRuntimeError(
            "replay saved frame number count does not match frame count"
        )
    frames: list[int] = []
    previous = initial_frame
    for frame in raw_frames:
        if isinstance(frame, bool) or not isinstance(frame, int) or frame <= previous:
            raise ControlRuntimeError("replay saved frame numbers are not monotonic")
        if frame - previous != step_frames:
            raise ControlRuntimeError(
                "replay saved frame numbers are not aligned to step cadence"
            )
        frames.append(frame)
        previous = frame
    return frames


def _action_trace_sha256(
    actions: Sequence[int], saved_frame_numbers: Sequence[int], step_frames: int
) -> str:
    payload = {
        "actions": list(actions),
        "saved_frame_numbers": list(saved_frame_numbers),
        "step_frames": step_frames,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _oracle_timeout(last_frame: int) -> float:
    """Allow indexed-pixel capture enough time for long saved trajectories."""
    return max(120.0, 30.0 + 0.15 * max(1, last_frame))


def _count_differences(actual: bytes, expected: bytes) -> int:
    return sum(left != right for left, right in zip(actual, expected, strict=True))


def _first_difference(actual: bytes, expected: bytes) -> int:
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        if left != right:
            return index
    raise ValueError("pixel frames do not differ")


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlRuntimeError(f"{name} is invalid")
    return value


__all__ = [
    "PIXEL_COMPARISON",
    "PIXEL_REGRESSION_VERSION",
    "compare_saved_replay",
    "unavailable_pixel_regression",
]
