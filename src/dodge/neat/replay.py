from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dodge.control import ControlInputError, ControlRuntimeError, MovementCommand
from dodge.headless import HeadlessResult, replay_commands
from dodge.neat.environment import EpisodeTrace, load_episode

_MENU_START_FRAMES = 3


def trace_commands(trace: EpisodeTrace) -> list[MovementCommand]:
    return [
        MovementCommand("x", _frames_to_milliseconds(_MENU_START_FRAMES)),
        *[
            MovementCommand(action, _frames_to_milliseconds(trace.step_frames))
            for action in trace.actions
        ],
    ]


def replay_episode(trace: EpisodeTrace) -> HeadlessResult:
    result = replay_commands(
        trace_commands(trace),
        seed=trace.seed,
        wait_for_game_start=True,
        legacy_mouse_input=trace.input_mode == "legacy_mouse",
    )
    expected = trace.result
    actual = {
        "score": result["score"],
        "frames": result["frames"],
        "survival_frames": result["survival_frames"],
        "seed": result["seed"],
    }
    wanted = {
        "score": expected.score,
        "frames": expected.frames,
        "survival_frames": expected.survival_frames,
        "seed": expected.seed,
    }
    if actual != wanted:
        raise ControlRuntimeError(
            f"NEAT replay diverged: expected {wanted}, got {actual}"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-neat-replay",
        description="Replay a saved NEAT Dodge episode without live controls.",
    )
    parser.add_argument("episode", type=Path, help="saved NEAT episode JSON")
    arguments = parser.parse_args(argv)
    try:
        result = replay_episode(load_episode(arguments.episode))
    except (ControlInputError, ControlRuntimeError, ValueError) as error:
        print(f"dodge-neat-replay: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


def _frames_to_milliseconds(frames: int) -> int:
    return (frames * 1_000) // 60


if __name__ == "__main__":
    raise SystemExit(main())
