from __future__ import annotations

import shutil

import pytest

from dodge.control import PEMSA_PATH
from dodge.headless import run_headless
from dodge.neat.environment import DodgeEnv
from dodge.neat.replay import trace_commands


@pytest.mark.skipif(
    not PEMSA_PATH.is_file() or shutil.which("Xvfb") is None,
    reason="requires the checked-in Pemsa runtime and Xvfb",
)
def test_v5_live_trace_replays_with_identical_terminal_result() -> None:
    with DodgeEnv(step_frames=4) as environment:
        observation = environment.reset(seed=42)
        for action in ("right", "up_left", "neutral"):
            transition = environment.step(action)
            assert (
                transition.observation.raw_state.frame
                == observation.raw_state.frame + 4
            )
            observation = transition.observation

        while True:
            transition = environment.step("neutral")
            if transition.done:
                trace = environment.episode_trace
                break

    replayed = run_headless(trace_commands(trace), seed=trace.seed, timeout=30)
    assert replayed["score"] == trace.result.score
    assert replayed["frames"] == trace.result.frames
    assert replayed["survival_frames"] == trace.result.survival_frames
    assert replayed["seed"] == trace.seed
