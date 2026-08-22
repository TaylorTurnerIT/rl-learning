from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from dodge.neat.bridge import BridgeResult
from dodge.neat.environment import DodgeEnv, load_episode
from dodge.neat.state import RawState, parse_raw_state


def _state(frame: int) -> RawState:
    return parse_raw_state(
        f"__state__{frame}|10,20,1,0,4|30,20,-1,0,4,4,0,0|",
        prefix="__state__",
    )


class FakeBridge:
    instances: list[FakeBridge] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.actions: list[str] = []
        self.closed = False
        self.instances.append(self)

    def start(self) -> RawState:
        return _state(20)

    def step(self, action: str) -> RawState | BridgeResult:
        self.actions.append(action)
        if len(self.actions) == 1:
            return _state(24)
        return BridgeResult(
            state=_state(28),
            score=7,
            frames=28,
            survival_frames=8,
            seed=123,
            max_visible_enemies=17,
            max_visible_aoes=9,
            enemy_overflow_frames=2,
            aoe_overflow_frames=1,
        )

    def close(self) -> None:
        self.closed = True


def test_environment_steps_exact_intervals_and_saves_terminal_trace(
    tmp_path: Path,
) -> None:
    FakeBridge.instances.clear()
    environment = DodgeEnv(bridge_factory=FakeBridge)  # type: ignore[arg-type]

    initial = environment.reset(seed=123)
    first = environment.step("right")
    terminal = environment.step("up_left")

    assert initial.raw_state.frame == 20
    assert len(initial.projected.values) == 197
    assert first.reward == 4
    assert first.done is False
    assert terminal.reward == 4
    assert terminal.done is True
    assert terminal.result is not None
    assert terminal.result.survival_frames == 8
    assert FakeBridge.instances[0].actions == ["right", "up_left"]

    saved = environment.save_episode(
        tmp_path, created_at=datetime(2026, 8, 21, tzinfo=UTC)
    )
    trace = load_episode(saved)
    assert trace.seed == 123
    assert trace.actions == ("right", "up_left")
    assert trace.max_visible_enemies == 17
    assert trace.enemy_overflow_frames == 2


def test_environment_uses_entropy_when_seed_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeBridge.instances.clear()
    monkeypatch.setattr("dodge.neat.environment.secrets.randbelow", lambda _: 999)
    environment = DodgeEnv(bridge_factory=FakeBridge)  # type: ignore[arg-type]

    environment.reset()

    assert FakeBridge.instances[0].kwargs["seed"] == 999


def test_environment_rejects_steps_after_terminal_result() -> None:
    environment = DodgeEnv(bridge_factory=FakeBridge)  # type: ignore[arg-type]
    environment.reset(seed=123)
    environment.step("right")
    environment.step("up_left")

    with pytest.raises(RuntimeError, match="episode is complete"):
        environment.step("left")
