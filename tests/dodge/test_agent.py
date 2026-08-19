from __future__ import annotations

from dodge.agent.agent import Agent, State
from dodge.agent.brain import Brain
from dodge.control import MovementCommand


def test_brain_commands_wait_for_menu_transition() -> None:
    brain = Brain(0.1)
    brain.actions = ["left", "up_right"]

    assert brain.parse_actions()[:2] == [
        MovementCommand("x", 50),
        MovementCommand("neutral", 750),
    ]


def test_agent_fitness_uses_survival_frames() -> None:
    agent = Agent(Brain(0.1))

    assert (
        agent.calculate_fitness(
            {
                "score": 99,
                "frames": 120,
                "survival_frames": 37,
                "seed": 42,
                "started": True,
                "died": True,
            }
        )
        == 37
    )


def test_agent_marks_itself_dead_after_episode(monkeypatch) -> None:
    agent = Agent(Brain(0.1))
    monkeypatch.setattr(
        "dodge.agent.agent.run_headless",
        lambda **_: {
            "score": 0,
            "frames": 10,
            "survival_frames": 4,
            "seed": 42,
            "started": True,
            "died": True,
        },
    )

    agent.run_actions()

    assert agent.state is State.DEAD
