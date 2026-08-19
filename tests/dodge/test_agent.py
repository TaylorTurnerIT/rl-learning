from __future__ import annotations

from dodge.agent.agent import Agent
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
            }
        )
        == 37
    )
