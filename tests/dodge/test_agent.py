from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from dodge.agent.agent import Agent, State
from dodge.agent.brain import Brain
from dodge.control import MovementCommand
from dodge.main import evaluate_epoch


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


def test_evaluate_epoch_runs_each_agent_once(monkeypatch) -> None:
    agents = [Agent(Brain(0.1)) for _ in range(3)]
    expected = {
        id(agent): {
            "score": 0,
            "frames": 10,
            "survival_frames": index,
            "seed": 42,
            "started": True,
            "died": True,
        }
        for index, agent in enumerate(agents)
    }

    monkeypatch.setattr(
        Agent,
        "run_actions",
        lambda agent: expected[id(agent)],
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = evaluate_epoch(agents, executor)

    assert results == [expected[id(agent)] for agent in agents]
