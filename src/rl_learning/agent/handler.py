from dataclasses import dataclass
from logging import Logger

from ..env import Env, EnvBuilder
from .agent import Agent
from .brain import Brain


@dataclass
class Fitness:
    def __init__(
        self,
        win: bool,
        win_distance: int,
    ):
        pass


class AgentHandler:
    def __init__(self, population: int):
        # Assign the previous best or generate new brain
        self.brain = Brain()
        self.population: int = population
        self.agents: list[Agent] = []
        self.best_agent: Agent | None = None

        self.env = EnvBuilder()

    def create_agents(self, brain: Brain, env: Env, logger: Logger):
        if self.best_agent is not None:
            self.brain = self.best_agent.brain
        for _ in range(self.population):
            env =
            self.agents.append(Agent(brain, env, logger))
