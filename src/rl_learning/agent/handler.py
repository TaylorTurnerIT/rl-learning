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

    def create_agents(self, brain: Brain, logger: Logger):
        if self.best_agent is not None:
            self.brain = self.best_agent.brain
        for id in range(self.population):
            env_builder = (
                EnvBuilder()
                .logger(logger)
                .map_size(5)
                .agent_index(0)
                .win_index(4)
                .build()
            )
            self.agents.append(
                Agent(id=id, brain=brain, env=env_builder.env, logger=logger)
            )
