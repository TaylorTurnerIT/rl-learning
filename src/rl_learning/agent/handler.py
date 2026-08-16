from dataclasses import dataclass
from logging import Logger

from ..env import Env, EnvBuilder, EnvParams
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
    def __init__(self, population: int, env_params: EnvParams):
        # Assign the previous best or generate new brain
        self.brain = Brain()
        self.population: int = population
        self.agents: list[Agent] = []
        self.best_agent: Agent | None = None
        self.env_builder: EnvBuilder = EnvBuilder(env_params)

    def create_agents(self, brain: Brain, logger: Logger):
        if self.best_agent is not None:
            self.brain = self.best_agent.brain
        for id in range(self.population):
            self.agents.append(
                Agent(id=id, brain=brain, env=env_builder.build(), logger=logger)
            )
