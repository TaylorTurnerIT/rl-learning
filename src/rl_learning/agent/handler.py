from logging import Logger, getLogger

from ..env import EnvBuilder, EnvParams
from .agent import Agent
from .brain import Brain
from .fitness import Fitness

logger = getLogger(__name__)


class AgentHandler:
    def __init__(self, population: int, env_params: EnvParams):
        # Assign the previous best or generate new brain
        self.brain = Brain()
        self.population: int = population
        self.agents: list[Agent] = []
        self.best_agent: Agent | None = None
        self.env_builder: EnvBuilder = EnvBuilder(env_params)

    def create_agents(self, logger: Logger):
        if self.best_agent is not None:
            self.brain = self.best_agent.brain
        for id in range(self.population):
            self.agents.append(
                Agent(
                    id=id, brain=self.brain, env=self.env_builder.build(), logger=logger
                )
            )
        logger.info("created agent set: %d agents", len(self.agents))

    def reset_agents(self):
        for id in range(self.population):
            self.agents[id].reset()

    def run_agents(self):
        logger.info("running agent set: %d agents", len(self.agents))
        scores: list[Fitness] = []
        for id in range(self.population):
            scores.append(self.agents[id].run_actions())
        logger.info("agent set complete: %d agents", len(scores))
        winning_actions = (
            self.best_agent.brain.actions if self.best_agent is not None else None
        )
        logger.info("current winning action set: %s", winning_actions)

    def mutate_agents(self):
        for id in range(self.population):
            self.agents[id].mutate()
