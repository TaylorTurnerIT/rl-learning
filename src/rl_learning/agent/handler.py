from logging import Logger, getLogger

from ..env import EnvBuilder, EnvParams
from .agent import Agent
from .brain import Brain
from .fitness import Fitness

logger = getLogger(__name__)


class AgentHandler:
    def __init__(self, population: int, env_params: EnvParams):
        # Assign the previous best or generate new brain
        self.brain = Brain(action_count=env_params.map_size)
        self.population: int = population
        self.agents: list[Agent] = []
        self.best_agent: tuple[Agent, Fitness]
        self.env_builder: EnvBuilder = EnvBuilder(env_params)

    def create_agents(self, logger: Logger):
        self.brain = self.best_agent[0].brain
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
        for id in range(self.population):
            results = self.agents[id].run_actions()
            if results > self.best_agent[1]:
                self.best_agent = (self.agents[id], results)

        logger.info("agent set complete")
        winning_actions = (
            self.best_agent[0].brain.actions if self.best_agent is not None else None
        )
        logger.info("current winning action set: %s", winning_actions)

    def mutate_agents(self):
        for id in range(self.population):
            self.agents[id].mutate()
