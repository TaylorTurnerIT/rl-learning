from ..env import Env
from .agent import Agent
from .brain import Brain


class AgentHandler:
    def __init__(self, population: int):
        # Assign the previous best or generate new brain
        self.brain = Brain()
        self.population: int = population
        self.agents: list[Agent] = []
        self.best_agent: Agent | None = None

    def create_agents(self, brain: Brain):
        if self.best_agent is not None:
            self.brain = self.best_agent.brain
        for _ in range(self.population):
            self.agents.append(Agent(self.brain, ))
