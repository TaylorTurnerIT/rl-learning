from ..env import Env
from .brain import Brain


class AgentHandler:
    def __init__(self, env: Env, population: int, brain: Brain | None = None):
        # Assign the previous best or generate new brain
        self.brain = brain
        if self.brain is None:
            self.brain = Brain()

        
