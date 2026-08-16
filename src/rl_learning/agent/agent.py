from ..env import Env
from .brain import Brain


class Agent:
    def __init__(self, brain: Brain, env: Env):
        self.brain = brain
        self.env = env
