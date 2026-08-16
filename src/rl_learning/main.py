import logging
from enum import IntEnum

from rl_learning.agent.handler import AgentHandler

from .env import Direction, Env, EnvBuilder, EnvParams


class LogLevel(IntEnum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def main():
    logger = logging.getLogger()
    level = LogLevel.DEBUG
    logging.basicConfig(
        level=level,  # Capture INFO and above
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log message structure
    )

    done = False
    env_params = EnvParams(map_size=5, agent_idx=0, win_idx=4, logger=logger)
    agents = AgentHandler(population=100, env_params=env_params)
    while not done:
        done = True


if __name__ == "__main__":
    main()
