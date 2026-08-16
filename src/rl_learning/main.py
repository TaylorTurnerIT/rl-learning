import logging
from enum import IntEnum

from .agent.handler import AgentHandler
from .env import EnvParams


class LogLevel(IntEnum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def main():
    logger = logging.getLogger()
    level = LogLevel.INFO
    logging.basicConfig(
        level=level,  # Capture INFO and above
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log message structure
    )

    iteration = 0
    epochs = 100
    population = 100
    env_params = EnvParams(map_size=5, agent_idx=0, win_idx=4, logger=logger)
    logger.info("starting workflow with population=%d", population)
    agents = AgentHandler(population=population, env_params=env_params)
    agents.create_agents(logger=logger)
    while iteration != epochs:
        agents.run_agents()
        agents.mutate_agents()
        iteration += 1
    logger.info("workflow complete")


if __name__ == "__main__":
    main()
