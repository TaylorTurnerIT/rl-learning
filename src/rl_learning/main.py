import logging

from .env import Direction, Env, EnvBuilder, LogLevel


def main():
    logger = logging.getLogger()
    level = LogLevel.DEBUG

    state: Env = (
        EnvBuilder()
        .logger(logger, level)
        .map_size(5)
        .agent_index(0)
        .win_index(4)
        .build()
    )

    done = False
    while not done:
        state.show()
        state.move(Direction.RIGHT)
        state.show()
        state.move(Direction.RIGHT)
        state.show()
        state.move(Direction.LEFT)
        state.show()
        done = True


if __name__ == "__main__":
    main()
