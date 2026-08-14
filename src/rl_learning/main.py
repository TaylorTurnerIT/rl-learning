import logging

from .env import Direction, Env


def main():
    logger = logging.getLogger()
    done = False

    state: Env = Env(logger)

    while not done:
        state.show()
        state = state.move(Direction.RIGHT)
        state.show()
        state = state.move(Direction.RIGHT)
        state.show()
        state = state.move(Direction.LEFT)
        state.show()
        done = True


if __name__ == "__main__":
    main()
