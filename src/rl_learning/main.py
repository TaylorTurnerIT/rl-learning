import logging

from .env import Env


def main():
    logger = logging.getLogger()
    done = False

    state = Env(logger)

    while not done:
        state.show()
        done = True


if __name__ == "__main__":
    main()
