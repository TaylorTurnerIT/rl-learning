from .env import Env

def main():
    done = False
    env = Env

    state = env.reset(env)

    while not done:
        state
        done = True


if __name__ == "__main__":
    main()
