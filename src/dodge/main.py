from dodge.agent.agent import Agent
from dodge.agent.brain import Brain
from dodge.control import control


def main():
    # Initialize config
    population: int = 100
    mutation_chance: float = 0.1

    # Create brain
    starting_brain: Brain = Brain(mutation_chance)

    # Create agents
    agents: list[Agent] = []
    for _ in range(population):
        agents.append(Agent(brain=starting_brain))

    return 0


if __name__ == "__main__":
    main()
