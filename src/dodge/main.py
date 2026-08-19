from copy import deepcopy

from dodge.agent.agent import Agent
from dodge.agent.brain import Brain


def main():
    # Initialize config
    population: int = 100
    mutation_chance: float = 0.1

    # Create brain
    starting_brain: Brain = Brain(mutation_chance)

    # Create agents
    agents: list[Agent] = []
    epoch = 1
    best_agent: tuple[int, int] | None = None
    done = False
    while not done:
        for id in range(population):
            if epoch == 1:
                brain = deepcopy(starting_brain)
                brain.mutate_actions()
                agents.append(Agent(brain=brain))

            result = agents[id].run_actions()
            fitness = agents[id].calculate_fitness(result)

            print(id + 1, "out of ", population, f"complete. score: {fitness} moves: ")
            print(agents[id].brain.actions)

            if best_agent is None or fitness > best_agent[1]:
                best_agent = (id, fitness)

            # agents[id].brain.mutate_actions()
        if best_agent is not None:
            print(best_agent[1])
        epoch += 1
    return 0


if __name__ == "__main__":
    main()
