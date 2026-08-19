from copy import deepcopy

from dodge.agent.agent import Agent
from dodge.agent.brain import Brain
from dodge.headless import HeadlessResult


def main():
    # Initialize config
    population: int = 100
    mutation_chance: float = 0.1

    # Create brain
    starting_brain: Brain = Brain(mutation_chance)

    # Create agents
    agents: list[Agent] = []
    epoch = 1
    best_agent: tuple[int, HeadlessResult] | None = None
    for id in range(population):
        if epoch == 1:
            agents.append(Agent(brain=deepcopy(starting_brain)))

        result = agents[id].run_actions()

        if best_agent is None or result["score"] > best_agent[1]["score"]:
            best_agent = (id, result)
        print(id, "out of ", population, "complete.")
    if best_agent is not None:
        print(best_agent[1]["score"])
    return 0


if __name__ == "__main__":
    main()
