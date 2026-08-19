import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from dodge.agent.agent import Agent
from dodge.agent.brain import Brain
from dodge.headless import HeadlessResult, replay_commands


def evaluate_epoch(
    agents: list[Agent], executor: ThreadPoolExecutor
) -> list[HeadlessResult]:
    return list(executor.map(Agent.run_actions, agents))


def main():
    # Initialize config
    population: int = 10
    mutation_chance: float = 0.1

    # Create brain
    starting_brain: Brain = Brain(mutation_chance)

    # Create agents
    agents: list[Agent] = []
    epoch = 1
    max_epoch = 50
    best_agent: tuple[int, int] | None = None
    done = False
    worker_count = min(population, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        while not done:
            if epoch == 1:
                for _ in range(population):
                    brain = deepcopy(starting_brain)
                    brain.mutate_actions()
                    agents.append(Agent(brain=brain))

            results = evaluate_epoch(agents, executor)
            for id, (agent, result) in enumerate(zip(agents, results, strict=True)):
                fitness = agent.calculate_fitness(result)

                print(
                    id + 1,
                    "out of ",
                    population,
                    f"complete. score: {fitness} moves: ",
                )
                print(agent.brain.actions)

                if best_agent is None or fitness > best_agent[1]:
                    best_agent = (id, fitness)

                # agents[id].brain.mutate_actions()
            if best_agent is not None:
                print(best_agent[1])

            # brain surgery
            if best_agent is not None:
                elite_id = best_agent[0]
                elite_brain = deepcopy(agents[elite_id].brain)

                for id, agent in enumerate(agents):
                    agent.reset()

                    if id == elite_id:
                        agent.brain = elite_brain
                        continue

                    child_brain = deepcopy(elite_brain)
                    child_brain.mutate_actions()
                    agent.brain = child_brain
            if epoch == max_epoch:
                done = True
            epoch += 1
    if best_agent is not None:
        winner = agents[best_agent[0]]
        replay_commands(winner.brain.parse_actions(), seed=42)
    return 0


if __name__ == "__main__":
    main()
