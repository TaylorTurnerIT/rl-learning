import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from dodge.agent.agent import Agent
from dodge.agent.brain import Brain
from dodge.headless import HeadlessResult, replay_commands
from dodge.history import create_run, save_epoch, save_winner

ELITE_COUNT = 5


def evaluate_epoch(
    agents: list[Agent], executor: ThreadPoolExecutor
) -> list[HeadlessResult]:
    return list(executor.map(Agent.run_actions, agents))


def rank_agents(
    agents: list[Agent], results: list[HeadlessResult]
) -> list[tuple[int, int]]:
    return sorted(
        (
            (id, agent.calculate_fitness(result))
            for id, (agent, result) in enumerate(zip(agents, results, strict=True))
        ),
        key=lambda candidate: candidate[1],
        reverse=True,
    )


def breed_next_generation(
    agents: list[Agent],
    ranked_agents: list[tuple[int, int]],
    elite_count: int = ELITE_COUNT,
) -> None:
    parent_count = min(elite_count, len(ranked_agents))
    if parent_count == 0:
        return

    elite_brains = [
        deepcopy(agents[id].brain) for id, _ in ranked_agents[:parent_count]
    ]
    for id, agent in enumerate(agents):
        agent.reset()
        agent.brain = deepcopy(elite_brains[id % parent_count])
        if id >= parent_count:
            agent.brain.mutate_actions()


def main():
    # Initialize config
    population: int = 100
    mutation_chance: float = 0.05

    # Create brain
    starting_brain: Brain = Brain(mutation_chance)

    # Create agents
    agents: list[Agent] = []
    epoch = 1
    max_epoch = 100
    best_brain: Brain | None = None
    best_fitness: int | None = None
    done = False
    run_directory = create_run(
        seed=42,
        population=population,
        mutation_chance=mutation_chance,
        max_epochs=max_epoch,
    )
    worker_count = min(population, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        while not done:
            if epoch == 1:
                for _ in range(population):
                    brain = deepcopy(starting_brain)
                    brain.mutate_actions()
                    agents.append(Agent(brain=brain))

            results = evaluate_epoch(agents, executor)
            ranked_agents = rank_agents(agents, results)
            epoch_best = ranked_agents[0]
            if best_fitness is None or epoch_best[1] > best_fitness:
                best_brain = deepcopy(agents[epoch_best[0]].brain)
                best_fitness = epoch_best[1]

            if best_fitness is not None:
                print(
                    f"epoch {epoch} best: {epoch_best[1]} (global best: {best_fitness})"
                )

            if best_fitness is not None:
                epoch_agent = agents[epoch_best[0]]
                save_epoch(
                    epoch_agent.brain.parse_actions(),
                    epoch=epoch,
                    seed=42,
                    fitness=epoch_best[1],
                    global_best_fitness=best_fitness,
                    headless_result=results[epoch_best[0]],
                    directory=run_directory,
                )

            breed_next_generation(agents, ranked_agents)
            if epoch == max_epoch:
                done = True
            epoch += 1
    if best_brain is not None and best_fitness is not None:
        commands = best_brain.parse_actions()
        replay_result = replay_commands(commands, seed=42)
        history_path = save_winner(
            commands,
            seed=42,
            fitness=best_fitness,
            epochs=epoch - 1,
            replay_result=replay_result,
            directory=run_directory,
            filename="winner.json",
        )
        print(f"saved run: {run_directory}")
        print(f"saved winner: {history_path}")
    return 0


if __name__ == "__main__":
    main()
