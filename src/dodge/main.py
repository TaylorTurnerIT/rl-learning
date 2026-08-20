import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from dodge.agent.agent import Agent
from dodge.agent.brain import Brain
from dodge.headless import HeadlessResult, replay_commands
from dodge.history import create_run, save_epoch, save_winner


def evaluate_epoch(
    agents: list[Agent], executor: ThreadPoolExecutor
) -> list[HeadlessResult]:
    return list(executor.map(Agent.run_actions, agents))


def main():
    # Initialize config
    population: int = 100
    mutation_chance: float = 0.2

    # Create brain
    starting_brain: Brain = Brain(mutation_chance)

    # Create agents
    agents: list[Agent] = []
    epoch = 1
    max_epoch = 100
    best_agent: tuple[int, int] | None = None
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
            epoch_best: tuple[int, int] | None = None
            for id, (agent, result) in enumerate(zip(agents, results, strict=True)):
                fitness = agent.calculate_fitness(result)

                # print(
                #     id + 1,
                #     "out of ",
                #     population,
                #     f"complete. score: {fitness} moves: ",
                # )
                # print(agent.brain.actions)

                if best_agent is None or fitness > best_agent[1]:
                    best_agent = (id, fitness)
                if epoch_best is None or fitness > epoch_best[1]:
                    epoch_best = (id, fitness)

                # agents[id].brain.mutate_actions()
            if epoch_best is not None and best_agent is not None:
                print(
                    f"epoch {epoch} best: {epoch_best[1]} "
                    f"(global best: {best_agent[1]})"
                )

            if epoch_best is not None and best_agent is not None:
                epoch_agent = agents[epoch_best[0]]
                save_epoch(
                    epoch_agent.brain.parse_actions(),
                    epoch=epoch,
                    seed=42,
                    fitness=epoch_best[1],
                    global_best_fitness=best_agent[1],
                    headless_result=results[epoch_best[0]],
                    directory=run_directory,
                )

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
        commands = winner.brain.parse_actions()
        replay_result = replay_commands(commands, seed=42)
        history_path = save_winner(
            commands,
            seed=42,
            fitness=best_agent[1],
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
