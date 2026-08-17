from copy import deepcopy

from rl_learning.agent.agent import Agent
from rl_learning.agent.brain import Brain
from rl_learning.game import Game, Position, State
from rl_learning.visualization.replay import replay_history


def main():
    game = Game(
        x_size=50,
        y_size=50,
        pit_pos=Position(x=1, y=5),
        wumpus_pos=Position(x=0, y=0),
        win_pos=Position(x=10, y=50),
    )
    brain = Brain(mutation_chance=0.10)

    done = False
    epoch = 1
    history: list[list[Agent]] = []
    agents: list[Agent] = []
    while done is False:
        print("Epoch: ", epoch)

        # scores: list[tuple[int, int]] = []
        population = 100
        best_agent: tuple[int, int] = (-1, -100000)
        for id in range(population):
            if epoch == 1:
                agents.append(
                    Agent(starting_pos=Position(0, 0), brain=brain, game=game)
                )
            agents[id].run_actions()

            fitness_score = agents[id].calculate_fitness()
            # scores.append((id, fitness_score))

            if fitness_score > best_agent[1]:
                best_agent = (id, fitness_score)

        print("id:", best_agent[0])
        print("score:", best_agent[1])
        print("state:", agents[best_agent[0]].state)
        print("move count:", len(agents[best_agent[0]].brain.actions))
        # print("moves:", agents[best_agent[0]].brain.actions)

        # Capture the population that was just evaluated, not its children.
        history.append(deepcopy(agents))

        if agents[best_agent[0]].state == State.WON:
            print("WINNER")
            break

        # brain surgery
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

        epoch += 1

    replay_history(history)


if __name__ == "__main__":
    main()
