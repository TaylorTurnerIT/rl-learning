from rl_learning.agent.agent import Agent
from rl_learning.agent.brain import Brain
from rl_learning.game import Game, Position


def main():
    game = Game(
        x_size=5,
        y_size=5,
        pit_pos=Position(x=1, y=5),
        wumpus_pos=Position(x=0, y=0),
        win_pos=Position(x=1, y=3),
    )
    brain = Brain(mutation_chance=0.05)

    done = False
    epoch = 1
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
        print("move count:", agents[best_agent[0]].move_count)
        # print("moves:", agents[best_agent[0]].brain.actions)

        if best_agent[1] >= 10000:
            print("WINNER")
            break

        # brain surgery
        for id in range(population):
            if id == best_agent[0]:
                continue
            agents[id].brain = agents[best_agent[0]].brain
            agents[id].brain.mutate_actions()
        epoch += 1


if __name__ == "__main__":
    main()
