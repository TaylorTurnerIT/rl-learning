from rl_learning.agent.agent import Agent
from rl_learning.agent.brain import Brain
from rl_learning.game import Game, Position, State


def main():
    game = Game(
        x_size=5,
        y_size=5,
        pit_pos=Position(x=1, y=5),
        wumpus_pos=Position(x=0, y=0),
        win_pos=Position(x=1, y=3),
    )
    brain = Brain(mutation_chance=0.2)

    while True:
        agents: list[Agent] = []
        scores: list[tuple[int, int]] = []
        population = 100
        best_score: tuple[int, int] | None = None
        for id in range(population):
            agents.append(Agent(starting_pos=Position(0, 0), brain=brain, game=game))
            agents[id].run_actions()

            fitness_score = agents[id].calculate_fitness()
            scores.append((id, fitness_score))

            if best_score is None or fitness_score > best_score[1]:
                best_score = (id, fitness_score)

            if best_score[1] >= 10000:
                break
    # Step 6: If score > win_value in fitness: skip to Step 10. Else:
    # Step 7: Extract Brain from best agent
    # Step 8: Transplant best agent Brain into AgentHandler population
    # Step 9: Jump to step 4
    # Step 10:Output the best agent's path and score
    # Step 11:Exit


if __name__ == "__main__":
    main()
