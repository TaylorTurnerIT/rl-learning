from rl_learning.game import Game, Position


def main():
    # Step 1: Create Game
    map_x_size = 5
    map_y_size = 5
    pit_pos = Position(x=1, y=5)
    wumpus_pos = Position(x=0, y=0)
    win_pos = Position(x=1, y=3)
    Game(
        x_size=map_x_size,
        y_size=map_y_size,
        pit_pos=pit_pos,
        wumpus_pos=wumpus_pos,
        win_pos=win_pos,
    )
    # Step 2: Create Brain
    # Step 3: Create AgentHandler with Population and no Brain
    # Step 4: For each agent: set Agent brain
    # Step 4: For each agent: run Agent actions
    # Step 5: For each agent: get fitness score
    # Step 6: If score > win_value in fitness: skip to Step 10. Else:
    # Step 7: Extract Brain from best agent
    # Step 8: Transplant best agent Brain into AgentHandler population
    # Step 9: Jump to step 4
    # Step 10:Output the best agent's path and score
    # Step 11:Exit
    pass


if __name__ == "__main__":
    main()
