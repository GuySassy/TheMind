import numpy as np
import torch


def evaluate_q_network(agents, env, n_games=200, max_steps=50):
    """
    Evaluates the performance of a Q-network by simulating a series of games.

    Parameters:
        q_network (QNetwork): The trained Q-network to evaluate.
        env (gym.Env): The environment instance for the game.
        n_games (int): The number of game episodes to evaluate.
        max_steps (int): Maximum steps per game to avoid infinite loops.

    Returns:
        success_rate (float): The proportion of games that were successfully completed.
    """
    success_count = 0  # Counter for successful games

    for game in range(n_games):
        state = env.reset()  # Reset environment for a new game
        game_success = False  # Track if this game ends in success

        for step in range(max_steps):
            actions = []
            for i in range(env.n_agents):
                # Prepare state for the agent
                agent_state = np.concatenate((state['time_passed'],
                                              [state['special_integer'][i]],
                                              [state['reveal_flags'][i]]))
                action = agents[i].choose_action(agent_state, eval=True)
                actions.append(action)

            next_state, rewards, done, _ = env.step(actions)

            # Check if the game was completed successfully
            if done and not torch.any(rewards < 0):  # Positive rewards indicates a successful game end
                game_success = True
                break

            # Move to the next state
            state = next_state

        # Count the game as successful if it ended with all integers revealed in the correct order
        if game_success:
            success_count += 1

    # Calculate the success rate over the N games
    success_rate = success_count / n_games
    print(f"Success Rate: {success_count}/{n_games} ({success_rate * 100:.2f}%)")

    return success_rate
