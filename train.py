from games import MultiAgentRevealEnv
from algo import QLearningAgent
import numpy as np
env = MultiAgentRevealEnv(n_agents=3, max_time=50)
n_episodes = 1000

# Initialize agents
agents = [QLearningAgent(input_dim=3, output_dim=2) for _ in range(env.n_agents)]  # 3 state dimensions, 2 actions

for episode in range(n_episodes):
    state = env.reset()
    done = False

    while not done:
        actions = []
        for i in range(env.n_agents):
            # Prepare state for the agent
            agent_state = np.concatenate((state['time_passed'],
                                          [state['special_integer'][i]],
                                          [state['reveal_flags'][i]]))
            action = agents[i].choose_action(agent_state)
            actions.append(action)

        next_state, rewards, done, _ = env.step(actions)

        for i in range(env.n_agents):
            agent_state = np.concatenate((state['time_passed'],
                                          [state['special_integer'][i]],
                                          [state['reveal_flags'][i]]))
            next_agent_state = np.concatenate((next_state['time_passed'],
                                               [next_state['special_integer'][i]],
                                               [next_state['reveal_flags'][i]]))
            agents[i].update(agent_state, actions[i], rewards[i], next_agent_state, done)

        state = next_state

    if episode % 100 == 0:
        print(f"Episode {episode}: Finished")
