from games import MultiAgentRevealEnv
from algo import QLearningAgent
import numpy as np
from evaluation import evaluate_q_network
import matplotlib.pyplot as plt
import torch

def plot_loss(loss_tensor):
    """
    Plots the loss for each episode during training.

    Parameters:
        loss_tensor (torch.FloatTensor): A tensor containing the loss for each episode.
    """
    # Convert the tensor to a NumPy array for easy plotting
    loss_values = loss_tensor.detach().cpu().numpy() if loss_tensor.is_cuda else loss_tensor.detach().numpy()

    # Plot the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label="Loss per Episode", color='blue', linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss per Episode During Training")
    plt.legend()
    plt.grid(True)
    plt.show()

env = MultiAgentRevealEnv(n_agents=3, max_time=50)
n_episodes = 1000

# Initialize agents
agents = [QLearningAgent(input_dim=3, output_dim=2) for _ in range(env.n_agents)]  # 3 state dimensions, 2 actions
losses = torch.zeros(n_episodes)

for episode in range(n_episodes):
    state = env.reset()
    done = False
    losses_mean = 0
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
            # perform train step on agent i
            losses_mean += agents[i].update(agent_state, actions[i], rewards[i], next_agent_state, done)
        losses[i] = losses_mean
        state = next_state

    if episode % 100 == 0:
        eval_score = evaluate_q_network(agents, env)
        print(f"Episode {episode} finished with score: {eval_score}")

plot_loss(losses)

