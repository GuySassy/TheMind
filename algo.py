import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class QLearningAgent:
    def __init__(self, input_dim, output_dim, alpha=0.001, gamma=0.99, epsilon=0.1):
        self.q_network = QNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice([0, 1])  # Random action (exploration)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                return torch.argmax(self.q_network(state_tensor)).item()  # Best action (exploitation)

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)

        # Compute Q-value and target
        q_values = self.q_network(state_tensor)
        target = q_values.clone()

        if done:
            target[action] = reward
        else:
            next_q_values = self.q_network(next_state_tensor)
            target[action] = reward + self.gamma * torch.max(next_q_values)

        # Calculate loss and update Q-network
        loss = nn.MSELoss()(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
