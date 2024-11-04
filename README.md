# TheMind

**TheMind** is a multi-agent environment simulation built using OpenAI's Gym framework, where agents collaborate to reveal their cards in ascending order. The project demonstrates the application of reinforcement learning (RL) with Q-learning in a cooperative game setting.

## Project Overview

In this project, multiple agents are trained using a Q-learning algorithm to work together in a cooperative environment. Each agent has a unique integer card, and the goal is to reveal all cards in ascending order from lowest to highest. Agents must learn to coordinate and reveal their cards at the right time to maximize the team’s score.

## Environment Setup and Game Mechanics

### State Space
The state space for each agent includes:
- **Time Passed**: A continuous variable representing the time that has passed in the game.
- **Unique Integer**: Each agent has a unique integer value, which must be revealed in the correct order relative to other agents.
- **Reveal Status Flag**: A boolean flag indicating whether an agent has already revealed its card.

### Action Space
Each agent has one action:
- **Reveal**: The agent chooses to reveal its integer.

### Reward Structure
The game uses a state-based reward function:
- **Correct Reveal**: If an agent reveals its integer at the correct time (it’s the lowest among unrevealed values), it receives a reward of +25.
- **Incorrect Reveal**: If an agent reveals its integer out of order, it receives a penalty of -25.
- **Goal State**: If all integers are revealed, the team receives a bonus reward of +50.

This reward structure encourages agents to work together to reveal their integers in the right order.

## Q-Learning Agent with Neural Network

Since the state space includes a continuous variable (time passed), a neural network is used to approximate the Q-function. The Q-learning agent uses a two-layer fully connected neural network to estimate Q-values for each possible action. This network enables the agent to learn and act within a continuous state space effectively.

### Neural Network Architecture
The Q-function approximation network includes:
- **Input Layer**: Accepts the continuous state space inputs.
- **Hidden Layers**: Two fully connected layers with ReLU activation for nonlinear transformation.
- **Output Layer**: Produces Q-values for each action (reveal or no-action).

## Installation

1. Clone this repository:
   ```bash
   git clone git@github.com:username/TheMind.git
   cd TheMind
