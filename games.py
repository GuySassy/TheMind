import gym
import numpy as np
import time


class MultiAgentRevealEnv(gym.Env):
    def __init__(self, n_agents, max_time):
        super(MultiAgentRevealEnv, self).__init__()

        self.n_agents = n_agents
        self.max_time = max_time
        self.start_time = time.time()
        # Action space: [0, 1] (0 = don't reveal, 1 = reveal)
        self.action_space = gym.spaces.MultiDiscrete([2] * n_agents)

        self.cards_sampler = gym.spaces.Box(low=0, high=100, shape=(n_agents,), dtype=np.int32)
        # State space: [time_passed, special_integer, reveal_flag]
        self.observation_space = gym.spaces.Dict({
            "time_passed": gym.spaces.Box(low=0.0, high=max_time, shape=(1,), dtype=np.float32),
            "special_integer": gym.spaces.Box(low=0, high=100, shape=(n_agents,), dtype=np.int32),
            "reveal_flags": gym.spaces.MultiBinary(n_agents)
        })

        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.time_passed = 0.0
        self.reveal_flags = np.zeros(self.n_agents, dtype=bool)
        self.special_integers = self.cards_sampler.sample()
        self.goal_state_reached = False
        return self._get_observation()

    def _get_observation(self):
        # State space: [time_passed, special_integer, reveal_flag]
        return {
            "time_passed": np.array([time.time()-self.start_time], dtype=np.float32),
            "special_integer": self.special_integers,
            "reveal_flags": self.reveal_flags
        }

    def step(self, actions):

        rewards = np.zeros(self.n_agents, dtype=np.float32)

        # Find the smallest unrevealed integer
        unrevealed_integers = self.special_integers[~self.reveal_flags]
        if len(unrevealed_integers) > 0:
            min_unrevealed = np.min(unrevealed_integers)
        else:
            min_unrevealed = None

        for i, action in enumerate(actions):
            if action == 1 and not self.reveal_flags[i]:  # Reveal action
                self.reveal_flags[i] = True

                if self.special_integers[i] == min_unrevealed:
                    rewards[i] = 25  # Correct reveal
                else:
                    rewards[i] = -25  # Wrong reveal

        # Check if all integers are revealed
        if np.all(self.reveal_flags):
            self.goal_state_reached = True
            rewards += 50  # Additional reward for reaching goal state

        done = self.goal_state_reached or self.time_passed >= self.max_time
        return self._get_observation(), rewards, done, {}

    def render(self, mode='human'):
        print(f"Time: {self.time_passed}, Revealed: {self.reveal_flags}, Special integers: {self.special_integers}")