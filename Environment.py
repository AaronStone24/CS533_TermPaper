import gymnasium as gym

class Environment:
    def __init__(self, env_name: str):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        # Reset the environment and return the initial observation
        pass

    def step(self, action):
        # Take a step in the environment and return the observation, reward, done, and info
        pass