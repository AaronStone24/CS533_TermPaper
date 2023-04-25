import gymnasium as gym
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv

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

# class CustomMountainCarEnv(MountainCarEnv):
#     def __init__(self):
#         super(CustomMountainCarEnv, self).__init__()