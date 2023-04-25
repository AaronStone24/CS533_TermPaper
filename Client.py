import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.optimizers import Optimizer
from DRLAgent import DQNAgent
import gymnasium as gym

class Client:
    def __init__(self, id: int, environment: gym.Env, agent: DQNAgent):
        self.id = id
        self.environment = environment
        self.agent = agent

    def _create_model(self):
        # Create the local model
        pass

    def _create_optimizer(self):
        # Create the local optimizer
        pass

    def train(self, num_episodes):
        # Train the local model
        self.agent.train(num_episodes)

    def get_weights(self):
        # Get the weights of the local model
        return self.agent.q_net.get_weights()
        

    def set_weights(self, weights) -> None:
        # Set the weights of the local model
        self.agent.q_net.set_weights(weights)
        self.agent.target_q_net.set_weights(weights)