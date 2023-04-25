import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.optimizers import Optimizer
from Client import Client
import numpy as np
import logging

class FederatedLearningManager:
    def __init__(self, clients: list[Client]):
        self.clients = clients

    def _train_clients(self):
        # Train the local models on each client
        pass

    def _aggregate_weights(self, weights):
        # Aggregate the weights of the local models
        return np.mean(weights, axis=0)

    def _update_global_model(self):
        # Update the global model with the aggregated weights
        pass

    def train(self, num_epochs, num_episodes_per_epoch):
        new_weights = None
        for epoch in range(num_epochs):
            logging.debug(f'Epoch: {epoch}')
            weights = []
            for client in self.clients:
                logging.debug(f'Training client with id: {client.id}')
                client.train(num_episodes_per_epoch)
                weight = client.get_weights()
                logging.info(f'Client with id: {client.id} has weights: {weight} in epoch {epoch}')
                weights.append(weight)
            
            new_weights = self._aggregate_weights(weights)
            logging.info(f'New aggregated weights: {new_weights} in epoch {epoch}')
            for client in self.clients:
                client.set_weights(new_weights)

        return new_weights

    def _evaluate_global_model(self):
        # Evaluate the global model by running it on the environment
        pass