from FederatedLearningManager import FederatedLearningManager
from Client import Client
from Environment import Environment
from DRLAgent import DQNAgent
import gymnasium as gym
import numpy as np
import logging
import time

NUM_CLIENTS = 5
NUM_EPOCHS = 3
NUM_EPISODES_PER_EPOCH = 10
NUM_TEST_EPISODES = 10

# Set up the logging
logging.basicConfig(filename='test.log', level=logging.INFO, format='%(asctime)s %(message)s')
logging.info('Starting the program!')

# Create the clients

environment_options = {
    'gravity': np.random.uniform(0.0005, 0.0050, size=NUM_CLIENTS),
}

logging.info('Creating the clients...')
clients = []
for i in range(NUM_CLIENTS):
    # Create the environments
    env = gym.make('MountainCar-v0', render_mode='human')
    # Create the deep RL agent
    agent = DQNAgent(env)
    client = Client(i, env, agent)
    clients.append(client)

# Create the federated learning manager
logging.info('Creating the federated learning manager...')
fl_manager = FederatedLearningManager(clients)

# Train the deep RL agent using federated learning
logging.info('Training the deep RL agent using federated learning...')
logging.info(f'Number of epochs: {NUM_EPOCHS}')
logging.info(f'Number of episodes per epoch: {NUM_EPISODES_PER_EPOCH}')
final_weights = fl_manager.train(num_epochs=NUM_EPOCHS, num_episodes_per_epoch=NUM_EPISODES_PER_EPOCH)

logging.info('Training complete!!')
logging.info(f'Final aggregated weights: {final_weights}')

# Save the trained deep RL agent
logging.info('Saving the trained deep RL agent...')
fl_manager.clients[0].agent.q_net.save_weights(f'./trained_models/agent_{NUM_CLIENTS}_c_{NUM_EPOCHS}_e_{NUM_EPISODES_PER_EPOCH}_epe_{time.asctime}.h5')

# Test the trained deep RL agent
logging.info('Testing the trained deep RL agent...')
env = gym.make('MountainCar-v0')
for i in range(NUM_TEST_EPISODES):
    logging.info(f'Testing episode: {i}')
    state, info = env.reset()
    done, truncated = False, False
    test_agent = DQNAgent(env)
    test_agent.q_net.set_weights(final_weights)
    steps = 0

    while not done and not truncated:
        steps += 1
        action = test_agent.policy(state)
        state, reward, done, truncated, info = env.step(action)
        logging.info(f'State: {state}, Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}')
        env.render()

    logging.info(f'Testing episode {i} completed after {steps} steps')

# Close the environment
env.close()
