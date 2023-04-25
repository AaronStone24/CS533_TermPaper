import tensorflow as tf
from keras import Model
from keras.layers import Dense, Input, LeakyReLU
from keras.optimizers import Adam
import random
import numpy as np
import typing as tp
import gymnasium as gym
import logging

class DRLAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.model = self._create_model()

    def _create_model(self):
        # Create the deep neural network model
        model = tf.keras.Sequential()
        model.add(Dense(64, activation='relu', input_shape=self.observation_space.shape))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_space.n, activation='softmax'))
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def get_model(self):
        # Get the deep neural network model
        return self.model

    def predict(self, state):
        # Predict the action to take given the current state
        pass

    def train(self):
        # Train the deep neural network given the current state and target
        pass


class DQNAgent:
    input_shape = []
    output_shape = []
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.001
    c = 5

    def __init__(self, environment: gym.Env):
        self.env = environment
        if isinstance(self.env.observation_space, gym.spaces.Box):
            DQNAgent.input_shape = self.env.observation_space.shape
        elif isinstance(self.env.observation_space, gym.spaces.Discrete):
            DQNAgent.input_shape = (self.env.observation_space.n,)
        if isinstance(self.env.action_space, gym.spaces.Box):
            DQNAgent.output_shape = self.env.action_space.shape
        elif isinstance(self.env.action_space, gym.spaces.Discrete):
            DQNAgent.output_shape = (self.env.action_space.n,)
        logging.info(f'Input shape: {DQNAgent.input_shape}')
        logging.info(f'Output shape: {DQNAgent.output_shape}')
        self.observation, self.info = self.env.reset()
        self.memory = []
        self.episode_max_len = 100
        self.optimizer = Adam(learning_rate=DQNAgent.learning_rate)
        DQNAgent.model = self._build_dqn_model(DQNAgent.input_shape, DQNAgent.output_shape)
        self.q_net = self._build_dqn_model(DQNAgent.input_shape, DQNAgent.output_shape)    # TODO: Add the input_shape parameter
        self.target_q_net = self._build_dqn_model(DQNAgent.input_shape, DQNAgent.output_shape)

    # @classmethod
    # def get_model(cls):
    #     return cls.model

    def _build_dqn_model(self, input_shape, output_shape):
        q_network = tf.keras.Sequential()
        q_network.add(Input(shape=input_shape))
        q_network.add(Dense(16))
        q_network.add(LeakyReLU(alpha=0.05))
        q_network.add(Dense(24))
        q_network.add(LeakyReLU(alpha=0.05))
        q_network.add(Dense(8))
        q_network.add(LeakyReLU(alpha=0.05))
        q_network.add(Dense(np.prod(output_shape), activation="linear"))
        q_network.compile(loss="mse", optimizer=self.optimizer)

        return q_network

    def add_experience(self, S, A, R, S_dash, done):
        # state=state[0]
        self.memory.append((S, A, R, S_dash, done))

    def policy(self, S):    #Policy
        # print((state))
        # print(type(tf.convert_to_tensor(np.expand_dims(state, axis=0))))
        if np.random.rand() <= DQNAgent.epsilon:
            return np.random.choice(np.prod(DQNAgent.output_shape))

        S = np.reshape(S,(1,-1))
        action_values = self.q_net.predict(tf.convert_to_tensor(S))
        return np.argmax(action_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([transition[0] for transition in minibatch]).reshape(batch_size, np.prod(DQNAgent.input_shape))
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch]).reshape(batch_size, np.prod(DQNAgent.input_shape))
        # print(np.shape(next_states))
        dones = np.array([transition[4] for transition in minibatch]).reshape(-1, 1)

        q_values_next = self.target_q_net.predict(next_states)
        targets = rewards + DQNAgent.gamma * np.max(q_values_next, axis=1) * (1 - dones.squeeze())
        # print(np.shape(targets))
        # print(tf.shape(tf.convert_to_tensor(states)))
        targets_full = self.q_net.predict(tf.convert_to_tensor(states))

        targets_full[np.arange(batch_size), actions] = targets
        history = self.q_net.fit(tf.convert_to_tensor(states), targets_full, epochs=1)
        absolute_errors = np.abs(history.history['loss'])
        
        if DQNAgent.epsilon > DQNAgent.epsilon_min:
            DQNAgent.epsilon *= DQNAgent.epsilon_decay
        return absolute_errors

    def train(self, num_episodes=250):
        total_rewards = 0.0
        avg_reward_per_epidsode=[]
        avg_q=[]
        abs_errors=[]
        for i in range(num_episodes):
            logging.info(f"Episode {i}:")
            S = self.env.reset()
            S = S[0]
            done = False
            episode_reward = 0.0
            error = []
            # q_val=[]
            step = 0
            truncated = False
            while not done and not truncated:
                logging.info(f"Step {step}:")
                A = self.policy(S)
                S_dash, R, done, truncated, _ = self.env.step(A)
                step += 1
                logging.info(f"State: {S}, Action: {A}, Reward: {R}, Next State: {S_dash}, Done: {done}, Truncated: {truncated}")
                
                # print(step) #Store experience 
                self.add_experience(S, A, R, S_dash, done)
                if isinstance(R, float):
                    episode_reward += R
                S = S_dash

                # if step > self.episode_max_len:
                #   done = True

                if step % DQNAgent.c == 0: 
                    logging.info(f"Copying the q_net to target_q_net")
                    self.target_q_net = self.q_net

                if done or truncated:
                    if truncated:
                        logging.info(f"Episode {i} truncated after {step} steps")
                    else:
                        logging.info(f"Episode {i} finished after {step} steps")
                    total_rewards += episode_reward
                    avg_reward_per_epidsode.append(total_rewards/(i+1))
                    logging.info(f"Average reward per episode: {total_rewards/(i+1)}, total reward: {total_rewards}")
                    S = self.env.reset()
                    S = S[0]
                    S = np.reshape(S, (1,-1))

                    q_val= self.target_q_net.predict(tf.convert_to_tensor(S))

                    avg_q.append(np.mean(q_val))

                    if (i+1)%10==0:
                      print("episode: {}/{}, score: {}".format(i + 1, num_episodes, episode_reward))
                      logging.info("episode: {}/{}, score: {}".format(i + 1, num_episodes, episode_reward))
                    break

                if len(self.memory) > 32:
                    abs_err=self.replay(32)
                    error.append(abs_err)
                    # print(np.mean(q_val,axis=1).shape

            if(len(error)==0):
              abs_errors.append(0)
            else:
              abs_errors.append(np.mean(error))
            
        # print(np.shape(q_val_ep))
        return avg_reward_per_epidsode, avg_q, abs_errors