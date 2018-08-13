from .agent import Agent
from collections import deque
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Lambda, Input, Add, Flatten
import random
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
from preprocess import preprocess_img
import copy 

class DDQN(Agent):

    def __init__(self, 
        q_network: Model, 
        target_network: Model, 
        memory_length: int,
        gamma,
        epsilon_decay,
        epsilon,
        action_size):
        self.num_actions = action_size
        self.q_network = q_network
        self.target_network = target_network
        self.memory_length = memory_length
        self.memory = deque(maxlen=self.memory_length)
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon

    def act(self, state):
        if random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_network.predict(np.expand_dims(state, axis=0))[0])

        if random.random() < self.epsilon:
            self.epsilon *= self.epsilon_decay
        
        return action 

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, new_state, done):
        next_state = np.concatenate((state[1:], np.expand_dims(new_state, axis=0)))
        self.memory.append((state, action, reward, next_state, done))
        return next_state

    def fit(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        x_batch = []
        y_batch = []

        for state, action, reward, next_state, done in minibatch:
            x_batch.append(state)

            target = self.q_network.predict(np.expand_dims(state, axis=0))

            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.max(self.q_network.predict(np.expand_dims(next_state, axis=0))[0])
            y_batch.append(target)
            self.q_network.fit(np.expand_dims(state, axis=0), target, verbose=0, epochs=1)