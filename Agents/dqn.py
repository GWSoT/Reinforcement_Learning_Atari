from .agent import Agent
from collections import deque
import random
import numpy as np


class DQN(Agent):

    def __init__(self, a_size, s_size, model, batch_size, memory_size, epsilon, epsilon_min, epsilon_decay, gamma):
        self.a_size = a_size,
        self.s_size = s_size
        self.model = model
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
    
    def remember(self, state, reward, action, nextstate, done):
        self.memory.append((state, reward, action, nextstate, done))

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            return self.memory
        return random.sample(self.memory, batch_size)
    
    def act(self, state):
        if np.random.uniform() < self.epsilon:
            return random.randint(0, 1)
        actions = self.model.predict(state)
        return np.argmax(actions[0])

    def fit(self, batch_size):
        minibatch = self.sample(batch_size)
        for state, reward, action, next_state, done in minibatch:
            target = reward
        
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            target_f = self.model.predict(state)

            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        