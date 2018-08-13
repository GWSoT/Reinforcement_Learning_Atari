import keras


class Agent():
    def act(self, action):
        raise NotImplementedError
    
    def remember(self, state, reward, action, new_state, done):
        raise NotImplementedError
    
    def replay(self, batch_size):
        raise NotImplementedError

    