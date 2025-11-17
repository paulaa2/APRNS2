import random
import numpy as np

class ReplayBuffer(object):
    """A replay buffer as commonly used for off-policy Q-Learning methods."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def put(self, obs, action, reward, next_obs, terminated, truncated):
        """Put a tuple of (obs, action, rewards, next_obs, terminated) into the replay buffer.
        The max length specified by capacity should never be exceeded. 
        The oldest elements inside the replay buffer should be overwritten first.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs, action, reward, next_obs, terminated, truncated)
        self.position = (self.position + 1) % self.capacity


    def get(self, batch_size):
        """Gives batch_size random samples from the replay buffer."""
        batch = random.sample(self.buffer, min(len(self.buffer),batch_size))
        state, action, reward, next_state, terminated, truncated = map(np.stack, zip(*batch))
        return state, action, reward, next_state, terminated, truncated
    
    def __len__(self):
        """Returns the number of tuples inside the replay buffer."""
        return len(self.buffer)