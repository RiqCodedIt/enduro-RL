from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        # self.memory = deque([], maxlen=capacity)
        self.memory = []
        self.capacity = capacity
        self.pointer = 0  # Track the next position to overwrite if needed

    def push(self, *args):
        """Save a transition"""
        if len(self.memory) < self.capacity:
            # If buffer is not full, add new transition normally
            self.memory.append(None)
        
        # If buffer is full, overwrite the oldest element
        self.memory[self.pointer] = Transition(*args)
        # Update the pointer
        self.pointer = int((self.pointer + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*batch))
        return batch

    def __len__(self):
        return len(self.memory)