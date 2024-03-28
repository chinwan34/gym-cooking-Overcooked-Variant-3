from collections import deque
import random

class UER_memory:

    def __init__(self, maxCapacity):
        self.maxCapacity = maxCapacity
        self.memory = deque(maxlen = self.maxCapacity)
    
    def store_transition(self, transition):
        self.memory.append(transition)
    
    def uniform_sample(self, size):
        if size > len(self.memory):
            return random.sample(self.memory, len(self.memory))
        else:
            return random.sample(self.memory, size)
    
