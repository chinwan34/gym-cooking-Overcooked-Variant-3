import random
from utils.UER import *

class DQNAgent:
    def __init__(self, name, role, maxCapacity, batchSize, gamma=0.95, alpha=0.2, epsilon=0.05):
        self.name = name
        self.role = role
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = 0.98
        self.epsilon_minimum = 0
        self.maxCapacity = maxCapacity
        self.batchSize = batchSize

        self.memory = UER_memory(maxCapacity)
    
    def epsilon_greedy(self, state):
        """
        Epsilon-greedy on action selection
        """
        value = random.randint(0, 1)
        if value < self.epsilon:
            return 

    def epsilon_decay(self):
        if self.epsilon > self.epsilon_minimum:
            self.epsilon = self.epsilon * self.epsilon_decay_rate
    
    def observeTransition(self, transition):
        self.memory.store_transition(transition)
    
    def y_i_update(self, batch_used):
        
    
    def replay(self):
        batch_used = self.memory.uniform_sample(self.batchSize)

    


            

    
