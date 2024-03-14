import random
from utils.UER import *
import numpy as np
from utils.dlmodel import *

class DQNAgent:
    def __init__(self, st_size, action_size, name, role, arglist, gamma=0.95, alpha=0.2, epsilon=0.05):
        self.name = name
        self.st_size = st_size
        self.action_size = action_size
        self.role = role
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = 0.98
        self.epsilon_minimum = 0
        self.maxCapacity = arglist["maxCapacity"]
        self.batchSize = arglist["batch_size"]
        self.frequency = arglist["update_frequency"]
        self.current = 0

        self.memory = UER_memory(self.maxCapacity)
        self.dlmodel = DLModel(self,st_size, self.action_size, "DLModel", arglist)
    
    def epsilon_greedy(self, state):
        """
        Epsilon-greedy on action selection
        """
        value = random.randint(0, 1)
        if value < self.epsilon:
            return 
        else:
            return self.dlmodel.max_Q_action(state)

    def epsilon_decay(self):
        self.current += 1
        if self.epsilon > self.epsilon_minimum:
            self.epsilon = self.epsilon * self.epsilon_decay_rate
    
    def observeTransition(self, transition):
        self.memory.store_transition(transition)
    
    def y_i_update(self, batch_used):
        current_states = np.array([batch[0] for batch in batch_used])
        next_states = np.array([batch[3] for batch in batch_used])

        predict_current = self.dlmodel.predict(current_states)
        predict_next_target = self.dlmodel.predict(next_states, target=True)

        x = np.zeros((len(batch_used), self.st_size))
        y = np.zeros((len(batch_used), self.action_size))
        errors = np.zeros(len(batch_used))

        for i in range(len(batch_used)):
            cState = batch_used[0]
            action = batch_used[1]
            reward = batch_used[2]
            done = batch_used[4]

            t = predict_current[i]
            oldVal = t[action]
            if done:
                t[action] = reward
            else:
                t[action] = reward + self.gamma * np.max(predict_next_target[i])
        
            x[i] = cState
            y[i] = t
            errors[i] = np.abs(t[action] - oldVal)
        
        return [x, y, errors]
    
    def update_target(self):
        if self.current % self.frequency == 0:
            self.dlmodel.update_target()
    
    def replay(self):
        batch_used = self.memory.uniform_sample(self.batchSize)
        x, y = self.y_i_update(batch_used)
        self.dlmodel.train(x, y)
    

    


            

    
