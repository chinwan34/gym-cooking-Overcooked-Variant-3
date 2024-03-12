import numpy as np
import random
from utils.DQNagent import DQNAgent

class mainAlgorithm:
    def __init__(self, environment, num_training, max_timestep):
        self.environment = environment
        self.num_training = num_training
        self.max_timestep = max_timestep

    def run(self, agents):
        for episodes in range(self.num_training):
            state = self.environment.reset()
            
        
            done = False
            step = 0
            action_dict = []

            while not done and step < self.max_timestep:
                for agent in agents:
                    action_dict.append(agent.epsilon_greedy(state))
                
                next_state, reward, done, info = self.environment.step(action_dict)
                next_state = np.array(next_state)
                # next_state = next_state.ravel()

                for agent in agents:
                    agent.observeTransition((state, action_dict, reward, next_state, done))
                    
