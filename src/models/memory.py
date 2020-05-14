import numpy as np
import sys

class Trajectory():
    def __init__(self, states, actions, solution_costs, expanded):
        self._states = states
        self._actions = actions
        self._expanded = expanded
        self._solution_costs = solution_costs
        self._is_normalized = False
        
    def get_states(self):
        return self._states
    
    def get_actions(self):
        return self._actions
    
    def get_expanded(self):
        return self._expanded
    
    def get_solution_costs(self):
        return self._solution_costs
    
    def normalize_expanded(self, factor):
        if not self._is_normalized:
            self._expanded /= factor
            self._is_normalized = True

class Memory():
    def __init__(self):
        self._trajectories = []
        self._max_expanded = -sys.maxsize
        
    def add_trajectory(self, trajectory):
        if trajectory.get_expanded() > self._max_expanded:
            self._max_expanded = trajectory.get_expanded() 
        
        self._trajectories.append(trajectory)
        
    def shuffle_trajectories(self):
        self._random_indices = np.random.permutation(len(self._trajectories))
        
    def next_trajectory(self):     
        
        for i in range(len(self._trajectories)):
            traject = np.array(self._trajectories)[self._random_indices[i]]
            traject.normalize_expanded(self._max_expanded)
            yield traject
    
    def clear(self):
        self._trajectories.clear()
        self._max_expanded = -sys.maxsize