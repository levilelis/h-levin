import copy
import random

import numpy as np
import sys

class Trajectory():
    def __init__(self, states, actions, solution_costs, expanded, solution_pi=0.0):
        self._states = states
        self._actions = actions
        self._expanded = expanded
        self._non_normalized_expanded = expanded
        self._solution_costs = solution_costs
        self._solution_pi = solution_pi
        self._is_normalized = False
        
    def get_states(self):
        return self._states
    
    def get_actions(self):
        return self._actions
    
    def get_expanded(self):
        return self._expanded
    
    def get_solution_costs(self):
        return self._solution_costs
    
    def get_non_normalized_expanded(self):
        return self._non_normalized_expanded
    
    def get_solution_pi(self):
        return self._solution_pi
    
    def normalize_expanded(self, factor):
        if not self._is_normalized:
            self._expanded /= factor
            self._is_normalized = True

class Memory():
    def __init__(self):
        self._trajectories = []
        self._max_expanded = -sys.maxsize
        self._state_action_pairs = []
        self._preprocessed_pairs = []
        
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
            
    def number_trajectories(self):
        return len(self._trajectories)
    
    def merge_trajectories(self, other):
        for t in other._trajectories:
            self._trajectories.append(t)

    def clear(self):
        self._trajectories.clear()
        self._max_expanded = -sys.maxsize
        self._state_action_pairs.clear()

    def shuffle_state_action(self):
        self._state_action_pairs.clear()

        for t in self._trajectories:
            t.normalize_expanded(self._max_expanded)
            states = t.get_states()
            actions = t.get_actions()
            for i in range(len(states)):
                self._state_action_pairs.append([states[i], actions[i]])

        random.shuffle(self._state_action_pairs)

    def get_preprocessed_pairs(self):
        return self._preprocessed_pairs

    def preprocess_data(self):
        self.shuffle_state_action()

        swap_colors = [False, True]
        flip_up_down = [False, True]
        number_of_rotations = [0, 1, 2, 3]
        for swap in swap_colors:
            for flip in flip_up_down:
                for r in number_of_rotations:
                    for i in range(len(self._state_action_pairs)):
                        r_state = copy.deepcopy(self._state_action_pairs[i][0])
                        a = self._state_action_pairs[i][1]
                        for _ in range(r):
                            r_state.rotate90()
                            a = r_state.get_rotate90_action(a)
                        if flip:
                            r_state.flip_up_down()
                            a = r_state.get_flip_up_down_action(a)
                        if swap:
                            r_state.swap_colors()
                        self._preprocessed_pairs.append([r_state, a])
        self.clear()