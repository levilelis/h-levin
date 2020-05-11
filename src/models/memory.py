import numpy as np

class Trajectory():
    def __init__(self, states, actions, f_values):
        self._states = states
        self._actions = actions
        self._f_values = f_values
        
    def get_states(self):
        return self._states
    
    def get_actions(self):
        return self._actions
    
    def get_f_values(self):
        return self._f_values


class Memory():
    def __init__(self):
        self._trajectories = []
#         self._trajectories_actions = []
#         self._trajectories_f_values = []
        
    def add_trajectory(self, trajectory):
        self._trajectories.append(trajectory)
#         self._trajectories_actions.append(actions)
#         self._trajectories_f_values.append(f_values)
        
    def next_trajectory(self):
        random_indices = np.random.permutation(len(self._trajectories))

        for i in range(len(self._trajectories)):
            traject = np.array(self._trajectories)[random_indices[i]]
#             y = np.array(self._trajectories_actions)[random_indices[i]]
#             f = np.array(self._trajectories_f_values)[random_indices[i]]

            yield traject