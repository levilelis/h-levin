import numpy as np
import math

# A simple uniform categorical policy
class SimplePolicy:
    def __init__(self, branch):
        self._number_actions = branch
        self._probs = [1./branch for i in range(branch)]
        self._log_probs = [math.log(p) for p in self._probs]


    def predict(self, image_representation):
        return np.array([[self._log_probs]]), np.array([[self._probs]])

    def get_number_actions(self):
        return self._number_actions
