
from abc import ABC

class Environment(ABC):
    
    def successors(self):
        pass
    
    def is_solution(self):
        pass
    
    def apply_action(self, action):
        pass
    
    def get_image_representation(self):
        pass
    
    def heuristic_value(self):
        pass
    
    def reset(self):
        pass
    
    def copy(self):
        pass