
from abc import ABC

class Environment(ABC):
    
    def successor(self):
        pass
    
    def is_solution(self):
        pass
    
    def apply_action(self):
        pass