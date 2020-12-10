from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from models.conv_net import ConvNet, TwoHeadedConvNet, HeuristicConvNet

class KerasModel():    
    def __init__(self):
        self.mutex = Lock()
        self.model = None

    def initialize(self, loss_name, search_algorithm, two_headed_model=False):
        if (search_algorithm == 'Levin' 
            or search_algorithm == 'LevinMult' 
            or search_algorithm == 'LevinStar' 
            or search_algorithm == 'PUCT'):
            if two_headed_model:
                self.model = TwoHeadedConvNet((2, 2), 32, 4, loss_name)
            else:
                self.model = ConvNet((2, 2), 32, 4, loss_name)
        if search_algorithm == 'AStar' or search_algorithm == 'GBFS':
                self.model = HeuristicConvNet((2, 2), 32, 4)
                
    def predict(self, x):
        with self.mutex:
            return self.model.predict(x)
        
    def train_with_memory(self, memory):
        return self.model.train_with_memory(memory)
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
        
    def load_weights(self, filepath):
        self.model.load_weights(filepath).expect_partial()

class KerasManager(BaseManager):
    pass