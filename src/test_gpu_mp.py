import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
from multiprocessing import Lock
from multiprocessing.managers import BaseManager

import numpy as np
from concurrent.futures.process import ProcessPoolExecutor


class KerasModel():
    def __init__(self):
        self.mutex = Lock()
        self.model = None

    def initialize(self):
        import tensorflow
#         self.model = ConvNet((2, 2), 32, 4, 'CrossEntropyLoss')
        self.model = tensorflow.keras.Sequential(
            [tensorflow.keras.layers.Dense(1, activation='relu', input_shape=(10,)),
             ])

        self.model.compile('sgd', 'mse')

    def predict(self, arr):
        with self.mutex:
            return self.model.predict(arr)


class KerasManager(BaseManager):
    pass


KerasManager.register('KerasModel', KerasModel)

def test_func(data):
    x = np.random.random((1, 10))
    return data.predict(x)

if __name__ == '__main__':
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default = 6))
    
    with KerasManager() as manager:
        print('Main', os.getpid())
        kerasmodel = manager.KerasModel()
        kerasmodel.initialize()
        
        with ProcessPoolExecutor(max_workers = ncpus) as executor:
            args = ((kerasmodel) for _ in range(100)) 
            results = executor.map(test_func, args)
        
        for result in results:
            print(result)