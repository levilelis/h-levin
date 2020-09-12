import argparse
import numpy as np
import copy
import os
import random
from domains.sliding_tile_puzzle import SlidingTilePuzzle
from os.path import join

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-filetrain', action='store', dest='file_train', 
                        default='train_puzzle_output', 
                        help='File where the generated training instances will be saved')
    
    parser.add_argument('-filetest', action='store', dest='file_test', 
                        default='test_puzzle_output', 
                        help='File where the generated test instances will be saved')
       
    parser.add_argument('-width', action='store', dest='width', 
                        default=4, 
                        help='Width of puzzles to be generated')
    
    parser.add_argument('-ntrain', action='store', dest='ntrain', 
                        default=1000, 
                        help='Number of training puzzles to be generated')
    
    parser.add_argument('-ntest', action='store', dest='ntest', 
                        default=1000, 
                        help='Number of test puzzles to be generated')
    
    parser.add_argument('-steps', action='store', dest='steps', 
                        default=10000, 
                        help='Number of steps performed backwards from the goal')
    
    parser.add_argument('-minsteps', action='store', dest='minsteps', 
                        default=50, 
                        help='Minimum number of steps performed backwards from the goal')
    
    parameters = parser.parse_args()
    
    if not os.path.exists(parameters.file_train):
        os.makedirs(parameters.file_train)
        
    if not os.path.exists(parameters.file_test):
        os.makedirs(parameters.file_test)
    
    width = int(parameters.width)
    ntrain = int(parameters.ntrain)
    ntest = int(parameters.ntest)
    steps = int(parameters.steps)
    minsteps = int(parameters.minsteps)
    
    test_instances = set()
    train_instances = set()
    
    tiles = [i for i in range(0, width * width)]
    goal = SlidingTilePuzzle(tiles)
    
    # generating training instances
    for j in range(ntrain):
        state = copy.deepcopy(goal)
        
        number_steps = random.randint(minsteps, steps)
        for _ in range(number_steps):
            actions = state.successors()
            random_index = random.randint(0, len(actions) - 1)
            random_action = actions[random_index]
            state.apply_action(random_action)
        
        state.save_state(join(parameters.file_train, 'puzzle_' + str(j + 1)))
        train_instances.add(copy.deepcopy(state))
    
    # generating test instances
    j = 0
    while len(test_instances) < ntest:
        tiles = [i for i in range(0, width * width)]
        np.random.shuffle(tiles)
         
        state = SlidingTilePuzzle(tiles)
        if state.is_valid() and state not in test_instances and state not in train_instances:
            state.save_state(join(parameters.file_test, 'puzzle_' + str(j + 1)))
            test_instances.add(copy.deepcopy(state))
            j += 1
            
if __name__ == "__main__":
    main()
