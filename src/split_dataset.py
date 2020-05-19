import sys
import os
import numpy as np

from os import listdir
from os.path import isfile, join

from domains.witness import WitnessState

def main():
    if len(sys.argv[1:]) < 1:
        print('Usage: split_dataset.py <name folder>')
        return
    
    folder = sys.argv[1]
    folder = folder.replace('/', '')
    folder_test = folder + '_test'
    folder_train = folder + '_train'
    
    path_to_problems = 'problems/witness/'
    
    if not os.path.exists(join(path_to_problems, folder_test)):
        os.makedirs(join(path_to_problems, folder_test))
         
    if not os.path.exists(join(path_to_problems, folder_train)):
        os.makedirs(join(path_to_problems, folder_train))
    
    states_4x4 = {}    
    states_5x5 = {}
    puzzle_files_4x4 = []
    puzzle_files_5x5 = []
    puzzle_files = [f for f in listdir(join(path_to_problems, folder)) if isfile(join(path_to_problems, folder, f))]
    
    
    for file in puzzle_files:
        if '.' in file:
            continue
        s = WitnessState()
        s.read_state(join(path_to_problems, folder, file))
        
        if s._lines == 4:
            states_4x4[file] = s
            puzzle_files_4x4.append(file)
        else:
            states_5x5[file] = s
            puzzle_files_5x5.append(file)
        
    random_indices = np.random.permutation(len(puzzle_files_4x4))
    
    for i in range(len(random_indices)):
        if i < len(random_indices) / 2:
            states_4x4[puzzle_files_4x4[random_indices[i]]].save_state(join(path_to_problems, folder_test, puzzle_files_4x4[random_indices[i]]))
        else:
            states_4x4[puzzle_files_4x4[random_indices[i]]].save_state(join(path_to_problems, folder_train, puzzle_files_4x4[random_indices[i]]))
    
    random_indices = np.random.permutation(len(puzzle_files_5x5))
    
    for i in range(len(random_indices)):
        if i < len(random_indices) / 2:
            states_5x5[puzzle_files_5x5[random_indices[i]]].save_state(join(path_to_problems, folder_test, puzzle_files_5x5[random_indices[i]]))
        else:
            states_5x5[puzzle_files_5x5[random_indices[i]]].save_state(join(path_to_problems, folder_train, puzzle_files_5x5[random_indices[i]]))
        
if __name__ == "__main__":
    main()