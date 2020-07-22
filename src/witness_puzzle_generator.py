import argparse
import numpy as np
import copy
import os
import time
from concurrent.futures.process import ProcessPoolExecutor

from domains.witness import WitnessState
import random

class PuzzleGenerator:
    
    def fill_region(self, state, region, color, prob):
        """
        Fills the percentage specified in variable prob of grid cells with bullets of the color 
        specified in the variable color.  
        """
        filled_with_color = False
        for i in range(0, len(region)):
            random_number = random.random()
                        
            if random_number <= prob:
                state.add_color(region[i][0], region[i][1], color)   
                filled_with_color = True
        return filled_with_color
        
    def generate_random_path(self, lines, columns, line_init, column_init, line_goal, column_goal):
        """
        Generates a solution path for a grid with starting position at (line_init, column_init)
        and finishing position at (line_goal, column_goal). The grid is defined by a grid of size
        lines x columns.

        Returns a GameState instance representing the path.
        """
        while True:
            state = WitnessState(lines, columns, line_init, column_init, line_goal, column_goal)
            actions = state.successors()
            while actions:
                rand_action = random.randint(0, len(actions)-1)
                state.apply_action(actions[rand_action])
                if state.has_tip_reached_goal():
                    return state
                actions = state.successors()
    
    
    def generate_puzzles_of_size(self, input_data):
        
        size = input_data[0]
        minimum_number_regions = input_data[1]
        color_bullets = input_data[2]
        bullet_probability = input_data[3]
          
        states = []
        for i in range(1, size[1] + 1):
            #lines, columns, line_init, column_init, line_goal, column_goal
            states.append(self.generate_random_path(size[0], size[1], 0, 0, 0, i))
            states.append(self.generate_random_path(size[0], size[1], 0, 0, size[0], i))
        for i in range(1, size[0] + 1):
            #lines, columns, line_init, column_init, line_goal, column_goal
            states.append(self.generate_random_path(size[0], size[1], 0, 0, i, 0))
            states.append(self.generate_random_path(size[0], size[1], 0, 0, i, size[1]))
        
#             np.random.shuffle(states)
        filled_states = []

        for state in states:
            regions = state.partition_cells()
            
            if len(regions) < minimum_number_regions:
                continue
            
            used_colors = set()
            for region in regions:
                color = np.random.randint(1, high=len(color_bullets) + 1)
                while size[0] > 2 and color in used_colors and len(used_colors) < len(color_bullets):
                    color = np.random.randint(1, high=len(color_bullets)+1)
                if self.fill_region(state, region, color, bullet_probability):
                    used_colors.add(color)
                    
            if len(used_colors) < 2:
                continue  
            
            filled_states.append(state)                         
        
        return filled_states
                
    def generate_puzzles_with_random_paths(self, puzzle_size, bullet_probability, color_bullets, time_limit, puzzle_folder, number_puzzles, ncpus):
        """
        Generates a set of puzzles for the sizes given in puzzle_sizes. Currently the puzzles
        have their starting position at (0, 0) and any finishing position at the edge of the grid. 
        
        All generated puzzles are saved in a folder determined by variable puzzle_folder. 
        """
        start_time = time.time()
        
        puzzles_generated = set()
        minimum_number_regions = 2
        if puzzle_size[0] == 3:
            minimum_number_regions = 2   
        if puzzle_size[0] >= 4:
            minimum_number_regions = 4
        if puzzle_size[0] >= 10:
            minimum_number_regions = 5                
        
        while time.time() - start_time < time_limit - 10 and len(puzzles_generated) < number_puzzles:
            
            with ProcessPoolExecutor(max_workers = ncpus) as executor:
                args = ((puzzle_size, minimum_number_regions, color_bullets, bullet_probability) for _ in range(10 * ncpus)) 
                results = executor.map(self.generate_puzzles_of_size, args)
                
                for result in results:
                    puzzles = result
                
                    for puzzle in puzzles:
                        puzzle.clear_path()
                    
                        if puzzle not in puzzles_generated:
                            puzzles_generated.add(puzzle)        
        puzzle_id = 1
        for puzzle in puzzles_generated:
            puzzle.save_state(puzzle_folder + '/' + str(puzzle_size[0]) + 'x' + str(puzzle_size[1]) + '_' + str(puzzle_id))
            
            if puzzle_id == number_puzzles:
                break
            
            puzzle_id += 1
            
        print('Generated ', len(puzzles_generated), ' puzzles in ', time.time() - start_time, ' seconds')

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-folder', action='store', dest='puzzle_folder', 
                        default='test_puzzle_output', 
                        help='Folder where the generated instances will be saved')
    
    parser.add_argument('-time', action='store', dest='time_limit', 
                        default=300, 
                        help='Time limit in seconds for generating instances')
    
    parser.add_argument('-colors', action='store', dest='colors', 
                        default=2, 
                        help='Number of different colors to be used')
    
    parser.add_argument('-l', action='store', dest='lines', 
                        default=1, 
                        help='Number of lines in puzzles to be generated')
    
    parser.add_argument('-c', action='store', dest='columns', 
                        default=4, 
                        help='Number of columns in puzzles to be generated')
    
    parser.add_argument('-p', action='store', dest='bullet_probability', 
                        default=0.6, 
                        help='Probability of placing a bullet in an empty cell')
    
    parser.add_argument('-n', action='store', dest='number_puzzles', 
                        default=1000, 
                        help='Number of puzzles to be generated')
    
    parameters = parser.parse_args()
    
    if not os.path.exists(parameters.puzzle_folder):
        os.makedirs(parameters.puzzle_folder)
        
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default = 2))
    
    color_bullets = [i for i in range(1, int(parameters.colors) + 1)]
    
    generator = PuzzleGenerator()
    generator.generate_puzzles_with_random_paths((int(parameters.lines), int(parameters.columns)), 
                                                 float(parameters.bullet_probability), 
                                                 color_bullets, 
                                                 int(parameters.time_limit), 
                                                 parameters.puzzle_folder,
                                                 int(parameters.number_puzzles),
                                                 ncpus)
            
if __name__ == "__main__":
    main()
