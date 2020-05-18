from domains.witness import WitnessState
from collections import deque
import itertools, random
import numpy as np
import copy
import os
import sys

class PuzzleGenerator:
    """
    Generates instances of the "separable-color puzzles" inspired in the game The Witness
    
    All puzzles generated are guaranteed to be solvable. This is achieved by first generating
    a path from the initial position to the goal position. Then, bullets with different colors
    are placed into different regions of the puzzle (i.e., set of reachable cells). 
    """
    def generate_paths(self, lines, columns, line_init, column_init, line_goal, column_goal, budget=100000):
        """
        Generates all possible paths for a grid with starting position at (line_init, column_init)
        and finishing position at (line_goal, column_goal). The grid is defined by a grid of size
        lines x columns. 
        
        Returns a list of paths in the form of GameState instances. 
        """
        open_list = deque()
        closed_list = set()
        
        root = WitnessState(lines, columns, line_init, column_init, line_goal, column_goal)
        open_list.append(root)
        closed_list.add(root)
        valid_paths = []
        
        while len(open_list) > 0:
            state = open_list.popleft()
            actions = state.successors() 
            for a in actions:
                c = copy.deepcopy(state)
                c.apply_action(a)
                if c.has_tip_reached_goal():
                    valid_paths.append(c)
                elif not c in closed_list:
                    open_list.append(c)
                    closed_list.add(c)
            if budget > 0 and len(open_list) >= budget:
                break
        return valid_paths
    
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
    
    def generate_puzzles(self, puzzle_sizes, bullet_probability, puzzle_folder):
            """
            Generates a set of puzzles for the sizes given in puzzle_sizes. Currently the puzzles
            have their starting position at (0, 0) and finishing position at (size[0], size[1]),
            which is the size of the puzzle's grid. 
            
            For each puzzle size the method calls generate_paths to create all paths from the
            starting and finishing positions. Then, it generates a puzzle while placing a number
            of bullets in each region of the puzzle (defined by a path). The number of bullets
            in each region is specified by the parameter bullet_probability, which specifies the
            probability of a given cell receiving a collored bullet or being left empty.
            
            All generated puzzles are saved in a folder determined by variable puzzle_folder. 
            """
            color_bullets = [1, 2, 3, 4]
            puzzles_generated = set()
            for size in puzzle_sizes:
                states = self.generate_paths(size[0], size[1], 0, 0, size[0], size[1])
                puzzle_id = 1

                for state in states:
                    regions = state.partition_cells()
                    
                    for comb in itertools.product([0] + color_bullets, repeat=len(regions)):
                        for dot_probability in bullet_probability:
                            copy_state = copy.deepcopy(state)

                            for index, color in enumerate(comb): 
                                self.fill_region(copy_state, regions[index], color, dot_probability)
                                
                            #copy_state.plot()
                            copy_state.clear_path()
                            if copy_state in puzzles_generated:
                                continue
                            copy_state.save_state(puzzle_folder + '/' + str(size[0]) + 'x' + str(size[1]) + '_' + str(puzzle_id))
                            puzzles_generated.add(copy_state)
                            puzzle_id += 1
                            
    def generate_puzzles_random(self, puzzle_sizes, bullet_probability, puzzle_folder, number_puzzles):
        """
        Generates a set of puzzles for the sizes given in puzzle_sizes. Currently the puzzles
        have their starting position at (0, 0) and finishing position at (size[0], size[1]),
        which is the size of the puzzle's grid. 
        
        For each puzzle size the method calls generate_paths to create all paths from the
        starting and finishing positions. Then, it generates a puzzle while placing a number
        of bullets in each region of the puzzle (defined by a path). The number of bullets
        in each region is specified by the parameter bullet_probability, which specifies the
        probability of a given cell receiving a collored bullet or being left empty.
        
        All generated puzzles are saved in a folder determined by variable puzzle_folder. 
        """
        #color_bullets = [1, 2, 3, 4]
        color_bullets = [1, 2]
        puzzles_generated = {}
        indice_number_puzzle = 0
        for size in puzzle_sizes:
            max_difficulty = False
            minimum_number_regions = 2
            if size[0] == 3:
                max_difficulty = True
                minimum_number_regions = 2   
            if size[0] >= 4:
                max_difficulty = True
                minimum_number_regions = 4
            if size[0] >= 6:
                max_difficulty = True
                minimum_number_regions = 5    
            
                               
            print('Generating Paths...')
            
            states = []
            for i in range(1, size[1] + 1):
                states = states + self.generate_paths(size[0], size[1], 0, 0, 0, i)
                states = states + self.generate_paths(size[0], size[1], 0, 0, size[0], i)
            for i in range(1, size[0] + 1):
                states = states + self.generate_paths(size[0], size[1], 0, 0, i, 0)
                states = states + self.generate_paths(size[0], size[1], 0, 0, i, size[1])
            
            print('Generating Puzzles...')
            np.random.shuffle(states)
            puzzles_generated[size] = set()
            puzzle_id = 1
            while len(puzzles_generated[size]) < number_puzzles[indice_number_puzzle]:
                print('Currently have ', len(puzzles_generated[size]), ' puzzles')
                for state in states:
                    regions = state.partition_cells()
                    copy_state = copy.deepcopy(state)
                    
                    if len(regions) < minimum_number_regions:
                        continue    
                    used_colors = set()
                    for region in regions:
                        color = np.random.randint(1, high=len(color_bullets)+1)
                        if max_difficulty:
                            while color in used_colors and len(used_colors) < len(color_bullets):
                                color = np.random.randint(1, high=len(color_bullets)+1)
                        if self.fill_region(copy_state, region, color, bullet_probability):
                            used_colors.add(color)
                            
                    if len(used_colors) < 2:
                        continue
                        
                    copy_state.clear_path()                            
                    
                    if copy_state in puzzles_generated[size]:
                        continue
                    
                    puzzles_generated[size].add(copy_state)
                    copy_state.save_state(puzzle_folder + '/' + str(size[0]) + 'x' + str(size[1]) + '_' + str(puzzle_id))
                    puzzle_id += 1
                    
                    if len(puzzles_generated[size]) == number_puzzles[indice_number_puzzle]:
                        break
                if len(puzzles_generated[size]) == 0:
                    print('No paths with the minimum number of partitions')
                    break
                
            indice_number_puzzle += 1

def main():
    if len(sys.argv[1:]) < 1:
        print('Usage: puzzle_generator.py <name folder>')
        return
    
    puzzle_folder = sys.argv[1]
    
    if not os.path.exists(puzzle_folder):
        os.makedirs(puzzle_folder)

#     puzzle_sizes = [(1, 2), (1, 3), (2, 2), (3, 3), (4, 4)]
#     number_puzzles = [10, 50, 300, 1000, 1000]

    puzzle_sizes = [(6, 6)]
    number_puzzles = [4000]
    
    #bullet_probability = [0.1, 0.5, 0.9]
    generator = PuzzleGenerator()
    #generator.generate_puzzles(puzzle_sizes, bullet_probability, puzzle_folder)
    generator.generate_puzzles_random(puzzle_sizes, 0.8, puzzle_folder, number_puzzles)
            
if __name__ == "__main__":
    main()
