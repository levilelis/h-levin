import os
import time
from os.path import join
from models.memory import Memory
from concurrent.futures.process import ProcessPoolExecutor

class Bootstrap:
    def __init__(self, states, output, ncpus=1, initial_budget=2000, gradient_steps=10):
        self._states = states
        self._model_name = output
        self._number_problems = len(states)
        
        self._ncpus = ncpus
        self._initial_budget = initial_budget
        self._gradient_steps = gradient_steps
#         self._k = ncpus * 3
        self._batch_size = 32
        
        self._kmax = 10
        
        self._log_folder = 'training_logs/'
        self._models_folder = 'trained_models_online/' + self._model_name
        
        if not os.path.exists(self._models_folder):
            os.makedirs(self._models_folder)
            
        if not os.path.exists(self._log_folder):
            os.makedirs(self._log_folder)
    
    def solve_uniform_online(self, planner, nn_model):
        iteration = 1
        number_solved = 0
        total_expanded = 0
        total_generated = 0
        
        budget = self._initial_budget
        memory = Memory()
        start = time.time()
        
        current_solved_puzzles = set()
                
        while len(current_solved_puzzles) < self._number_problems:
            number_solved = 0
            
            batch_problems = {}
            for name, state in self._states.items():
                
#                 if name in current_solved_puzzles:
#                     continue
                
                batch_problems[name] = state
                
                if len(batch_problems) < self._batch_size and self._number_problems - len(current_solved_puzzles) > self._batch_size:
                    continue
            
                with ProcessPoolExecutor(max_workers = self._ncpus) as executor:
                    args = ((state, name, budget, nn_model) for name, state in batch_problems.items()) 
                    results = executor.map(planner.search_for_learning, args)
                for result in results:
                    has_found_solution = result[0]
                    trajectory = result[1]
                    total_expanded += result[2]
                    total_generated += result[3]
                    puzzle_name = result[4]
                    
                    if has_found_solution:
                        memory.add_trajectory(trajectory)
                     
                    if has_found_solution and puzzle_name not in current_solved_puzzles:
                        number_solved += 1
                        current_solved_puzzles.add(puzzle_name)
                        
                if memory.number_trajectories() > 0:
                    for _ in range(self._gradient_steps):
                        loss = nn_model.train_with_memory(memory)
                        print('Loss: ', loss)
                    memory.clear()
                    nn_model.save_weights(join(self._models_folder, 'model_weights'))
                
                batch_problems.clear() 
            
            end = time.time()
            with open(join(self._log_folder + 'training_bootstrap_' + self._model_name), 'a') as results_file:
                results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(iteration, 
                                                                                 number_solved, 
                                                                                 self._number_problems - len(current_solved_puzzles), 
                                                                                 budget,
                                                                                 total_expanded,
                                                                                 total_generated, 
                                                                                 end-start)))
                results_file.write('\n')
            
            print('Number solved: ', number_solved)
            if number_solved == 0:
                budget *= 2
                print('Budget: ', budget)
                continue
                                    
            iteration += 1
    