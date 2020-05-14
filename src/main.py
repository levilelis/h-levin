import os
import sys
import time
import numpy as np
from os import listdir
from os.path import isfile, join
from domains.witness import WitnessState
from search.bfs_levin import BFSLevin
from models.memory import Memory
from models.conv_net import ConvNet, TwoHeadedConvNet
from models.model_wrapper import KerasManager, KerasModel
from concurrent.futures.process import ProcessPoolExecutor
    
   
def levin_search(states, planner, nn_model, ncpus):
    """
    This function runs (iterative-deepening) Levin tree search with a learned policy on a set of problems    
    """   
    total_expanded = 0
    total_generated = 0
    total_cost = 0
    
    for _, state in states.items():
        state.clear_path()
    
    start_total = time.time()
    
    with ProcessPoolExecutor(max_workers = ncpus) as executor:
        args = ((state, nn_model) for name, state in states.items()) 
        results = executor.map(planner.search, args)
    for result in results:
        solution_depth = result[0]
        expanded = result[1]
        generated = result[2]
        
        total_expanded += expanded
        total_generated += generated
        total_cost += solution_depth
        
        print("{:d}, {:d}, {:d}".format(solution_depth, expanded, generated))
    
#     for _, state in states.items():
#         start = time.time()
#         
#         solution_depth, expanded, generated = planner.search(state, nn_model)
#         total_expanded += expanded
#         total_generated += generated
#         total_cost += solution_depth
#         
#         end = time.time()
        
    end_total = time.time()
    print("Cost: {:d} \t Expanded: {:d} \t Generated: {:d}, Time: {:.2f}".format(total_cost,
                                                                                 total_expanded, 
                                                                                total_generated, 
                                                                                end_total - start_total))    


def bootstrap_learning_bfs(states, planner, nn_model, output, initial_budget, ncpus):
    
    levin_search(states, planner, nn_model, ncpus)
    
    log_folder = 'logs/'
    models_folder = 'trained_models/' + output
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
        
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    number_problems = len(states)
    
    iteration = 1
    number_solved = 0
    total_expanded = 0
    total_generated = 0
    
    budget = initial_budget
    memory = Memory()
    start = time.time()
    
    current_solved_puzzles = set()
    
    while len(current_solved_puzzles) < number_problems:
        number_solved = 0
        
        with ProcessPoolExecutor(max_workers = ncpus) as executor:
            args = ((state, name, budget, nn_model) for name, state in states.items()) 
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
            
#         for file, state in states.items():
#             state.clear_path()
#                         
#             has_found_solution, trajectory, expanded, generated = planner.search_for_learning(state, budget, nn_model)
#              
#             total_expanded += expanded
#             total_generated += generated
#              
#             if has_found_solution:
#                 memory.add_trajectory(trajectory)
#              
#             if has_found_solution and file not in current_solved_puzzles:
#                 number_solved += 1
#                 current_solved_puzzles.add(file)
        
        end = time.time()
        with open(join(log_folder + 'training_bootstrap_' + output), 'a') as results_file:
            results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(iteration, 
                                                                             number_solved, 
                                                                             number_problems - len(current_solved_puzzles), 
                                                                             budget,
                                                                             total_expanded,
                                                                             total_generated, 
                                                                             end-start)))
            results_file.write('\n')
        
        print('Number solved: ', number_solved)
        if number_solved > 0:
            for _ in range(50):
                loss = nn_model.train_with_memory(memory)
                print(loss)
            budget = initial_budget
            memory.clear()
        else:
            budget *= 2
            print('Budget: ', budget)
            continue
                                
        iteration += 1
    
    nn_model.save_weights(join(models_folder, 'model_weights')) 
    
    levin_search(states, planner, nn_model, ncpus)


def main():
    """
    It is possible to use this system to either train a new neural network model through the bootstrap system and
    Levin tree search (LTS) algorithm, or to use a trained neural network with LTS. 
    
    Examples of usage:
        For learning a new mode use the following:
        
        main bootstrap_dfs_lvn_learning_planner puzzles_1x2 puzzles_1x2_output 1.0 1024
        
        The program will run LTS on the set of puzzles in the folder puzzles_1x2. The learned model will be saved in
        the folder called puzzles_1x2_output. The dropout rate is set to 1.0, which means that no dropout is used. The
        batch size is set to 1024. The program will generate the following log files:
            
            File puzzles_1x2_output_log_training_bootstrap containing in each line the id of the LTS iteration,
            number of problems solved in that iteration, number of problems yet to be solved, training set size
            (including all possible reflection of the image of puzzles solved), and time spent size the beginning
            of the learning process. 
            
            File puzzles_1x2_output_puzzle_names_ordering_bootstrap containing the id of a given iteration of LTS,
            followed by a list of name of the puzzles solved in that iteration, and the LTS search budget used
            to solve those problems. The budget is computed as e^b, where b is the budget. 
            
            Folder puzzles_1x2_output_models with the last neural network model trained by the system.  
        
        For using a learned model to solve a set of puzzles use the following:
        
        main learned_planner puzzles_1x2 puzzles_1x2_output puzzles_1x2_output_models
        
        The program will run LTS on the set of problems in the folder puzzles_1x2 with the model encountered in the folder
        puzzles_1x2_output_models. The program will then generate two files as output:
        
            puzzles_1x2_output_log_learned_bootstrap, which is exactly as the log file described above while learning a model.
        
            puzzles_1x2_output_puzzle_names_ordering_bootstrap_learned_model, which is exactly as the file with the puzzle 
            names described above while learning a model.          
    """
    if len(sys.argv) < 3:
        print('Usage for learning a new model: main folder-with-puzzles output-file loss-name use-heuristic model-to-be-loaded')
        print('Loss can be either LevinLoss or CrossEntropyLoss')
        print('Use heuristic is either y or n')
        print('Model to be loaded is optional. If name is not provided, then a new model will be trained.')
        return
    puzzle_folder = sys.argv[1]
    loss_name = sys.argv[2]
    output_file = sys.argv[3]
    use_heuristic = sys.argv[4]
    use_learned_heuristic = sys.argv[5]
    model_file = None
        
    if len(sys.argv) == 7:
        model_file = sys.argv[6]
    
    states = {}
    puzzle_files = [f for f in listdir(puzzle_folder) if isfile(join(puzzle_folder, f))]
    
    for file in puzzle_files:
        if '.' in file:
            continue
        s = WitnessState()
        s.read_state(join(puzzle_folder, file))
        states[file] = s
        
    KerasManager.register('KerasModel', KerasModel)
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default = 2))
    
    print('Number of cpus available: ', ncpus)
    
    with KerasManager() as manager:
                
        nn_model = manager.KerasModel()
    
        if use_heuristic == 'y' and use_learned_heuristic == 'f':
            bfs_planner = BFSLevin(use_heuristic=True, use_learned_heuristic=False)
            nn_model.initialize(loss_name, two_headed_model=False)
    
        elif use_heuristic == 'y' and use_learned_heuristic == 'y':
            bfs_planner = BFSLevin(use_heuristic=True, use_learned_heuristic=True)
            nn_model.initialize(loss_name, two_headed_model=True)
            
        else:
            bfs_planner = BFSLevin(use_heuristic=False, use_learned_heuristic=False)
            nn_model.initialize(loss_name, two_headed_model=False)
        
        if model_file == None:
            bootstrap_learning_bfs(states, bfs_planner, nn_model, output_file, 500, ncpus)
        else:
            nn_model.load_weights(model_file).expect_partial()
            levin_search(states, bfs_planner, nn_model, ncpus)
            
if __name__ == "__main__":
    main()
