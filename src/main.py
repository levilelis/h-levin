import os
import sys
import time
from os import listdir
from os.path import isfile, join
from search.dfs_levin import DFSLevin
from domains.witness import WitnessState
    
def bootstrap_learning(states, loss_name, output):
    """
    This function runs (iterative-deepening) Levin tree search within the Bootstrap framework. The process starts
    with policy encoded in a neural network with randomly initialized weights and the minimum search budget,
    which is given by an interger b (stored in the dictionary state_budget for different problems in the code)
    and computed as e^b. In every iteration LTS tries to solve every problem in the problem set with a fixed
    search budget. If no problems are solved, then we increase the budget b by 1. If at least one problem 
    is solved in a given iteration, then we train the neural network with the solved problems and set b = 1.
    
    We repeat this process until all problems in the set of problems (denoted by list states in the code).    
    """

    planner = DFSLevin(loss_name) 
    
#     dfs_levin(states, planner)
    
    models_folder = 'trained_models/' + output
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    
    state_budget = {}
    for file, state in states.items():
        state_budget[state] = 1
    
    unsolved_puzzles = states
    id_solved = 1
    id_batch = 1
    total_expanded = 0
    total_generated = 0
    
    start = time.time()
    
    while len(unsolved_puzzles) > 0:
        number_solved = 0
        number_unsolved = 0
        current_unsolved_puzzles = {}
        
        for file, state in unsolved_puzzles.items():
            state.clear_path()
            has_found_solution, new_bound, expanded, generated = planner.search_for_learning(state, state_budget[state])
            
            total_expanded += expanded
            total_generated += generated
            
            if has_found_solution:
                number_solved += 1
                id_solved += 1
            else:
                if new_bound != -1:
                    state_budget[state] = new_bound
                else:
                    state_budget[state] = state_budget[state] + 1 
                number_unsolved += 1
                current_unsolved_puzzles[file] = state
        
        end = time.time()
        with open(join('logs/training_bootstrap_' + output), 'a') as results_file:
            results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(id_batch, 
                                                                             number_solved, 
                                                                             number_unsolved, 
                                                                             planner.size_training_set(), 
                                                                             planner.current_budget(),
                                                                             total_expanded,
                                                                             total_generated, 
                                                                             end-start)))
            results_file.write('\n')
        
        if number_solved != 0:
            planner.reset_budget()
            for file, state in states.items():
                state_budget[state] = 1
        else:
            planner.increase_budget()
            continue
                
        unsolved_puzzles = current_unsolved_puzzles
        
        planner.preprocess_data()
        
        for _ in range(5):
            error = planner.learn()
            print(error)
        
        if id_batch % 1 == 0:
            planner.save_model(join(models_folder, 'model_weights')) 
        
        id_batch += 1
    
    planner.save_model(join(models_folder, 'model_weights')) 
    
    dfs_levin(states, planner)
    
def dfs_levin(states, planner):
    """
    This function runs (iterative-deepening) Levin tree search with a learned policy on a set of problems    
    """   
    total_expanded = 0
    total_generated = 0
    total_cost = 0
    
    start_total = time.time()
    
    for _, state in states.items():
        start = time.time()
        
        state.clear_path()
        solution_depth, expanded, generated = planner.search(state)
        total_expanded += expanded
        total_generated += generated
        total_cost += solution_depth
        
        end = time.time()
        
        print("{:d}, {:d}, {:d}, {:.2f}".format(solution_depth, expanded, generated, end - start))
    
    end_total = time.time()
    print("Cost: {:d} \t Expanded: {:d} \t Generated: {:d}, Time: {:.2f}".format(total_cost,
                                                                                 total_expanded, 
                                                                                total_generated, 
                                                                                end_total - start_total))    
    
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
        print('Usage for learning a new model: main folder-with-puzzles output-file loss-name model-to-be-loaded')
        print('Loss can be either LevinLoss or CrossEntropyLoss')
        print('Model to be loaded is optional. If name is not provided, then a new model will be trained.')
        return
    puzzle_folder = sys.argv[1]
    loss_name = sys.argv[2]
    output_file = sys.argv[3]
    model_file = None
    
    if len(sys.argv) == 5:
        model_file = sys.argv[4]
    
    states = {}
    puzzle_files = [f for f in listdir(puzzle_folder) if isfile(join(puzzle_folder, f))]
    
    for file in puzzle_files:
        if '.' in file:
            continue
        s = WitnessState()
        s.read_state(join(puzzle_folder, file))
        states[file] = s
    
    if model_file == None:
        bootstrap_learning(states, loss_name, output_file)
    else:
        planner = DFSLevin(loss_name, model_file)
        dfs_levin(states, planner)
            
if __name__ == "__main__":
    main()
