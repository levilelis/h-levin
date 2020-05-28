import os
import time
from os import listdir
from os.path import isfile, join
from domains.witness import WitnessState
from search.bfs_levin import BFSLevin
from models.memory import Memory
from models.model_wrapper import KerasManager, KerasModel
from concurrent.futures.process import ProcessPoolExecutor
import argparse
from search.a_star import AStar
from search.gbfs import GBFS
from search.bfs_levin_mult import BFSLevinMult
from domains.sliding_tile_puzzle import SlidingTilePuzzle
from domains.sokoban import Sokoban
from search.puct import PUCT
    
   
def search(states, planner, nn_model, ncpus, output='', single_file=False):
    """
    This function runs (best-first) Levin tree search with a learned policy on a set of problems    
    """   
    problem_name = ''
    if single_file:
        log_folder = 'logs_search/' + output + '/'
        problem_name = next(iter(states.keys()))
        
        if '.pro' in problem_name:
            problem_name = problem_name[0:problem_name.rfind('.')]
                            
#         if not os.path.exists(log_folder):
#             os.makedirs(log_folder)
        
    total_expanded = 0
    total_generated = 0
    total_cost = 0
    
    for _, state in states.items():
        state.reset()
    
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
        
        partial_time = time.time()
        
        if single_file:
            with open(join(log_folder + problem_name), 'a') as results_file:
                results_file.write("{:d}, {:d}, {:d}, {:.2f} \n".format(solution_depth, expanded, generated, partial_time - start_total))
            print("{:d}, {:d}, {:d}, {:.2f}".format(solution_depth, expanded, generated, partial_time - start_total))
        
    if not single_file:
        end_total = time.time()
        print("Cost: {:d} \t Expanded: {:d} \t Generated: {:d}, Time: {:.2f}".format(total_cost,
                                                                                    total_expanded, 
                                                                                    total_generated, 
                                                                                    end_total - start_total))

def bootstrap_learning_bfs(states, planner, nn_model, output, initial_budget, ncpus):
#     
#     search(states, planner, nn_model, ncpus)
    
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
            for _ in range(500):
                loss = nn_model.train_with_memory(memory)
                print(loss)
#             if number_solved < 20:
#                 budget *= 2
            memory.clear()
            
            nn_model.save_weights(join(models_folder, 'model_weights')) 
        else:
            budget *= 2
            print('Budget: ', budget)
            continue
                                
        iteration += 1
    
    search(states, planner, nn_model, ncpus)


def main():
    """
    It is possible to use this system to either train a new neural network model through the bootstrap system and
    Levin tree search (LTS) algorithm, or to use a trained neural network with LTS.         
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-l', action='store', dest='loss_function', 
                        default='LevinLoss', 
                        help='Loss Function')
    
    parser.add_argument('-p', action='store', dest='problems_folder',
                        help='Folder with problem instances')
    
    parser.add_argument('-m', action='store', dest='model_name',
                        help='Name of the folder of the neural model')
    
    parser.add_argument('-a', action='store', dest='search_algorithm',
                        help='Name of the search algorithm (Levin, LevinStar, AStar, GBFS, PUCT)')
    
    parser.add_argument('-d', action='store', dest='problem_domain',
                        help='Problem domain (Witness or SlidingTile)')
    
    parser.add_argument('-b', action='store', dest='search_budget',
                        help='The initial budget (nodes expanded) allowed to the bootstrap procedure')
    
    parser.add_argument('-cpuct', action='store', dest='cpuct', default='1.0', 
                        help='Constant C used with PUCT.')
    
    parser.add_argument('--default-heuristic', action='store_true', default=False,
                        dest='use_heuristic',
                        help='Use the default heuristic as input')
    
    parser.add_argument('--learned-heuristic', action='store_true', default=False,
                        dest='use_learned_heuristic',
                        help='Use/learn a heuristic')
    
    parser.add_argument('--blind-search', action='store_true', default=False,
                        dest='blind_search',
                        help='Perform blind search')
    
    parser.add_argument('--single-test-file', action='store_true', default=False,
                        dest='single_test_file',
                        help='Use this if problem instance is a file containing a single instance.')
    
    parser.add_argument('--learn', action='store_true', default=False,
                        dest='learning_mode',
                        help='Train as neural model out of the instances from the problem folder')
    
    parameters = parser.parse_args()
    
    states = {}
    
    if parameters.problem_domain == 'SlidingTile' and parameters.single_test_file:
        with open(parameters.problems_folder, 'r') as file:
            problem = file.readline()
            instance_name = parameters.problems_folder[parameters.problems_folder.rfind('/') + 1:len(parameters.problems_folder)]
            puzzle = SlidingTilePuzzle(problem)
            states[instance_name] = puzzle
    
    elif parameters.problem_domain == 'SlidingTile':
        puzzle_files = [f for f in listdir(parameters.problems_folder) if isfile(join(parameters.problems_folder, f))]
    
        for filename in puzzle_files:
            with open(join(parameters.problems_folder, filename), 'r') as file:
                problems = file.readlines()
                
                for i in range(len(problems)):
                    puzzle = SlidingTilePuzzle(problems[i])
                    states['puzzle_' + str(i)] = puzzle
    
    elif parameters.problem_domain == 'Witness':
        puzzle_files = [f for f in listdir(parameters.problems_folder) if isfile(join(parameters.problems_folder, f))]
        
        for file in puzzle_files:
            if '.' in file:
                continue
            s = WitnessState()
            s.read_state(join(parameters.problems_folder, file))
            states[file] = s
    elif parameters.problem_domain == 'Sokoban':
        problem = []
        with open(parameters.problems_folder, 'r') as file:
            all_problems = file.readlines()
        
        problem_id = 0
        for line_in_problem in all_problems:
            if ';' in line_in_problem:
                if len(problem) > 0:
                    puzzle = Sokoban(problem)
                    states[problem_id] = puzzle
                
                problem = []
#                 problem_id = line_in_problem.split(' ')[1].split('\n')[0]
                problem_id += 1
            
            elif '\n' != line_in_problem:
                problem.append(line_in_problem.split('\n')[0])
                
        if len(problem) > 0:
            puzzle = Sokoban(problem)
            states[problem_id] = puzzle
                
#     input_size = s.get_image_representation().shape
            
    KerasManager.register('KerasModel', KerasModel)
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default = 2))
    
    k_expansions = 32
    
#     print('Number of cpus available: ', ncpus)
    
    with KerasManager() as manager:
                
        nn_model = manager.KerasModel()
        
        if parameters.search_algorithm == 'PUCT':
       
            bfs_planner = PUCT(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions, float(parameters.cpuct))
        
            if parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=True)     
            else:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=False)
            
            if parameters.learning_mode:
                bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)            
            elif parameters.blind_search:
                search(states, bfs_planner, nn_model, ncpus, parameters.model_name, parameters.single_test_file)
            else:
                nn_model.load_weights(join('trained_models', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, parameters.model_name, parameters.single_test_file)
        
        if parameters.search_algorithm == 'Levin' or parameters.search_algorithm == 'LevinStar':
        
            if parameters.search_algorithm == 'Levin':
                bfs_planner = BFSLevin(parameters.use_heuristic, parameters.use_learned_heuristic, False, k_expansions)
            else:
                bfs_planner = BFSLevin(parameters.use_heuristic, parameters.use_learned_heuristic, True, k_expansions)
        
            if parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=True)     
            else:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=False)
            
            if parameters.learning_mode:
                bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)            
            elif parameters.blind_search:
                search(states, bfs_planner, nn_model, ncpus, parameters.model_name, parameters.single_test_file)
            else:
                nn_model.load_weights(join('trained_models', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, parameters.model_name, parameters.single_test_file)
                
        if parameters.search_algorithm == 'LevinMult':
        
            bfs_planner = BFSLevinMult(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)
        
            if parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=True)     
            else:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=False)
            
            if parameters.learning_mode:
                bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)            
            elif parameters.blind_search:
                search(states, bfs_planner, nn_model, ncpus, parameters.model_name, parameters.single_test_file)
            else:
                nn_model.load_weights(join('trained_models', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, parameters.model_name, parameters.single_test_file)
        
        if parameters.search_algorithm == 'AStar':
            bfs_planner = AStar(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)
            
            if parameters.learning_mode and parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm)
                bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
            elif parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm) 
                nn_model.load_weights(join('trained_models', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, parameters.model_name, parameters.single_test_file)
            else:
                search(states, bfs_planner, nn_model, ncpus, parameters.model_name, parameters.single_test_file)  
                
        if parameters.search_algorithm == 'GBFS':
            bfs_planner = GBFS(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)
            
            if parameters.learning_mode:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm)
                bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
            elif parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm) 
                nn_model.load_weights(join('trained_models', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, parameters.model_name, parameters.single_test_file)
            else:
                search(states, bfs_planner, nn_model, ncpus, parameters.model_name, parameters.single_test_file)      
            
if __name__ == "__main__":
    main()
