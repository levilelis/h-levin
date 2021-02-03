import os
import time
from os import listdir
from os.path import isfile, join
from domains.witness import WitnessState
from search.bfs_levin import BFSLevin
from models.model_wrapper import KerasManager, KerasModel
from concurrent.futures.process import ProcessPoolExecutor
import argparse
from search.a_star import AStar
from search.gbfs import GBFS
from domains.sliding_tile_puzzle import SlidingTilePuzzle
from domains.sokoban import Sokoban
from search.puct import PUCT
from bootstrap import Bootstrap

def search_time_limit(states, planner, nn_model, ncpus, time_limit_seconds):
    """
    This function runs (best-first) Levin tree search with a learned policy on a set of problems.
    The search will be bounded by a time limit. The number of nodes expanded and generated will be
    reported, independently if the planner solved the problem or not. If the planner solves the 
    problem, then the procedure also reports solution depth.     
    """          
    total_expanded = 0
    total_generated = 0
    total_cost = 0
    
    solutions = {}
        
    for name, state in states.items():
        state.reset()
        solutions[name] = (-1, -1, -1, -1)
    
    with ProcessPoolExecutor(max_workers = ncpus) as executor:
        args = ((state, name, nn_model, -1, time.time(), time_limit_seconds, 0) for name, state in states.items()) 
        results = executor.map(planner.search, args)
    for result in results:
        solution_depth = result[0]
        expanded = result[1]
        generated = result[2]
        running_time = result[3]
        puzzle_name = result[4]
        
        solutions[puzzle_name] = (solution_depth, expanded, generated, running_time)
        
        if solution_depth > 0:
            total_expanded += expanded
            total_generated += generated
            total_cost += solution_depth
                           
    for name, data in solutions.items():
        print("{:s}, {:d}, {:d}, {:d}, {:.2f}".format(name, data[0], data[1], data[2], data[3]))
    
   
def search(states, planner, nn_model, ncpus, time_limit_seconds, search_budget=-1):
    """
    This function runs (best-first) Levin tree search with a learned policy on a set of problems    
    """          
    total_expanded = 0
    total_generated = 0
    total_cost = 0
    
    slack_time = 600
    
    solutions = {}
        
    for name, state in states.items():
        state.reset()
        solutions[name] = (-1, -1, -1, -1)
    
    start_time = time.time()
    
    while len(states) > 0:
    
#         args = [(state, name, nn_model, search_budget, start_time, time_limit_seconds, slack_time) for name, state in states.items()]
#         solution_depth, expanded, generated, running_time, puzzle_name = planner.search(args[0])
      
        with ProcessPoolExecutor(max_workers = ncpus) as executor:
            args = ((state, name, nn_model, search_budget, start_time, time_limit_seconds, slack_time) for name, state in states.items()) 
            results = executor.map(planner.search, args)
        for result in results:
            solution_depth = result[0]
            expanded = result[1]
            generated = result[2]
            running_time = result[3]
            puzzle_name = result[4]
            
            if solution_depth > 0:
                solutions[puzzle_name] = (solution_depth, expanded, generated, running_time)
                del states[puzzle_name]
            
            if solution_depth > 0:
                total_expanded += expanded
                total_generated += generated
                total_cost += solution_depth
        
        partial_time = time.time()
                
        if partial_time - start_time + slack_time > time_limit_seconds or len(states) == 0 or search_budget >= 1000000:            
            for name, data in solutions.items():
                print("{:s}, {:d}, {:d}, {:d}, {:.2f}".format(name, data[0], data[1], data[2], data[3]))
            return
        
        search_budget *= 2

def main():
    """
    It is possible to use this system to either train a new neural network model through the bootstrap system and
    Levin tree search (LTS) algorithm, or to use a trained neural network with LTS.         
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-l', action='store', dest='loss_function', 
                        default='CrossEntropyLoss', 
                        help='Loss Function')
    
    parser.add_argument('-p', action='store', dest='problems_folder',
                        help='Folder with problem instances')
    
    parser.add_argument('-m', action='store', dest='model_name',
                        help='Name of the folder of the neural model')
    
    parser.add_argument('-a', action='store', dest='search_algorithm',
                        help='Name of the search algorithm (Levin, LevinStar, AStar, GBFS, PUCT)')
    
    parser.add_argument('-d', action='store', dest='problem_domain',
                        help='Problem domain (Witness or SlidingTile)')
    
    parser.add_argument('-b', action='store', dest='search_budget', default=1000,
                        help='The initial budget (nodes expanded) allowed to the bootstrap procedure')
    
    parser.add_argument('-g', action='store', dest='gradient_steps', default=10,
                        help='Number of gradient steps to be performed in each iteration of the Bootstrap system')
    
    parser.add_argument('-cpuct', action='store', dest='cpuct', default='1.0', 
                        help='Constant C used with PUCT.')
    
    parser.add_argument('-time', action='store', dest='time_limit', default='43200', 
                        help='Time limit in seconds for search')
        
    parser.add_argument('-mix', action='store', dest='mix_epsilon', default='0.0', 
                        help='Mixture with a uniform policy')
    
    parser.add_argument('-w', action='store', dest='weight_astar', default='1.0', 
                        help='Weight to be used with WA*.')
    
    parser.add_argument('--default-heuristic', action='store_true', default=False,
                        dest='use_heuristic',
                        help='Use the default heuristic as input')
    
    parser.add_argument('--learned-heuristic', action='store_true', default=False,
                        dest='use_learned_heuristic',
                        help='Use/learn a heuristic')
    
    parser.add_argument('--blind-search', action='store_true', default=False,
                        dest='blind_search',
                        help='Perform blind search')
        
    parser.add_argument('--learn', action='store_true', default=False,
                        dest='learning_mode',
                        help='Train as neural model out of the instances from the problem folder')
    
    parser.add_argument('--fixed-time', action='store_true', default=False,
                        dest='fixed_time',
                        help='Run the planner for a fixed amount of time (specified by time_limit) for each problem instance')
    
    parser.add_argument('-number-test-instances', action='store', dest='number_test_instances', default='0', 
                        help='Maximum number of test instances (value of zero will use all instances in the test file).')
    
    parameters = parser.parse_args()
    
    states = {}
    
    if parameters.problem_domain == 'SlidingTile':
        puzzle_files = [f for f in listdir(parameters.problems_folder) if isfile(join(parameters.problems_folder, f))]
    
        j = 1
        for filename in puzzle_files:
            with open(join(parameters.problems_folder, filename), 'r') as file:
                problems = file.readlines()
                
                for i in range(len(problems)):
                    puzzle = SlidingTilePuzzle(problems[i])
                    states['puzzle_' + str(j)] = puzzle
                    
                    j += 1
                        
    elif parameters.problem_domain == 'Witness':
        puzzle_files = [f for f in listdir(parameters.problems_folder) if isfile(join(parameters.problems_folder, f))]
        
        j = 1
               
        for filename in puzzle_files:
            if '.' in filename:
                continue
            
            with open(join(parameters.problems_folder, filename), 'r') as file:
                puzzle = file.readlines()
                
                i = 0
                while i < len(puzzle):
                    k = i
                    while k < len(puzzle) and puzzle[k] != '\n':
                        k += 1
                    s = WitnessState()
                    s.read_state_from_string(puzzle[i:k])
                    states['puzzle_' + str(j)] = s
                    i = k + 1
                    j += 1
#             s.read_state(join(parameters.problems_folder, filename))
#             states[filename] = s
            
    elif parameters.problem_domain == 'Sokoban':
        problem = []
        puzzle_files = []
        if isfile(parameters.problems_folder):
            puzzle_files.append(parameters.problems_folder)
        else:
            puzzle_files = [join(parameters.problems_folder, f) for f in listdir(parameters.problems_folder) if isfile(join(parameters.problems_folder, f))]
        
        problem_id = 0
        
        for filename in puzzle_files:
            with open(filename, 'r') as file:
                all_problems = file.readlines()
            
            
            for line_in_problem in all_problems:
                if ';' in line_in_problem:
                    if len(problem) > 0:
                        puzzle = Sokoban(problem)
                        states['puzzle_' + str(problem_id)] = puzzle
                    
                    problem = []
    #                 problem_id = line_in_problem.split(' ')[1].split('\n')[0]
                    problem_id += 1
                
                elif '\n' != line_in_problem:
                    problem.append(line_in_problem.split('\n')[0])
                    
            if len(problem) > 0:
                puzzle = Sokoban(problem)
                states['puzzle_' + str(problem_id)] = puzzle
    
    if int(parameters.number_test_instances) != 0:
        states_capped = {}
        counter = 0
        
        for name, puzzle in states.items():
            states_capped[name] = puzzle
            counter += 1
            
            if counter == int(parameters.number_test_instances):
                break
        
        states = states_capped     
    
    print('Loaded ', len(states), ' instances')
#     input_size = s.get_image_representation().shape
           
    KerasManager.register('KerasModel', KerasModel)
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default = 1))
    
    k_expansions = 32
    
#     print('Number of cpus available: ', ncpus)
    
    start = time.time()
    
    with KerasManager() as manager:
                
        nn_model = manager.KerasModel()
        bootstrap = None
        
        if parameters.learning_mode:
            bootstrap = Bootstrap(states, parameters.model_name, 
                                  ncpus=ncpus, 
                                  initial_budget=int(parameters.search_budget),
                                  gradient_steps=int(parameters.gradient_steps))
        
        if parameters.search_algorithm == 'PUCT':
       
            bfs_planner = PUCT(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions, float(parameters.cpuct))
        
            if parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=True)     
            else:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=False)
            
            if parameters.learning_mode:
                #bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_uniform_online(bfs_planner, nn_model)            
            elif parameters.blind_search:
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
            elif parameters.fixed_time:
                nn_model.load_weights(join('trained_models_online', parameters.model_name, 'model_weights'))
                search_time_limit(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit))
            else:
                nn_model.load_weights(join('trained_models_online', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
        
        if parameters.search_algorithm == 'Levin' or parameters.search_algorithm == 'LevinStar':
        
            if parameters.search_algorithm == 'Levin':
                bfs_planner = BFSLevin(parameters.use_heuristic, parameters.use_learned_heuristic, False, k_expansions, float(parameters.mix_epsilon))
            else:
                bfs_planner = BFSLevin(parameters.use_heuristic, parameters.use_learned_heuristic, True, k_expansions, float(parameters.mix_epsilon))
        
            if parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=True)     
            else:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=False)
            
            if parameters.learning_mode:
#                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_uniform_online(bfs_planner, nn_model)            
            elif parameters.blind_search:
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
            elif parameters.fixed_time:
                nn_model.load_weights(join('trained_models_online', parameters.model_name, 'model_weights'))
                search_time_limit(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit))
            else:
                nn_model.load_weights(join('trained_models_online', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
                
        if parameters.search_algorithm == 'LevinMult':
        
            bfs_planner = BFSLevinMult(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)
        
            if parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=True)     
            else:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=False)
            
            if parameters.learning_mode:
#                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_uniform_online(bfs_planner, nn_model)            
            elif parameters.blind_search:
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
            else:
                nn_model.load_weights(join('trained_models_online', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
        
        if parameters.search_algorithm == 'AStar':
            bfs_planner = AStar(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions, float(parameters.weight_astar))
            
            if parameters.learning_mode and parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm)
#                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_uniform_online(bfs_planner, nn_model)
            elif parameters.fixed_time and parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm)
                nn_model.load_weights(join('trained_models_online', parameters.model_name, 'model_weights'))
                search_time_limit(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit))
            elif parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm) 
                nn_model.load_weights(join('trained_models_online', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
            else:
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))  
                
        if parameters.search_algorithm == 'GBFS':
            bfs_planner = GBFS(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)
            
            if parameters.learning_mode:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm)
#                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_uniform_online(bfs_planner, nn_model)
            elif parameters.fixed_time and parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm)
                nn_model.load_weights(join('trained_models_online', parameters.model_name, 'model_weights'))
                search_time_limit(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit))
            elif parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm) 
                nn_model.load_weights(join('trained_models_online', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
            else:
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))      
            
            
    print('Total time: ', time.time() - start)
if __name__ == "__main__":
    main()
    
    
