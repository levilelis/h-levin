import os
import time
from os import listdir
from os.path import isfile, join
from domains.witness import WitnessState
from models.model_wrapper import KerasManager, KerasModel
from concurrent.futures.process import ProcessPoolExecutor
import argparse
from search.a_star import AStar
from search.bfs_levin import BFSLevin
#from search.gbfs import GBFS
from search.puct import PUCT
from search.puct_levi import PUCT as PUCT_Levi
from domains.sliding_tile_puzzle import SlidingTilePuzzle
from domains.sokoban import Sokoban
from bootstrap import Bootstrap
from multiprocessing import set_start_method

    
def search(states, planner, nn_model, ncpus, time_limit_seconds, search_budget=-1):
    total_expanded = 0
    total_generated = 0
    total_cost = 0
    
    slack_time = 600

    print("ncpus {:d} time_limit_seconds {:d} search_budget {:d}".format(ncpus, time_limit_seconds, search_budget))

    solutions = {}
        
    for name, state in states.items():
        state.reset()
        solutions[name] = (-1, -1, -1, -1)
    
    start_time = time.time()

    # search_budget = 100 # FOR TESTING PURPOSES

    # If the file puzzle_filter_file exists, it specifies the allow list
    # of the problem names to try, and not try the others.
    # If it doesn't exist, all problems are tried.
    # TODO: make these prog args, not env vars
    model_name = os.environ.get('model')    
    puzzle_filter_file = os.environ.get('puzzle_filter_file')
    print("puzzle_filter_file={}".format(puzzle_filter_file))
    puzzle_filter = False
    if os.path.isfile(puzzle_filter_file):
        with open(puzzle_filter_file) as f:
            puzzle_filter = f.read().splitlines()
        
    args = [{'state': state,
             'puzzle_name': name,
             'nn_model': nn_model,
             'node_budget': search_budget,
             # 'time_budget': -1, # time budget for the problem itself
             'overall_time_budget': time_limit_seconds,
             'overall_start_time': time.time(),
             'slack_time': slack_time} for name, state in states.items()]
    print( "(search_budget {} start_time {} time_limit_seconds {} slack_time {})".format(search_budget, start_time, time_limit_seconds, slack_time))

    if puzzle_filter:
        print("Only trying these puzzles: {}".format(puzzle_filter))
        args = [a for a in args if a['puzzle_name'] in puzzle_filter]
        
    print("Filtered puzzles:") 
    for a in args:
        print(a['puzzle_name'])
        
    # exit() # useful to make sure most goes through without actually running anything
    
    # args = args[1:5] # FOR TESTING PURPOSES
    
    while len(args) > 0:
        new_args = []
        for arg in args:
            state = arg['state']
            state.reset()
            res = planner.search(arg)
            if res['status'] == 'solved':
                actions = res['trajectory'].get_actions()
                print("actions = {}".format(actions))
            print("{:s}\tsolution_depth\t{}\texpanded\t{}\tgenerated\t{}\trunning_time\t{}".format(
                    res['puzzle_name'], res['solution_depth'], res['expanded'], res['generated'], res['time']),
                flush=True)
            if res['solution_depth'] == -1:
                arg['node_budget'] *= 4 # search_budget
                new_args.append(arg)
        
        args = new_args
    print("Finished")

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
    
    parser.add_argument('-scheduler', action='store', default='uniform',
                        dest='scheduler',
                        help='Run Bootstrap with a scheduler (either uniform or gbs)')
    
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
        
        for file in puzzle_files:
            if '.' in file:
                continue
            s = WitnessState()
            s.read_state(join(parameters.problems_folder, file))
            states[file] = s
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
    
    print('Loaded ', len(states), ' instances')
#     input_size = s.get_image_representation().shape

#     set_start_method('forkserver', force=True)
            
    KerasManager.register('KerasModel', KerasModel)
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default = 1))
    
    k_expansions = 32
    
#     print('Number of cpus available: ', ncpus)
    
    with KerasManager() as manager:
                
        nn_model = manager.KerasModel()
        bootstrap = None
        
        if parameters.learning_mode:
            bootstrap = Bootstrap(states, parameters.model_name, 
                                  parameters.scheduler, 
                                  ncpus=ncpus, 
                                  initial_budget=int(parameters.search_budget),
                                  gradient_steps=int(parameters.gradient_steps))
        
        PUCT_constructor = PUCT
        if parameters.search_algorithm == 'PUCT_Levi':
            PUCT_constructor = PUCT_Levi
            parameters.search_algorithm = 'PUCT'
        if parameters.search_algorithm == 'PUCT':
       
            bfs_planner = PUCT_constructor(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions, float(parameters.cpuct))
        
            if parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=True)     
            else:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=False)
            
            if parameters.learning_mode:
                #bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_problems(bfs_planner, nn_model)            
            elif parameters.blind_search:
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
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
                bootstrap.solve_problems(bfs_planner, nn_model)            
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
                bootstrap.solve_problems(bfs_planner, nn_model)
            elif parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm) 
                nn_model.load_weights(join('trained_models_online', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
            else:
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))  
                
        if parameters.search_algorithm == 'GBFS':
            #bfs_planner = GBFS(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)
            # NEW: USE A* with w=-1 instead    
            bfs_planner = AStar(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions, -1)
            
            if parameters.learning_mode:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm)
#                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_problems(bfs_planner, nn_model)
            elif parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm) 
                nn_model.load_weights(join('trained_models_online', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
            else:
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))      
            
if __name__ == "__main__":
    main()
    
    
