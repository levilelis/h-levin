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
from search.bfs_levin_mult import BFSLevinMult
from domains.sliding_tile_puzzle import SlidingTilePuzzle
from domains.sokoban import Sokoban
from search.puct import PUCT
from bootstrap import Bootstrap
from multiprocessing import set_start_method

from game_state import GameState
from bootstrap_dfs_learning_planner import BootstrapDFSLearningPlanner
    
   
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


def test_bootstrap_dfs_lvn_learning_planner(states, output, epsilon, beta, dropout, batch):
    planner = BootstrapDFSLearningPlanner (beta, dropout,
                                           batch)  # FD: creates an instance of the class "BootstrapDFSLearningPlanner"
    models_folder = output + '_models'
    if not os.path.exists (models_folder):
        os.makedirs (models_folder)

    state_budget = {}  # FD: empty dictionary containing the budget
    for file, state in states.items ():  # FD: recall: states is a dictionary containing the -- I suppose -- initial states of each puzzle
        state_budget[state] = 1  # FD: initialize budget to 1 for each state in the states dictionary

    unsolved_puzzles = states  # FD: initially all puzzles -- represented by their initial states in the "states" dictionary are unsolved
    id_solved = 1  # FD: ID of puzzles solved
    id_batch = 1

    with open (join (output + '_puzzle_names_ordering_bootstrap'), 'a') as results_file:
        results_file.write (("{:d}".format (id_batch)))
        results_file.write ('\n')

    start = time.time ()
    while len (unsolved_puzzles) > 0:  # FD: while there is at least one unsolved puzzle
        number_solved = 0
        number_unsolved = 0
        current_unsolved_puzzles = {}

        for file, state in unsolved_puzzles.items ():  # FD: take each file and initial puzzle_state
            state.clear_path ()
            has_found_solution, new_bound = planner.lvn_search_budget_for_learning (state, state_budget[state])

            if has_found_solution:
                cost = planner.get_solution_depth ()
                number_solved += 1
                id_solved += 1

                with open (join (output + '_puzzle_names_ordering_bootstrap'), 'a') as results_file:
                    results_file.write (file + ', ' + str (
                        cost) + ' \n')  # FD:writes the line: "<file name>, <cost>" and moves to next line
            else:  # FD: planner has not yet found the solution
                if new_bound != -1:  # the planner has returned a new "budget" (new_bound) to solve the current puzzle?
                    state_budget[state] = new_bound
                else:
                    state_budget[state] = state_budget[state] + 1  # FD:increase budget by one
                number_unsolved += 1
                current_unsolved_puzzles[file] = state  # FD: create the k, v pair: <file_name>, <puzzle initial state>

        end = time.time ()  # FD: time that it takes to solve all the puzzles with current budget

        with open (join (output + '_log_training_bootstrap'), 'a') as results_file:
            results_file.write (("{:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format (id_batch, number_solved, number_unsolved,
                                                                               planner.size_training_set (),
                                                                               planner.current_budget (), end - start)))
            results_file.write ('\n')

        if number_solved != 0:  # if in the current while loop iteration you solved >= 1 puzzle, then reset planner's budget.
            planner.reset_budget ()  # FD: resets planner's budget to 1, and reset state's budget to 1
            for file, state in states.items ():
                state_budget[state] = 1
        else:
            planner.increase_budget ()  # FD: increases planner's budget by +1
            continue  # if there are 0 puzzles solved, loop back and try to solve all the puzzles again with an increased

        unsolved_puzzles = current_unsolved_puzzles

        # TODO: on each iteration: save the current weights: we only do this if we know that we are goint to train the NN,
        #  because that means that we will change the weights.
        # TODO: we should store: iteration number, weights, puzzles solved in current iteration (P),
        #  puzzles solved so far (T), where P in T,
        #  puzzles yet to be solved (S\P).
        # Note, if no puzzles get solved on iteration i, we increase the budget and loop back. so iteration i might have
        # "sub-iterations" (sub-iteration i1, i2, ..., ik) - maybe new puzzles get added in each sub-iteration.
        # if we have a sub-iteration, it means we did not solve any puzzles and P_new = {}. Also, it means we did not train
        # theta_ik. So, theta_n - theta_ik == theta_n - theta_i and P_new = {}, and thus we do not compute anything.
        # in our log file though, we should save the batch_id,
        # nd under batch_id, the puzzles solved and costs (which indicate the sub-iteration, to some extent). We are already ding this!

        planner.preprocess_data ()
        error = 1
        while error > epsilon:
            error = planner.learn ()  # this trains the policy? -- if there were 0 solved puzzles, then how does the planner learn?
            print (error, epsilon)

        id_batch += 1
        with open (join (output + '_puzzle_names_ordering_bootstrap'), 'a') as results_file:
            results_file.write (("{:d}".format (id_batch)))
            results_file.write ('\n')

    planner.save_model (join (models_folder, 'model_' + output), id_batch)  # FD: "id_batch" is the model id?


def main2():
    if len (sys.argv[1:]) < 1:
        print (
            'Usage for learning a new model: main bootstrap_dfs_lvn_learning_planner <folder-with-puzzles> <output-file> <dropout-rate> <batch-size>')
        print (
            'Usage for using a learned model: main learned_planner <folder-with-puzzles> <output-file> <model-folder>')
        return
    planner_name = sys.argv[1]
    puzzle_folder = sys.argv[2]  # FD: this is the <folder-with-puzzles>

    states = {}  # empty dictionary that will contain the states.
    puzzle_files = [f for f in listdir (puzzle_folder) if
                    isfile (join (puzzle_folder, f))]  # make a list of files in puzzle_folder

    for file in puzzle_files:
        if '.' in file:
            continue  # breaks the current iteration, moves to next iteration of for-loop.
        s = GameState ()  # makes an intance of the class "GameState()"
        s.read_state (
            join (puzzle_folder, file))  # calls method "read_state" and passes the current puzzle_file to this method
        states[file] = s  # adds key = file, value = s to the states dictionary

    if planner_name == 'bootstrap_dfs_lvn_learning_planner':
        output_file = sys.argv[3]
        dropout = float (sys.argv[4])
        batch = int (sys.argv[5])
        beta = 0.0  # beta is an entropy regularizer term; it isn't currently used in the code
        # FD: what is an entropy regularizer?
        test_bootstrap_dfs_lvn_learning_planner (states, output_file, 1e-1, beta, dropout, batch)
        # arguments: states, output, epsilon, beta, dropout, batch

    if planner_name == 'learned_planner':
        output_file = sys.argv[3]
        model_folder = sys.argv[4]
        test_bootstrap_dfs_lvn_learned_model_planner (states, output_file, 0, 1.0, 0, model_folder)
        # FD: inputs are "states, output, beta, dropout, batch, model_folder"
    return


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
        
        if parameters.search_algorithm == 'PUCT':
       
            bfs_planner = PUCT(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions, float(parameters.cpuct))
        
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
                nn_model.load_weights(join('trained_models_large', parameters.model_name, 'model_weights'))
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
                nn_model.load_weights(join('trained_models_large', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
                
        if parameters.search_algorithm == 'LevinMult':
        
            bfs_planner = BFSLevinMult(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)
        
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
                nn_model.load_weights(join('trained_models_large', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
        
        if parameters.search_algorithm == 'AStar':
            bfs_planner = AStar(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)
            
            if parameters.learning_mode and parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm)
#                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_problems(bfs_planner, nn_model)
            elif parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm) 
                nn_model.load_weights(join('trained_models_large', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
            else:
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))  
                
        if parameters.search_algorithm == 'GBFS':
            bfs_planner = GBFS(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)
            
            if parameters.learning_mode:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm)
#                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
                bootstrap.solve_problems(bfs_planner, nn_model)
            elif parameters.use_learned_heuristic:
                nn_model.initialize(parameters.loss_function, parameters.search_algorithm) 
                nn_model.load_weights(join('trained_models_large', parameters.model_name, 'model_weights'))
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
            else:
                search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))      
            
if __name__ == "__main__":
    # main()
    main2()
    
    
