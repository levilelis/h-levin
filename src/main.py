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
from log_reader import TrajectoryGenerator
from multiprocessing import set_start_method

   
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

# def bootstrap_learning_bfs(states, planner, nn_model, output, initial_budget, ncpus):
#  
#     log_folder = 'logs_large/'
#     models_folder = 'trained_models_large/' + output
#      
#     if not os.path.exists(models_folder):
#         os.makedirs(models_folder)
#          
#     if not os.path.exists(log_folder):
#         os.makedirs(log_folder)
#      
#     number_problems = len(states)
#      
#     iteration = 1
#     number_solved = 0
#     total_expanded = 0
#     total_generated = 0
#      
#     budget = initial_budget
#     memory = Memory()
#     start = time.time()
#      
#     current_solved_puzzles = set()
#      
#     while len(current_solved_puzzles) < number_problems:
#         number_solved = 0
#          
#         with ProcessPoolExecutor(max_workers = ncpus) as executor:
#             args = ((state, name, budget, nn_model) for name, state in states.items()) 
#             results = executor.map(planner.search_for_learning, args)
#         for result in results:
#             has_found_solution = result[0]
#             trajectory = result[1]
#             total_expanded += result[2]
#             total_generated += result[3]
#             puzzle_name = result[4]
#              
#             if has_found_solution:
#                 memory.add_trajectory(trajectory)
#               
#             if has_found_solution and puzzle_name not in current_solved_puzzles:
#                 number_solved += 1
#                 current_solved_puzzles.add(puzzle_name)
#          
#         end = time.time()
#         with open(join(log_folder + 'training_bootstrap_' + output), 'a') as results_file:
#             results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(iteration, 
#                                                                              number_solved, 
#                                                                              number_problems - len(current_solved_puzzles), 
#                                                                              budget,
#                                                                              total_expanded,
#                                                                              total_generated, 
#                                                                              end-start)))
#             results_file.write('\n')
#          
#         print('Number solved: ', number_solved)
#         if number_solved > 0:
#             for _ in range(10):
#                 loss = nn_model.train_with_memory(memory)
#                 print(loss)
# #             if number_solved < 20:
# #                 budget *= 2
#             memory.clear()
#              
#             nn_model.save_weights(join(models_folder, 'model_weights')) 
#         else:
#             budget *= 2
#             print('Budget: ', budget)
#             continue
#                                  
#         iteration += 1


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

	parser.add_argument('-of', action='store', dest='ordering_file',
						help='Ordering of solved puzzles (Used on Witness --learn-curriculum)')

	parser.add_argument('-apf', action='store', dest='all_paths_file',
						help='All possible solution paths for each puzzle (Used on Witness --learn-curriculum)')

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
						help='Train a neural model out of the instances from the problem folder')

	parser.add_argument('--curriculum-gen', action='store_true', default=False,
						dest='curriculum_gen',
						help='Train a neural model out of the instances from the problem folder and generate a curriculum')

	parser.add_argument('--multi-model', action='store_true', default=False,
						dest='multi_model',
						help='Uses multiple models (the number of available cpus) to train and generate a curriculum')

	parser.add_argument('-test', action='store_true', default=False,
						dest='test_mode',
						help='Enters in this code test mode')

	parameters = parser.parse_args()

	states = {}
	ordering = None
	all_paths = None
	solutions = None

	"""if parameters.ordering_file:  # Used for --learn-curriculum, result of the first Neural Network training
		with open(parameters.ordering_file, 'r') as file:
			ordering = []
			solutions = {}
			lines = file.readlines()
			for line in lines:
				line_split = line.strip().split(', ')
				puzzle = line_split[0]
				pi = line_split[1]
				path = line_split[3]
				ordering.append([puzzle, pi])

				actions = path.split(' ')
				actions = actions[1:len(actions)-1]
				solution = []
				for action in actions:
					solution.append(int(action))
				solutions[puzzle] = solution


	if parameters.all_paths_file:  # Used for --learn-curriculum
		all_paths = {}
		with open(parameters.all_paths_file, "r") as stream:
			for line in stream:
				data = line.split('\n')
				# Read and treat data to endup in the model: {'puzzle': [[path1], [path2], ..., [pathn]}
				for data_line in data:
					if ':' in data_line:
						puzzle = data_line.split(':')[0]
						paths = data_line.split(':')[1]
						actions = paths.split(' ')
						actions = actions[1:len(actions)-1]
						path = []
						all_paths_for_puzzle = []
						for action in actions:
							if '|' not in action:
								path.append(int(action))
							else:
								all_paths_for_puzzle.append(path)
								path = []
						all_paths[puzzle] = all_paths_for_puzzle"""

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

	if not parameters.test_mode and not parameters.multi_model:
#       set_start_method('forkserver', force=True)

		KerasManager.register('KerasModel', KerasModel)
		ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default = 3))
		print('ncpus:', ncpus)

		k_expansions = 1  # To ensure BFS ordering

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
					nn_model.initialize(parameters.loss_function, parameters.search_algorithm, domain=parameters.problem_domain, two_headed_model=False)

				if parameters.learning_mode:
	#                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
					# bootstrap.solve_problems(bfs_planner, nn_model, ordering, all_paths)
					bootstrap.solve_problems(bfs_planner, nn_model, ordering, solutions)
				elif parameters.blind_search:
					search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))
				else:
					nn_model.load_weights(join('trained_models_online', parameters.model_name, 'model_weights'))
					search(states, bfs_planner, nn_model, ncpus, int(parameters.time_limit), int(parameters.search_budget))

			if parameters.search_algorithm == 'LevinMult':

				bfs_planner = BFSLevinMult(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)

				if parameters.use_learned_heuristic:
					nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=True)
				else:
					nn_model.initialize(parameters.loss_function, parameters.search_algorithm, two_headed_model=False)

				if parameters.learning_mode: # == '--learn' or '--learn-curriculum':
	#                 bootstrap_learning_bfs(states, bfs_planner, nn_model, parameters.model_name, int(parameters.search_budget), ncpus)
					bootstrap.solve_problems(bfs_planner, nn_model, ordering)
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
				bfs_planner = GBFS(parameters.use_heuristic, parameters.use_learned_heuristic, k_expansions)

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

	elif parameters.multi_model:
		ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default = 3))
		print('ncpus:', ncpus)
		num_models = ncpus
		models = set()
		k_expansions = 1  # To ensure BFS ordering

		KerasManager.register('KerasModel', KerasModel)
		with KerasManager() as manager:
			for i in range(num_models):
				nn_model_solver = manager.KerasModel()
				nn_model_solver.initialize('CrossEntropyLoss', 'Levin', domain='Witness', two_headed_model=False)
				models.add(nn_model_solver)

			bootstrap = None

			if parameters.learning_mode:
				bootstrap = Bootstrap(states, parameters.model_name,
									  parameters.scheduler,
									  ncpus=ncpus,
									  initial_budget=int(parameters.search_budget),
									  gradient_steps=int(parameters.gradient_steps))

			if parameters.search_algorithm == 'Levin':
				bfs_planner = BFSLevin(parameters.use_heuristic, parameters.use_learned_heuristic, False, k_expansions, float(parameters.mix_epsilon))
			else:
				bfs_planner = BFSLevin(parameters.use_heuristic, parameters.use_learned_heuristic, True, k_expansions, float(parameters.mix_epsilon))

			bootstrap.solve_problems(bfs_planner, models, ordering, solutions, parameters.curriculum_gen)

	else:
		"""states['puzzle_1'].print()
		print()
		states['puzzle_1'].flip_up_down()
		print()
		states['puzzle_1'].rotate90()
		print()
		states['puzzle_1'].flip_up_down()
		print()
		states['puzzle_1'].rotate90()
		print()
		states['puzzle_1'].flip_up_down()
		print()
		states['puzzle_1'].rotate90()
		print()
		states['puzzle_1'].flip_up_down()"""

		t_gen = TrajectoryGenerator("orderings/multi-model-31-05-and-01-06/testerino_average_ordering", states)
		solved_blocks, solutions = t_gen.get_solved_blocks_and_solutions()

		KerasManager.register('KerasModel', KerasModel)
		ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK', default = 3))
		print('ncpus:', ncpus)

		k_expansions = 1  # To ensure BFS ordering

		with KerasManager() as manager:

			nn_model = manager.KerasModel()

			bootstrap = Bootstrap(states, parameters.model_name,
								  parameters.scheduler,
								  ncpus=ncpus,
								  initial_budget=int(parameters.search_budget),
								  gradient_steps=int(parameters.gradient_steps))

			ordering = [['witness_1', 0]]
			models = set()
			nn_model.initialize('CrossEntropyLoss', 'Levin', domain='Witness', two_headed_model=False)
			models.add(nn_model)
			bootstrap._curriculum_selection_only(models, ordering, solutions, solved_blocks)


if __name__ == "__main__":
	main()


