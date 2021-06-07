import copy
import os
import time
from os.path import join
from models.memory import Memory
from concurrent.futures.process import ProcessPoolExecutor
import heapq
import math
from models.model_wrapper import KerasManager, KerasModel
from search.bfs_levin import TreeNode
import numpy as np
import tracemalloc
import gc

class ProblemNode:
	def __init__(self, k, n, name, instance):
		self._k = k
		self._n = n
		self._name = name
		self._instance = instance

		self._cost = 125 * (4 ** (self._k - 1)) * self._n * self._k * (self._k + 1)

	def __lt__(self, other):
		"""
		Function less-than used by the heap
		"""
		if self._cost != other._cost:
			return self._cost < other._cost
		else:
			return self._k < other._k

	def get_budget(self):
		budget = 125 *  (4 ** (self._k - 1)) * self._n - (125 * (4 ** (self._k - 1) * (self._n - 1)))
		return budget

	def get_name(self):
		return self._name

	def get_k(self):
		return self._k

	def get_n(self):
		return self._n

	def get_instance(self):
		return self._instance

class GBS:
	def __init__(self, states, planner):
		self._states = states
		self._number_problems = len(states)

		self._planner = planner

		# maximum budget
		self._kmax = 10

		# counter for the number of iterations of the algorithm, which is marked by the number of times we train the model
		self._iteration = 1

		# number of problemsm solved in a given iteration of the procedure
		self._number_solved = 0

		# total number of nodes expanded
		self._total_expanded = 0

		# total number of nodes generated
		self._total_generated = 0

		# data structure used to store the solution trajectories
		self._memory = Memory()

		# open list of the scheduler
		self._open_list = []

		# dictionary storing the problem instances to be solved
		self._problems = {}

		self._last_tried_instance = [0 for _ in range(0, self._kmax + 1)]
		self._has_solved = {} #[False for _ in range(0, self._number_problems + 1)]

		# populating the problems dictionary
		id_puzzle = 1
		for name, instance in self._states.items():
			self._problems[id_puzzle] = (name, instance)
			self._has_solved[id_puzzle] = False
			id_puzzle += 1

		# create ProblemNode for the first puzzle in the list of puzzles to be solved
		node = ProblemNode(1, 1, self._problems[1][0], self._problems[1][1])

		# insert such node in the open list
		heapq.heappush(self._open_list, node)

		# list containing all puzzles already solved
		self._closed_list = set()

	def closed_list(self):
		return self._closed_list

	def run_prog(self, k, budget, nn_model):

		last_idx = self._last_tried_instance[k]
		idx = last_idx + 1

		while idx < self._number_problems + 1:
			if not self._has_solved[idx]:
				break
			idx += 1

		if idx > self._number_problems:
			return True, None, None, None, None

		self._last_tried_instance[k] = idx

		data = (self._problems[idx][1], self._problems[idx][0], budget, nn_model)
		is_solved, trajectory, expanded, generated, _ = self._planner.search_for_learning(data)

		if is_solved:
			self._has_solved[idx] = True

		return idx == self._number_problems, is_solved, trajectory, expanded, generated


	def solve(self, nn_model, max_steps):
		# counter for the number of steps in this schedule
		number_steps = 0

		# number of problems solved in this iteration
		number_solved_iteration = 0

		# reset the current memory
		self._memory.clear()

		# main loop of scheduler, iterate while there are problems still to be solved
		while len(self._open_list) > 0 and len(self._closed_list) < self._number_problems:

			# remove the first problem from the scheduler's open list
			node = heapq.heappop(self._open_list)

			# if the problem was already solved, then we bypass the solving part and
			# add the children of this node into the open list.
			if False and node.get_n() in self._closed_list:
				# if not halted
				if node.get_n() < self._number_problems:
					# if not solved, then reinsert the same node with a larger budget into the open list
					child = ProblemNode(node.get_k(),
										node.get_n() + 1,
										self._problems[node.get_n() + 1][0],
										self._problems[node.get_n() + 1][1])
					heapq.heappush(self._open_list, child)

				# if the first problem in the list then insert it with a larger budget
				if node.get_n() == 1:
					# verifying whether there is a next puzzle in the list
					if node.get_k() + 1 < self._kmax:
						# create an instance of ProblemNode for the next puzzle in the list of puzzles.
						child = ProblemNode(node.get_k() + 1,
											1,
											self._problems[1][0],
											self._problems[1][1])
						# add the node to the open list
						heapq.heappush(self._open_list, child)
				continue


#             data = (node.get_instance(), node.get_name(), node.get_budget(), nn_model)
#             solved, trajectory, expanded, generated, _ = self._planner.search_for_learning(data)
			has_halted, solved, trajectory, expanded, generated = self.run_prog(node.get_k(), node.get_budget(), nn_model)

			# if not halted
			#if node.get_n() < self._number_problems:
			if not has_halted:
				self._total_expanded += expanded
				self._total_generated += generated


				# if not solved, then reinsert the same node with a larger budget into the open list
				child = ProblemNode(node.get_k(),
									node.get_n() + 1,
									self._problems[node.get_n() + 1][0],
									self._problems[node.get_n() + 1][1])
				heapq.heappush(self._open_list, child)

			if solved is not None and solved:
				# if it has solved, then add the puzzle's name to the closed list
				self._closed_list.add(node.get_n())
				# store the trajectory as training data
				self._memory.add_trajectory(trajectory)
				# increment the counter of problems solved, for logging purposes
				self._number_solved += 1
				number_solved_iteration += 1

			# if this is the puzzle's first trial, then share its computational budget with the next puzzle in the list
			if node.get_n() == 1:
				# verifying whether there is a next puzzle in the list
				if node.get_k() + 1 < self._kmax:
					# create an instance of ProblemNode for the next puzzle in the list of puzzles.
					child = ProblemNode(node.get_k() + 1,
										1,
										self._problems[1][0],
										self._problems[1][1])
					# add the node to the open list
					heapq.heappush(self._open_list, child)

			# increment the number of problems attempted solve
			number_steps += 1

			# if exceeds the maximum of steps allowed, then return training data, expansions, generations
			if number_steps >= max_steps:
				return self._memory, self._total_expanded, self._total_generated, number_solved_iteration, self

		return self._memory, self._total_expanded, self._total_generated, number_solved_iteration, self


class Bootstrap:
	def __init__(self, states, output, scheduler, ncpus=1, initial_budget=2000, gradient_steps=10):
		self._states = states
		self._model_name = output
		self._number_problems = len(states)

		self._ncpus = ncpus
		self._initial_budget = initial_budget
		self._gradient_steps = gradient_steps
#         self._k = ncpus * 3
		self._batch_size = 32

		self._kmax = 10

		self._scheduler = scheduler

		self._log_folder = 'training_logs/'
		self._models_folder = 'trained_models_online/' + self._model_name

		if not os.path.exists(self._models_folder):
			os.makedirs(self._models_folder)

		if not os.path.exists(self._log_folder):
			os.makedirs(self._log_folder)

	def map_function(self, data):
		gbs = data[0]
		nn_model = data[1]
		max_steps = data[2]

		return gbs.solve(nn_model, max_steps)

	def _parallel_gbs(self, planner, nn_model):
		schedulers = []

		number_problems_per_cpu = math.ceil(self._number_problems / self._ncpus)

		states = {}
		counter_puzzles = 1
		for id_puzzle, instance in self._states.items():
			if counter_puzzles > number_problems_per_cpu:
				gbs = GBS(states, planner)
				schedulers.append(gbs)
				counter_puzzles = 0
				states = {}

			states[id_puzzle] = instance
			counter_puzzles += 1

		if counter_puzzles > 0:
			gbs = GBS(states, planner)
			schedulers.append(gbs)

#         print('Schedulers: ', schedulers)

		number_problems_solved = 0
		problems_solved_iteration = 0
		total_expanded = 0
		total_generated = 0

		start = time.time()
		start_segment = start

		iteration = 1

		# while there are problems yet to be solved
		while number_problems_solved < self._number_problems:

			# invokes planning algorithm for solving the instance represented by node
			with ProcessPoolExecutor(max_workers = self._ncpus) as executor:
				args = ((gbs, nn_model, 10) for gbs in schedulers)
				results = executor.map(self.map_function, args)

			# collect the results of search for the states
			schedulers = []
			memory = Memory()
			for result in results:
				memory.merge_trajectories(result[0])

				total_expanded += result[1]
				total_generated += result[2]

				problems_solved_iteration += result[3]
				number_problems_solved += result[3]

				gbs = result[4]
				schedulers.append(gbs)

			print('Total number of problems solved: ', number_problems_solved)

			# if problems were solved in the previous batch, then use them to train the model
			if memory.number_trajectories() > 0:
				for _ in range(self._gradient_steps):
					# perform a number of gradient descent steps
					loss = nn_model.train_with_memory(memory)
					print(loss)
				memory.clear()
				# saving the weights the latest neural model
				nn_model.save_weights(join(self._models_folder, 'model_weights'))


			# it will report in the log file every 30 minutes of search
			end = time.time()
			if end - start_segment > 1800:

				# readjusting elapsed time
				start_segment = end

				# logging details of the latest iteration
				with open(join(self._log_folder + 'training_bootstrap_' + self._model_name), 'a') as results_file:
					results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(iteration,
																					 problems_solved_iteration,
																					 self._number_problems - number_problems_solved,
																					 total_expanded,
																					 total_generated,
																					 end-start)))
					results_file.write('\n')


				iteration +=1
				problems_solved_iteration = 0


	def _solve_gbs(self, planner, nn_model):

		# counter for the number of iterations of the algorithm, which is marked by the number of times we train the model
		iteration = 1

		# number of problemsm solved in a given iteration of the procedure
		number_solved = 0

		# total number of nodes expanded
		total_expanded = 0

		# total number of nodes generated
		total_generated = 0

		# data structure used to store the solution trajectories
		memory = Memory()

		# start timer for computing the running time of the iterations of the procedure
		start = time.time()
		start_segment = start

		# open list of the scheduler
		open_list = []

		# dictionary storing the problem instances to be solved
		problems = {}
		has_solved_problem = [None]

		# populating the problems dictionary
		id_puzzle = 1
		for _, instance in self._states.items():
			problems[id_puzzle] = instance
			id_puzzle += 1
			has_solved_problem.append(False)

		# create ProblemNode for the first puzzle in the list of puzzles to be solved
		node = ProblemNode(1, 1, problems[1])

		# insert such node in the open list
		heapq.heappush(open_list, node)

		# list containing all puzzles already solved
		closed_list = set()

		# list of problems that will be solved in parallel
		problems_to_solve = {}

		# main loop of scheduler, iterate while there are problems still to be solved
		while len(open_list) > 0 and len(closed_list) < self._number_problems:

			# remove the first problem from the scheduler's open list
			node = heapq.heappop(open_list)

			# if the problem was already solved, then we bypass the solving part and
			# add the children of this node into the open list.
			if node.get_n() in closed_list:
				# if not halted
				if node.get_n() < self._number_problems:
					# if not solved, then reinsert the same node with a larger budget into the open list
					child = ProblemNode(node.get_k(),
										node.get_n() + 1,
										problems[node.get_n() + 1])
					heapq.heappush(open_list, child)

				# if the first problem in the list then insert it with a larger budget
				if node.get_n() == 1:
					# verifying whether there is a next puzzle in the list
					if node.get_k() + 1 < self._kmax:
						# create an instance of ProblemNode for the next puzzle in the list of puzzles.
						child = ProblemNode(node.get_k() + 1,
											1,
											problems[1])
						# add the node to the open list
						heapq.heappush(open_list, child)
				continue

			# append current node in the list of problems to be solved
			problems_to_solve[node.get_n()] = node

			# is there are at least k problems we will attempt to solve them in parallel
			if len(problems_to_solve) >= self._batch_size or len(open_list) == 0:
				# invokes planning algorithm for solving the instance represented by node
				with ProcessPoolExecutor(max_workers = self._ncpus) as executor:
					args = ((p.get_instance(), p.get_n(), p.get_budget(), nn_model) for _, p in problems_to_solve.items())
					results = executor.map(planner.search_for_learning, args)

				# collect the results of search for the states
				for result in results:
					solved = result[0]
					trajectory = result[1]
					total_expanded += result[2]
					total_generated += result[3]
					puzzle_id = result[4]

					# if not halted
					if problems_to_solve[puzzle_id].get_n() < self._number_problems:
						# if not solved, then reinsert the same node with a larger budget into the open list
						child = ProblemNode(problems_to_solve[puzzle_id].get_k(),
											puzzle_id + 1,
											problems[puzzle_id + 1])
						heapq.heappush(open_list, child)

					if solved:
						# if it has solved, then add the puzzle's name to the closed list
						closed_list.add(puzzle_id)
						# store the trajectory as training data
						memory.add_trajectory(trajectory)
						# mark problem as solved
						has_solved_problem[puzzle_id] = True
						# increment the counter of problems solved, for logging purposes
						number_solved += 1

					# if this is the puzzle's first trial, then share its computational budget with the next puzzle in the list
					if puzzle_id == 1:
						# verifying whether there is a next puzzle in the list
						if problems_to_solve[puzzle_id].get_k() + 1 < self._kmax:
							# create an instance of ProblemNode for the next puzzle in the list of puzzles.
							child = ProblemNode(problems_to_solve[puzzle_id].get_k() + 1,
												1,
												problems[1])
							# add the node to the open list
							heapq.heappush(open_list, child)

				# clear the problems to solve
				problems_to_solve.clear()

				if memory.number_trajectories() > 0:
					# perform a number of gradient descent steps
					for _ in range(self._gradient_steps):
						loss = nn_model.train_with_memory(memory)
						print(loss)

					# remove current trajectories from memory
					memory.clear()

					# saving the weights the latest neural model
					nn_model.save_weights(join(self._models_folder, 'model_weights'))

			# if the number of attempts for solving problems is equal to the number of remaining problems and
			# if the procedure solved problems a new problem, then perform learning
			if time.time() - start_segment > 1800 and number_solved > 0:
				# time required in this iteration of the algorithm
				end = time.time()

				# readjusting elapsed time
				start_segment = end

				# logging details of the latest iteration
				with open(join(self._log_folder + 'training_bootstrap_' + self._model_name), 'a') as results_file:
					results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(iteration,
																					 number_solved,
																					 self._number_problems - len(closed_list),
																					 total_expanded,
																					 total_generated,
																					 end-start)))
					results_file.write('\n')

				# set the number of problems solved and trials to zero and increment the iteration counter
				number_solved = 0
				iteration += 1

		# if the system solves all instances and there are new instances to learn from, then log details and train the model
		if number_solved > 0:
			# time required in this iteration of the algorithm
			end = time.time()
			# logging details of the latest iteration
			with open(join(self._log_folder + 'training_bootstrap_' + self._model_name), 'a') as results_file:
				results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(iteration,
																				 number_solved,
																				 self._number_problems - len(closed_list),
																				 total_expanded,
																				 total_generated,
																				 end-start)))
				results_file.write('\n')

	def select_curriculum_puzzle(self, solved_blocks, previous_probs, current_probs):
		"""Checks for the puzzle in the earlier block that had its probability increase and return it with its probability and position in solved_blocks"""
		easiest_worsen_puzzle = None
		best_prob = 0
		position = 999999
		current_pos = 0
		for puzzle, prob in previous_probs.items():
			if previous_probs[puzzle] > current_probs[puzzle]:  # First, check if a puzzle got harder than last iteration
				for number, block in solved_blocks.items():
					if puzzle in block.keys():  # If so, check the block in which it was solved during ordering
						current_pos = number
						break
				if current_pos < position:  # If its' position is earlier than the last best puzzle selected, hold him as the new selected
					position = current_pos
					easiest_worsen_puzzle = puzzle
					best_prob = current_probs[puzzle]

				elif current_pos == position:  # If the last puzzle selected and this one are from the same block, check the one with the highest solving probability
					if current_probs[puzzle] > best_prob:
						easiest_worsen_puzzle = puzzle
						best_prob = current_probs[puzzle]

		return [easiest_worsen_puzzle, best_prob], position

	def get_easiest_worsen_puzzle(self, solved_blocks, previous_probs, current_probs):
		"""Verifies the puzzle with the highest solution probability that got harder during training, and return it with its probability and position in solved_blocks"""
		easiest_worsen_puzzle = None
		best_prob = 0
		position = 0
		# print(current_probs)
		# print(previous_probs)
		for puzzle, prob in previous_probs.items():
			if (previous_probs[puzzle] > current_probs[puzzle]) and (current_probs[puzzle] > best_prob):  # Puzzle got harder and is the highest prob from all that got harder
				easiest_worsen_puzzle = puzzle
				best_prob = current_probs[puzzle]

		"""for i in range(len(ordering)):
			if ordering[i][0] == easiest_worsen_puzzle:
				position = i
				break"""
		for number, block in solved_blocks.items():
			if easiest_worsen_puzzle in block.keys():
				position = number
				break

		return [easiest_worsen_puzzle, best_prob], position

	def verify_best_paths(self, planner, all_paths, nn_model):
		"""Returns an array with the best probabilities of solution for every puzzle, among all their solution paths"""
		puzzles_prob = {}
		print("Verifying probabilities of puzzles...")
		for puzzle, paths in all_paths.items():
			if puzzle in self._states.keys():
				best_prob = 0  # Highest prob of all paths

				for p in paths:
					# print('working on puzzle', puzzle, 'path:', p)
					current_prob = planner.verify_path_probability(self._states[puzzle], p, nn_model)
					if current_prob > best_prob:
						best_prob = current_prob

				puzzles_prob[puzzle] = best_prob

		return puzzles_prob

	def verify_path_probability(self, state, path, nn_model):
		"""
		This function receives a puzzle state and one of its solution paths and checks the probability (prob)
		of the current policy solving that instance.
		Calculated by the productory of probabilities of actions in path.
		"""
		state.clear_path()

		parent = None
		child = state
		p = 0
		depth = 1
		last_action = -1

		for action in path:
			_, action_distribution = nn_model.predict(np.array([child.get_image_representation()]))
			action_distribution_log = np.log(action_distribution)

			node = TreeNode(parent, child, p, depth, -1, last_action)

			node.set_probability_distribution_actions(action_distribution_log[0])
			probability_distribution_log = node.get_probability_distribution_actions()

			child = copy.deepcopy(node.get_game_state())
			child.apply_action(action)

			parent = copy.deepcopy(node)
			p = node.get_p() + probability_distribution_log[action]
			depth = node.get_g() + 1
			last_action = action

		state.clear_path()

		return math.exp(p)

	def mult_verify_current_probabilities(self, solutions, models):
		"""Returns an array with the current average probabilities of solution for every puzzle between each model, using the solution path found in the training of the ordering CNN (CNN1)"""
		puzzles_prob = {}
		print("Verifying current average solution probabilities for the", len(models), "models ...")
		for puzzle in self._states.keys():
			sum_probs = 0
			for model in models:
				prob = self.verify_path_probability(self._states[puzzle], solutions[puzzle], model)
				sum_probs += prob
			avg_prob = sum_probs/len(models)
			puzzles_prob[puzzle] = avg_prob

		return puzzles_prob

	def verify_current_probabilities(self, solutions, nn_model):
		"""Returns an array with the current probabilities of solution for every puzzle, using the solution path found in the training of the ordering CNN (CNN1)"""
		puzzles_prob = {}
		print("Verifying current solution probabilities...")
		#for puzzle in solutions.keys():
		for puzzle in self._states.keys():
			current_prob = self.verify_path_probability(self._states[puzzle], solutions[puzzle], nn_model)
			puzzles_prob[puzzle] = current_prob

		return puzzles_prob

	def test_current_probs(self, test_states, planner, solutions, nn_model):
		puzzles_prob = {}
		print("Verifying current solution probabilities...")
		for puzzle in solutions.keys():
			if puzzle in test_states.keys():
				current_prob = planner.verify_path_probability(test_states[puzzle], solutions[puzzle], nn_model)
				puzzles_prob[puzzle] = current_prob

		return puzzles_prob

	def train_model(self, data):
		model = data[0]
		memory = data[1]
		batch_size = data[2]
		loss = 1
		while loss > 0.1:
			loss = model.train_with_state_action(memory, batch_size)
			print('Loss: ', loss)

		model.save_weights(join(self._models_folder, 'model_weights'))
		return loss

	def _curriculum_selection_only(self, models, ordering, solutions, solved_blocks):
		print("STARTING CURRICULUM SELECTION!")
		marker = 1  # Used to tell the solved block of the last selected puzzle on solved_blocks
		batch_problems = {}
		curriculum_puzzles = []
		trained_puzzles = set()

		iteration = 1
		memory = Memory()

		previous_probabilities = self.mult_verify_current_probabilities(solutions, models)  # Verifying with new CNN (CNN2 with random initialization)
		curriculum_puzzles.append(ordering[0][0])  # First puzzle in ordering is the first on the curriculum (if we work with blocks, how can we choose the very first puzzle?)
		with open(join(self._log_folder + self._model_name + '_curriculum_puzzles'), 'a') as result_file:
							result_file.write("{:s}".format(ordering[0][0]))
							result_file.write('\n')

		trajectories = {}  # Trajectories from puzzles of this iteration
		# Train with all puzzles in the first solved block
		for p in solved_blocks[marker].keys():
			batch_problems[p] = self._states[p]
			trajectories[p] = solved_blocks[marker][p]
			del self._states[p]  # Clear dictionary of the states that were already solved
			del previous_probabilities[p]
		first_iteration = True
		gc.collect()

		while len(trained_puzzles) < self._number_problems:
			if not first_iteration:  # We must train at least once to start using these
				current_probabilities = self.mult_verify_current_probabilities(solutions, models)
				chosen_puzzle, position = self.get_easiest_worsen_puzzle(solved_blocks, previous_probabilities, current_probabilities)  # Getting the higher probability in general
				# chosen_puzzle, position = self.select_curriculum_puzzle(solved_blocks, previous_probabilities, current_probabilities)  # Getting the earlier in the blocks
				print(current_probabilities)
				if chosen_puzzle[0] is not None:
						print("Chosen Puzzle is", chosen_puzzle[0])
						print("Previous Prob:", previous_probabilities[chosen_puzzle[0]])
						print("Current Prob:", chosen_puzzle[1])
						curriculum_puzzles.append(chosen_puzzle)

						with open(join(self._log_folder + self._model_name + '_curriculum_puzzles'), 'a') as result_file:
									result_file.write("{:s}".format(chosen_puzzle[0]))
									result_file.write('\n')

						"""puzzles = ordering[marker+1:position+1]
						for p, _ in puzzles:
							batch_problems[p] = self._states[p]"""

						batch_problems = {}
						trajectories = {}  # Trajectories from puzzles of this iteration
						for i in range(marker+1, position+1):
							if i in solved_blocks.keys():
								for p in solved_blocks[i].keys():
									batch_problems[p] = self._states[p]
									trajectories[p] = solved_blocks[i][p]


						print("Training with", batch_problems.keys(), 'from blocks', str(marker+1), 'to', position)

						marker = position
						previous_probabilities = copy.deepcopy(current_probabilities)

						for p in batch_problems.keys():  # Clear dictionary of the states that were already solved
							del self._states[p]
							del previous_probabilities[p]
						gc.collect()

				else:  # Trains again with only the next puzzle in ordering
					print("No puzzle was chosen in this iteration")
					marker += 1
					if marker > len(solved_blocks.keys()):
						break
					batch_problems = {}
					trajectories = {}  # Trajectories from puzzles of this iteration
					# Train with all puzzles in the next solved block
					for p in solved_blocks[marker].keys():
						batch_problems[p] = self._states[p]
						trajectories[p] = solved_blocks[marker][p]
						del self._states[p]  # Clear dictionary of the states that were already solved
						del previous_probabilities[p]
					gc.collect()

					print("Training with", batch_problems.keys(), 'from block', marker)

			first_iteration = False
			current_solved_puzzles = set()

			for p in batch_problems.keys():
				trained_puzzles.add(p)
				memory.add_trajectory(trajectories[p])

			memory.preprocess_data()
			print('preprocessed pairs:', len(memory.get_preprocessed_pairs()))
			with ProcessPoolExecutor(max_workers = self._ncpus) as executor:
				args = ((model, memory, 1024) for model in models)
				results = executor.map(self.train_model, args)
			for result in results:
				last_loss = result
				print('last_loss:', last_loss)
			# For memory usage tests only, comment when sending to cluster!!!
			"""current, peak = tracemalloc.get_traced_memory()
			print('Current memory usage is', str(round((current/10**6), 2)) + 'MB', 'with peak of', str(round((peak/10**6), 2)) + 'MB')"""

			iteration += 1
			new_batch_problems = {}
			for p in batch_problems.keys():  # Create a new training batch without puzzles already solved
				if p not in current_solved_puzzles:
					new_batch_problems[p] = batch_problems[p]

			batch_problems = new_batch_problems
		#tracemalloc.stop()  # For memory usage tests only, comment when sending to cluster!!!

	def _solve_uniform_online_curriculum_selection(self, planner, nn_model, ordering, solutions):
		marker = 0  # Used to tell the position of the last selected puzzle on the ordering
		batch_problems = {}
		ordered_states = []
		curriculum_puzzles = []
		previous_probabilities = {}
		total_current_solved_puzzles = set()

		iteration = 1
		total_expanded = 0
		total_generated = 0
		budget = self._initial_budget
		memory = Memory()
		start = time.time()

		state_budget = {}

		for name, state in self._states.items():  # Individual budget list
			state_budget[name] = self._initial_budget

		for name, _ in ordering:  # Creating ordered puzzle list, according to first ANN
			#previous_probabilities[name] = float(pi)
			ordered_states.append([name, self._states[name]])

		previous_probabilities = self.verify_current_probabilities(planner, solutions, nn_model)  # Verifying with new CNN (CNN2 with random initialization)

		# Using this to test how the probabilities of puzzle 2x2_269 and his reflections (2x2_152, 2x2_257, 2x2_104) are moving during this process
		"""test_states = {}
		test_states['2x2_269'] = self._states['2x2_269']
		test_states['2x2_152'] = self._states['2x2_152']
		test_states['2x2_257'] = self._states['2x2_257']
		test_states['2x2_104'] = self._states['2x2_104']
		with open(join(self._log_folder + self._model_name + '_2x2_269_reflections_probs'), 'a') as result_file:
				result_file.write("2x2_269: {:e}\n".format(previous_probabilities['2x2_269']))
				result_file.write("2x2_152: {:e}\n".format(previous_probabilities['2x2_152']))
				result_file.write("2x2_257: {:e}\n".format(previous_probabilities['2x2_257']))
				result_file.write("2x2_104: {:e}\n".format(previous_probabilities['2x2_104']))
				result_file.write("\n")"""

		curriculum_puzzles.append(ordered_states[marker][0])  # First puzzle in ordering is the first on the curriculum
		with open(join(self._log_folder + self._model_name + '_curriculum_puzzles'), 'a') as result_file:
							result_file.write("{:s}".format(ordered_states[0][0]))
							result_file.write('\n')

		puzzle_file = ordered_states[marker][0]  # First train only with first puzzle
		batch_problems[puzzle_file] = self._states[puzzle_file]
		first_iteration = True
		del self._states[puzzle_file]  # Clear dictionary of the states that were already solved
		del previous_probabilities[puzzle_file]

		while len(total_current_solved_puzzles) < self._number_problems:
			if not first_iteration:  # We must train at least once to start using these
				#current_probabilities = self.verify_best_paths(planner, all_paths, nn_model)
				current_probabilities = self.verify_current_probabilities(planner, solutions, nn_model)
				chosen_puzzle, position = self.get_easiest_worsen_puzzle(ordering, previous_probabilities, current_probabilities)

				# Using this to test how the probabilities of puzzle 2x2_269 and his reflections (2x2_152, 2x2_257, 2x2_104) are moving during this process
				"""test_probs = self.test_current_probs(test_states, planner, solutions, nn_model)
				with open(join(self._log_folder + self._model_name + '_2x2_269_reflections_probs'), 'a') as result_file:
						result_file.write("2x2_269: {:e}\n".format(test_probs['2x2_269']))
						result_file.write("2x2_152: {:e}\n".format(test_probs['2x2_152']))
						result_file.write("2x2_257: {:e}\n".format(test_probs['2x2_257']))
						result_file.write("2x2_104: {:e}\n".format(test_probs['2x2_104']))
						result_file.write("\n")"""

				if chosen_puzzle[0] is not None:
						# x = ['2x2_269', '2x2_152', '2x2_257', '2x2_104']
						print("Chosen Puzzle is", chosen_puzzle[0])
						print("Previous Prob:", previous_probabilities[chosen_puzzle[0]])
						print("Current Prob:", chosen_puzzle[1])
						curriculum_puzzles.append(chosen_puzzle)

						"""if chosen_puzzle[0] in x:
							# Using this to test how the probabilities of puzzle 2x2_269 and his reflections (2x2_152, 2x2_257, 2x2_104) are moving during this process
							with open(join(self._log_folder + self._model_name + '_2x2_269_reflections_probs'), 'a') as result_file:
								result_file.write("{:s}: {:e} (chosen)\n".format(chosen_puzzle[0], chosen_puzzle[1]))"""

						with open(join(self._log_folder + self._model_name + '_curriculum_puzzles'), 'a') as result_file:
									result_file.write("{:s}".format(chosen_puzzle[0]))
									result_file.write('\n')

						puzzles = ordering[marker+1:position+1]
						batch_problems = {}
						for p, _ in puzzles:
							batch_problems[p] = self._states[p]

						print("Training with", batch_problems.keys())
						# self._solve_uniform_online_train_at_end(planner, nn_model, states)

						marker = position
						previous_probabilities = copy.deepcopy(current_probabilities)

						for p in batch_problems.keys():  # Clear dictionary of the states that were already solved
							del self._states[p]
							del previous_probabilities[p]

				else:  # Trains again with only the next puzzle in ordering
					print("No puzzle was chosen in this iteration")
					marker += 1
					if marker >= len(ordering):
						break
					puzzle_file = ordering[marker][0]
					batch_problems = {puzzle_file: self._states[puzzle_file]}
					print("Training with", puzzle_file)
					# self._solve_uniform_online_train_at_end(planner, nn_model, states)
					del self._states[puzzle_file]  # Clear dictionary of the states that were already solved
					del previous_probabilities[puzzle_file]

			first_iteration = False
			current_solved_puzzles = set()
			batch_size = len(batch_problems)

			while len(current_solved_puzzles) < batch_size:
				number_solved = 0

				with ProcessPoolExecutor(max_workers = self._ncpus) as executor:
					args = ((state, name, state_budget[name], nn_model) for name, state in batch_problems.items())
					results = executor.map(planner.search_for_learning, args)
				for result in results:
					has_found_solution = result[0]
					trajectory = result[1]
					total_expanded += result[2]
					total_generated += result[3]
					puzzle_name = result[4]
					state_budget[puzzle_name] = result[5]  # new budget for this particular puzzle

					if has_found_solution:
						memory.add_trajectory(trajectory)
						number_solved += 1
						current_solved_puzzles.add(puzzle_name)
						total_current_solved_puzzles.add(puzzle_name)
						puzzle_solution_pi = trajectory.get_solution_pi()
						print(puzzle_name, 'pi:', puzzle_solution_pi)
						with open(join(self._log_folder + 'training_bootstrap_' + self._model_name + '_puzzles_ordering'), 'a') as result_file:
							result_file.write("{:s}, {:e}, {:d}".format(puzzle_name, puzzle_solution_pi, state_budget[puzzle_name]))
							result_file.write('\n')

						if 'witness' in puzzle_name:
							with open(join(self._log_folder + 'training_bootstrap_' + self._model_name + '_witness_ordering'), 'a') as result_file:
								result_file.write("{:s}, {:e}, {:d}, {:d}".format(puzzle_name, puzzle_solution_pi, state_budget[puzzle_name], iteration))
								result_file.write('\n')

				end = time.time()
				with open(join(self._log_folder + 'training_bootstrap_' + self._model_name), 'a') as results_file:
					results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(iteration,
																					 number_solved,
																					 self._number_problems - len(total_current_solved_puzzles),
					                                                                 len(memory.get_preprocessed_pairs()),
																					 budget,
																					 total_expanded,
																					 total_generated,
																					 end-start)))
					results_file.write('\n')

				if number_solved != 0:
					# budget = self._initial_budget
					for name, state in self._states.items():  # Resetting budget after train
						state_budget[name] = self._initial_budget
				else:  # If none solved, skip training
					continue

				memory.preprocess_data()
				print('preprocessed pairs:', len(memory.get_preprocessed_pairs()))
				#if memory.number_trajectories() > 0:
					#for _ in range(self._gradient_steps):
				loss = 1
				while loss > 0.1:
					#loss = nn_model.train_with_memory(memory)
					loss = nn_model.train_with_state_action(memory, 1024)
					print('Loss: ', loss)

				iteration += 1
				nn_model.save_weights(join(self._models_folder, 'model_weights'))

				new_batch_problems = {}
				for p in batch_problems.keys():  # Create a new training batch without puzzles already solved
					if p not in current_solved_puzzles:
						new_batch_problems[p] = batch_problems[p]

				batch_problems = new_batch_problems

				print('Number solved: ', number_solved)

	def _solve_uniform_online_train_at_end(self, planner, nn_model, states):
		iteration = 1
		number_solved = 0
		total_expanded = 0
		total_generated = 0

		budget = self._initial_budget
		memory = Memory()
		start = time.time()
		solved_puzzles = set()

		while len(solved_puzzles) < len(states):
			number_solved = 0
			batch_problems = {}
			for name, state in states.items():

	#                 if name in current_solved_puzzles:
	#                     continue

				batch_problems[name] = state

				if len(batch_problems) < self._batch_size and len(states) - len(solved_puzzles) > self._batch_size:
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
					print(puzzle_name, has_found_solution)

					if has_found_solution:
						memory.add_trajectory(trajectory)

					if has_found_solution and puzzle_name not in self._current_solved_puzzles:
						number_solved += 1
						self._current_solved_puzzles.add(puzzle_name)
						solved_puzzles.add(puzzle_name)
						puzzle_solution_pi = trajectory.get_solution_pi()
						print(puzzle_name, 'pi:', puzzle_solution_pi)

						with open(join(self._log_folder + 'training_bootstrap_' + self._model_name + '_puzzles_ordering'), 'a') as result_file:
							result_file.write("{:s}, {:e}".format(puzzle_name, puzzle_solution_pi))
							result_file.write('\n')

					batch_problems.clear()

			if memory.number_trajectories() > 0:
				print('memory trajec:', memory.number_trajectories())
				for _ in range(self._gradient_steps):
					loss = nn_model.train_with_memory(memory)
					print('Loss: ', loss)
				memory.clear()
				nn_model.save_weights(join(self._models_folder, 'model_weights'))

			#batch_problems.clear()

			end = time.time()
			with open(join(self._log_folder + 'training_bootstrap_' + self._model_name), 'a') as results_file:
				results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(iteration,
																				 number_solved,
																				 self._number_problems - len(self._current_solved_puzzles),
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
			#number_solved = 0
			iteration += 1

	def _solve_uniform_online(self, planner, nn_model):
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
						puzzle_solution_pi = trajectory.get_solution_pi()
						print(puzzle_name, 'pi:', puzzle_solution_pi)
						with open(join(self._log_folder + 'training_bootstrap_' + self._model_name + '_puzzles_ordering'), 'a') as result_file:
							result_file.write("{:s}, {:e}".format(puzzle_name, puzzle_solution_pi))
							result_file.write('\n')

				if memory.number_trajectories() > 0:
					#for _ in range(self._gradient_steps):
					loss = 1
					while loss > 0.1:
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

	def _solve_uniform(self, planner, nn_model):
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

			with ProcessPoolExecutor(max_workers = self._ncpus) as executor:
				args = ((state, name, budget, nn_model) for name, state in self._states.items())
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
			if number_solved > 0:
				for _ in range(self._gradient_steps):
					loss = nn_model.train_with_memory(memory)
					print(loss)
				memory.clear()

				nn_model.save_weights(join(self._models_folder, 'model_weights'))
			else:
				budget *= 2
				print('Budget: ', budget)
				continue

				iteration += 1

	def _solve_aggregated_online(self, planner, models, cur_gen):
		#tracemalloc.start()  # For memory usage tests only, comment when sending to cluster!!!
		iteration = 1
		number_solved = 0
		total_expanded = 0
		total_generated = 0

		# The next three are used on curriculum selection after this training
		solved_blocks = {}
		trajectories = {}
		solutions = {}
		ordering = []

		budget = self._initial_budget
		memory = Memory()
		start = time.time()

		current_solved_puzzles = set()
		state_budget = {}

		for name, state in self._states.items():
			state_budget[name] = self._initial_budget

		with open(join(self._log_folder + 'training_bootstrap_' + self._model_name + '_puzzles_ordering'), 'a') as result_file:
			result_file.write("{:d}".format(iteration))
			result_file.write('\n')

		while len(current_solved_puzzles) < self._number_problems:
			number_solved = 0
			batch_problems = {}
			for name, state in self._states.items():
				if name in current_solved_puzzles:
					continue
				batch_problems[name] = state

			with ProcessPoolExecutor(max_workers = self._ncpus) as executor:
				# args = ((state, name, budget, nn_model) for name, state in batch_problems.items())
				args = ((state, name, state_budget[name], models) for name, state in batch_problems.items())
				results = executor.map(planner.search_for_learning, args)  # results = executor.map(planner.search_for_learning, args)
			for result in results:
				has_found_solution = result[0]
				trajectory = result[1]
				total_expanded += result[2]
				total_generated += result[3]
				puzzle_name = result[4]
				state_budget[puzzle_name] = result[5]  # new budget for this particular puzzle

				if has_found_solution:
					memory.add_trajectory(trajectory)
					solution = list(reversed(trajectory.get_actions()))
					number_solved += 1
					current_solved_puzzles.add(puzzle_name)
					puzzle_solution_pi = trajectory.get_solution_pi()

					# The next three are used on curriculum selection after this training
					trajectories[puzzle_name] = trajectory
					solutions[puzzle_name] = solution
					ordering.append([puzzle_name, self._states[puzzle_name]])

					print(puzzle_name, 'pi:', puzzle_solution_pi)
					with open(join(self._log_folder + 'training_bootstrap_' + self._model_name + '_puzzles_ordering'), 'a') as result_file:
						result_file.write("{:s}, {:e}, {:d}, ".format(puzzle_name, puzzle_solution_pi, state_budget[puzzle_name]))
						for action in solution:
							result_file.write(("{:d} ".format(action)))
						result_file.write('\n')

					if 'witness' in puzzle_name:
						with open(join(self._log_folder + 'training_bootstrap_' + self._model_name + '_witness_ordering'), 'a') as result_file:
							result_file.write("{:s}, {:e}, {:d}, {:d}".format(puzzle_name, puzzle_solution_pi, state_budget[puzzle_name], iteration))
							result_file.write('\n')

			end = time.time()
			with open(join(self._log_folder + 'training_bootstrap_' + self._model_name), 'a') as results_file:
				results_file.write(("{:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:d}, {:f} ".format(iteration,
																				 number_solved,
																				 self._number_problems - len(current_solved_puzzles),
				                                                                 len(memory.get_preprocessed_pairs()),
																				 budget,
																				 total_expanded,
																				 total_generated,
																				 end-start)))
				results_file.write('\n')

			if number_solved != 0:
				for name, state in self._states.items():  # Resetting budget after train
					state_budget[name] = self._initial_budget
			else:  # If none solved, skip training
				continue

			memory.preprocess_data()
			print('preprocessed pairs:', len(memory.get_preprocessed_pairs()))
			with ProcessPoolExecutor(max_workers = self._ncpus) as executor:
				args = ((model, memory, 1024) for model in models)
				results = executor.map(self.train_model, args)
			for result in results:
				last_loss = result
				print('last_loss:', last_loss)
			"""loss = 1
			while loss > 0.1:
				loss = nn_model.train_with_state_action(memory, 1024)
				print('Loss: ', loss)"""
			# For memory usage tests only, comment when sending to cluster!!!
			"""current, peak = tracemalloc.get_traced_memory()
			print('Current memory usage is', str(round((current/10**6), 2)) + 'MB', 'with peak of', str(round((peak/10**6), 2)) + 'MB')"""
			solved_blocks[iteration] = trajectories  # Add the block of puzzles solved this iteration
			iteration += 1
			trajectories = {}  # Get a new block of trajectories for the solved_blocks
			# nn_model.save_weights(join(self._models_folder, 'model_weights'))  # Not saving for now, since we are using multiple models
			batch_problems.clear()

			with open(join(self._log_folder + 'training_bootstrap_' + self._model_name + '_puzzles_ordering'), 'a') as result_file:
				result_file.write("{:d}".format(iteration))
				result_file.write('\n')

			print('Number solved: ', number_solved)

		print(cur_gen, list(models)[0].get_domain())
		# Used for, after training and having the ordering, train a new ANN to select the curriculum puzzles if cur_gen=True
		if list(models)[0].get_domain() == 'Witness' and cur_gen:
			num_models = self._ncpus
			selector_models = set()
			KerasManager.register('KerasModel', KerasModel)
			with KerasManager() as manager:
				for i in range(num_models):
					nn_model_selector = manager.KerasModel()
					nn_model_selector.initialize('CrossEntropyLoss', 'Levin', domain='Witness', two_headed_model=False)
					selector_models.add(nn_model_selector)

				# Trying to free memory
				del memory
				del models
				del planner
				gc.collect()
				# For memory usage tests only, comment when sending to cluster!!!
				"""current, peak = tracemalloc.get_traced_memory()
				print('Current memory usage is', str(round((current/10**6), 2)) + 'MB', 'with peak of', str(round((peak/10**6), 2)) + 'MB')"""

				self._curriculum_selection_only(selector_models, ordering, solutions, solved_blocks)

	def solve_problems(self, planner, nn_model, ordering=None, solutions=None, cur_gen=False):
		if self._scheduler == 'gbs':
			self._solve_gbs(planner, nn_model)
		elif self._scheduler == 'online':
			self._solve_uniform_online(planner, nn_model)
		elif self._scheduler == 'pgbs':
			self._parallel_gbs(planner, nn_model)
		elif self._scheduler == 'curriculum':
			self._solve_uniform_online_curriculum_selection(planner, nn_model, ordering, solutions)
		elif self._scheduler == 'aggregated':
			self._solve_aggregated_online(planner, nn_model, cur_gen)
		else:
			self._solve_uniform(planner, nn_model)