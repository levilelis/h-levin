import copy
import os
import time
from os.path import join
from models.memory import Memory
from concurrent.futures.process import ProcessPoolExecutor
import heapq
import math

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

	def _solve_uniform_online_curriculum_selectionx(self, planner, nn_model, ordering):
		iteration = 1
		total_expanded = 0
		total_generated = 0

		budget = self._initial_budget
		memory = Memory()
		start = time.time()

		current_solved_puzzles = set()

		marker = 0
		self._batch_size = 1
		batch_problems = {}
		ordered_states = []
		curriculum_puzzles = []

		self._puzzles_probabilities = {}
		for name, pi in ordering:
			self._puzzles_probabilities[name] = [pi, 0]
			ordered_states.append([name, self._states[name]])

		curriculum_puzzles.append(ordered_states[0][0])  # First puzzle in ordering is the first on the curriculum

		while len(current_solved_puzzles) < self._number_problems:
			number_solved = 0

			for name, state in ordered_states:

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
						self._puzzles_probabilities[puzzle_name][1] = puzzle_solution_pi

					batch_problems.clear()

			if memory.number_trajectories() > 0:
				for _ in range(self._gradient_steps):
					loss = nn_model.train_with_memory(memory)
					print('Loss: ', loss)
				memory.clear()
				nn_model.save_weights(join(self._models_folder, 'model_weights'))

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

			print(self._puzzles_probabilities)

			for name, _ in self._states.items():
				self._puzzles_probabilities[name][0] = self._puzzles_probabilities[name][1]
				self._puzzles_probabilities[name][1] = 0

			last_pos = marker
			marker += 4

			self._batch_size = marker - last_pos

			print('Number solved: ', number_solved)
			if number_solved == 0:
				budget *= 2
				print('Budget: ', budget)
				continue

			iteration += 1

	def get_easiest_worsen_puzzle(self, ordering, previous_probs, current_probs):
		"""Verifies the puzzle with the highest solution probability that got harder during training, and return it with its probability and position in ordination"""
		easiest_worsen_puzzle = None
		best_prob = 0
		position = 0
		print(current_probs)
		print(previous_probs)
		for puzzle, prob in previous_probs.items():
			if (previous_probs[puzzle] > current_probs[puzzle]) and (current_probs[puzzle] > best_prob):  # Puzzle got harder and is the highest prob from all that got harder
				easiest_worsen_puzzle = puzzle
				best_prob = current_probs[puzzle]

		for i in range(len(ordering)):
			if ordering[i][0] == easiest_worsen_puzzle:
				position = i
				break

		return [easiest_worsen_puzzle, best_prob], position

	def verify_best_paths(self, planner, all_paths, nn_model):
		"""Returns an array with the best probabilities of solution for every puzzle, among all their solution paths"""
		puzzles_prob = {}
		print("Verifying probabilities of puzzles:")
		for puzzle, paths in all_paths.items():
			if puzzle in self._states.keys():
				best_prob = 0  # Highest prob of all paths

				for p in paths:
					print('working on puzzle', puzzle, 'path:', p)
					current_prob = planner.verify_path_probability(self._states[puzzle], p, nn_model)
					if current_prob > best_prob:
						best_prob = current_prob

				puzzles_prob[puzzle] = best_prob

		return puzzles_prob

	def _solve_uniform_online_curriculum_selection(self, planner, nn_model, ordering, all_paths):
		marker = 0  # Used to tell the position of the last selected puzzle on the ordering
		states = {}
		ordered_states = []
		curriculum_puzzles = []
		previous_probabilities = {}
		self._current_solved_puzzles = set()

		for name, pi in ordering:
			previous_probabilities[name] = float(pi)
			ordered_states.append([name, self._states[name]])

		curriculum_puzzles.append(ordered_states[marker][0])  # First puzzle in ordering is the first on the curriculum
		with open(join(self._log_folder + self._model_name + '_curriculum_puzzles'), 'a') as result_file:
							result_file.write("{:s}".format(ordered_states[0][0]))
							result_file.write('\n')

		puzzle_file = ordered_states[marker][0]  # First train only with first puzzle
		states[puzzle_file] = self._states[puzzle_file]

		self._solve_uniform_online_train_at_end(planner, nn_model, states)  # First training

		while len(self._current_solved_puzzles) < self._number_problems:
			current_probabilities = self.verify_best_paths(planner, all_paths, nn_model)
			chosen_puzzle, position = self.get_easiest_worsen_puzzle(ordering, previous_probabilities, current_probabilities)

			if chosen_puzzle[0] is not None:
				print("Chosen Puzzle is", chosen_puzzle[0])
				print("Previous Prob:", previous_probabilities[chosen_puzzle[0]])
				print("Current Prob:", chosen_puzzle[1])
				curriculum_puzzles.append(chosen_puzzle)

				with open(join(self._log_folder + self._model_name + '_curriculum_puzzles'), 'a') as result_file:
							result_file.write("{:s}".format(chosen_puzzle[0]))
							result_file.write('\n')

				puzzles = ordering[marker+1:position+1]
				states = {}
				for p in puzzles:
					states[p] = self._states[p]

				print("Training with", states.keys())
				self._solve_uniform_online_train_at_end(planner, nn_model, states)

				marker = position
				previous_probabilities = copy.deepcopy(current_probabilities)

			else:  # Trains again with only the next puzzle in ordering
				print("No puzzle was chosen in this iteration")
				marker += 1
				if marker >= len(ordering):
					break
				puzzle_file = ordering[marker][0]
				states = {puzzle_file: self._states[puzzle_file]}
				print("Training with", puzzle_file)
				self._solve_uniform_online_train_at_end(planner, nn_model, states)


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

	def _solve_aggregated_online(self, planner, nn_model):
		iteration = 1
		number_solved = 0
		total_expanded = 0
		total_generated = 0

		budget = self._initial_budget
		memory = Memory()
		start = time.time()

		current_solved_puzzles = set()
		state_budget = {}

		for name, state in self._states.items():
			state_budget[name] = self._initial_budget

		while len(current_solved_puzzles) < self._number_problems:
			number_solved = 0
			batch_problems = {}
			for name, state in self._states.items():
				if name in current_solved_puzzles:
					continue
				batch_problems[name] = state

			with ProcessPoolExecutor(max_workers = self._ncpus) as executor:
				# args = ((state, name, budget, nn_model) for name, state in batch_problems.items())
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
																				 self._number_problems - len(current_solved_puzzles),
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
			batch_problems.clear()

			print('Number solved: ', number_solved)
			"""if number_solved == 0:
				#budget *= 2
				budget += 1  # Trying the older code's strategy
				print('Budget: ', budget)
				continue
			else:"""
			"""if number_solved != 0:
				# budget = self._initial_budget
				for name, state in self._states.items():  # Resetting budget after train
					state_budget[name] = self._initial_budget
			iteration += 1"""

	def solve_problems(self, planner, nn_model, ordering=None, all_paths=None):
		if self._scheduler == 'gbs':
			self._solve_gbs(planner, nn_model)
		elif self._scheduler == 'online':
			self._solve_uniform_online(planner, nn_model)
		elif self._scheduler == 'pgbs':
			self._parallel_gbs(planner, nn_model)
		elif self._scheduler == 'curriculum':
			self._solve_uniform_online_curriculum_selection(planner, nn_model, ordering, all_paths)
		elif self._scheduler == 'aggregated':
			self._solve_aggregated_online(planner, nn_model)
		else:
			self._solve_uniform(planner, nn_model)