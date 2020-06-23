import os
import time
from os.path import join
from models.memory import Memory
from concurrent.futures.process import ProcessPoolExecutor
import heapq

class ProblemNode:
    def __init__(self, k, n, name, instance):
        self._k = k
        self._n = n
        self._name = name
        self._instance = instance
                       
        self._cost = (2 ** self._n - 2) #* self._k
                
    def __lt__(self, other):
        """
        Function less-than used by the heap
        """
        if self._cost != other._cost:
            return self._cost < other._cost
        else:
            return self._name < other._name
    
    def get_budget(self):
        return (2 ** self._n - 2) - (2 ** (self._n - 1) - 2)
    
    def get_problem_name(self):
        return self._name
    
    def get_k(self):
        return self._k
    
    def get_n(self):
        return self._n
    
    def get_name(self):
        return self._name
    
    def get_instance(self):
        return self._instance

class Bootstrap:
    def __init__(self, states, output, scheduler, ncpus=1, initial_budget=2000, gradient_steps=10):
        self._states = states
        self._model_name = output
        self._number_problems = len(states)
        self._ncpus = ncpus
        self._initial_budget = initial_budget
        self._gradient_steps = gradient_steps
        self._k = ncpus * 3
        
        self._scheduler = scheduler
        
        self._log_folder = 'logs_large/'
        self._models_folder = 'trained_models_large/' + self._model_name
        
        if not os.path.exists(self._models_folder):
            os.makedirs(self._models_folder)
            
        if not os.path.exists(self._log_folder):
            os.makedirs(self._log_folder)
    
    def _solve_gbs(self, planner, nn_model):
        
        # counter for the number of iterations of the algorithm, which is marked by the number of times we train the model
        iteration = 1
        
        # number of problemsm solved in a given iteration of the procedure
        number_solved = 0
        
        # total number of nodes expanded
        total_expanded = 0
        
        # total number of nodes generated
        total_generated = 0
        
        # number of times we try to solve a problem in a given iteration
        number_trials = 0
        
        # data structure used to store the solution trajectories 
        memory = Memory()
        
        # start timer for computing the running time of the iterations of the procedure
        start = time.time()
        
        # open list of the scheduler
        open_list = []
        
        # dictionary storing the problem instances to be solved 
        problems = {}
        
        # populating the problems dictionary
        id_puzzle = 1
        for name, instance in self._states.items():
            problems[id_puzzle] = (name, instance)
            id_puzzle += 1 
                   
        # create ProblemNode for the first puzzle in the list of puzzles to be solved
        # problems[1][0] is the puzzle's name
        # problems[1][1] is the puzzle instance
        node = ProblemNode(1, 1, problems[1][0], problems[1][1])
        
        # insert such node in the open list
        heapq.heappush(open_list, node)
        
        # list containing all puzzles already solved
        closed_list = set()
        
        # list of problems that will be solved in parallel
        problems_to_solve = {}
        
        # main loop of scheduler, iterate while there are problems still to be solved
        while len(open_list) > 0:
            
            # remove the first problem from the scheduler's open list
            node = heapq.heappop(open_list)
            
            # if the problem was already solved, then continue to the next unsolved problem
            if node.get_name() in closed_list:
                continue
            
            # append current node in the list of problems to be solved
            problems_to_solve[node.get_name()] = node
            
            # is there are at least k problems we will attempt to solve them in parallel
            if len(problems_to_solve) >= self._k or len(open_list) == 0:
                # invokes planning algorithm for solving the instance represented by node
                with ProcessPoolExecutor(max_workers = self._ncpus) as executor:
                    args = ((p.get_instance(), p.get_name(), p.get_budget(), nn_model) for _, p in problems_to_solve.items()) 
                    results = executor.map(planner.search_for_learning, args)
                # collect the results of search for the states
                for result in results:
                    solved = result[0]
                    trajectory = result[1]
                    total_expanded += result[2]
                    total_generated += result[3]
                    puzzle_name = result[4]
                    
                    number_trials += 1
                    
#                     if number_trials % 10 == 0:
#                         print('Number of Trials: ', number_trials)
            
                    if not solved:
                        # if not solved, then reinsert the same node with a larger budget into the open list              
                        child = ProblemNode(problems_to_solve[puzzle_name].get_k(), 
                                            problems_to_solve[puzzle_name].get_n() + 1, 
                                            puzzle_name, 
                                            problems_to_solve[puzzle_name].get_instance())
                        heapq.heappush(open_list, child)
                    else:
                        # if it has solved, then add the puzzle's name to the closed list
                        closed_list.add(puzzle_name)
                        # store the trajectory as training data
                        memory.add_trajectory(trajectory)
                        # increment the counter of problems solved, for logging purposes
                        number_solved += 1
                        
                    # if this is the puzzle's first trial, then share its computational budget with the next puzzle in the list
                    if problems_to_solve[puzzle_name].get_n() == 1:
                        # verifying whether there is a next puzzle in the list
                        if problems_to_solve[puzzle_name].get_k() + 1 < len(problems):
                            # create an instance of ProblemNode for the next puzzle in the list of puzzles. Here,
                            # problems[problems_to_solve[puzzle_name].get_k() + 1][0] is the puzzle's name
                            # problems[problems_to_solve[puzzle_name].get_k() + 1][0] is the puzzle's instance
                            child = ProblemNode(problems_to_solve[puzzle_name].get_k() + 1, 
                                                problems_to_solve[puzzle_name].get_n(), 
                                                problems[problems_to_solve[puzzle_name].get_k() + 1][0], 
                                                problems[problems_to_solve[puzzle_name].get_k() + 1][1])
                            # add the node to the open list
                            heapq.heappush(open_list, child)
                            
                # clear the problems to solve
                problems_to_solve.clear()
            
            # if the number of attempts for solving problems is equal to the number of remaining problems and
            # if the procedure solved problems a new problem, then perform learning 
            if number_trials >= self._number_problems - len(closed_list) and number_solved > 0:
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
                
                # perform 10 iterations of gradient descent
                for _ in range(self._gradient_steps):
                    loss = nn_model.train_with_memory(memory)
                    print(loss)
                
                # saving the weights the latest neural model
                nn_model.save_weights(join(self._models_folder, 'model_weights'))
                
                # set the number of problems solved and trials to zero and increment the iteration counter
                number_solved = 0
                number_trials = 0
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
            
            # perform 10 iterations of gradient descent
            for _ in range(self._gradient_steps):
                loss = nn_model.train_with_memory(memory)
                print(loss)
            
            # saving the weights the latest neural model
            nn_model.save_weights(join(self._models_folder, 'model_weights'))
    
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
    
    
    def solve_problems(self, planner, nn_model):
        if self._scheduler == 'gbs':
            self._solve_gbs(planner, nn_model)
        else:
            self._solve_uniform(planner, nn_model)