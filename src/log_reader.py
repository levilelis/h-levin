import argparse
import math
from search.bfs_levin import TreeNode
from search.bfs_levin import BFSLevin
from models.memory import Trajectory
from os import listdir
from os.path import isfile, join
import copy


class LogReader:
    def __init__(self, log_folder):
        self.log_folder = log_folder

    def read_logs_iterations_names(self):
        print("Getting bootstrap iteration for each puzzle...")
        files = [f for f in listdir(self.log_folder) if isfile(join(self.log_folder, f))]
        files.sort()

        seed_iterations_names = {}

        for f in files:
            if '_ordering' in f:
                iterations_names = {}
                seed = int(f.split('-')[6][0])
                bootstrap_iteration = 1
                with open(join(self.log_folder, f), "r") as stream:
                    names = []
                    for line in stream:
                        data = line.split('\n')
                        if len(line.split(', ')) == 1:
                            next_iteration = int(data[0])
                            if next_iteration != bootstrap_iteration:
                                iterations_names[bootstrap_iteration] = copy.deepcopy(names)
                                names = []
                            bootstrap_iteration = next_iteration
                        else:
                            puzzle = data[0].split(', ')[0]
                            names.append(puzzle)
                    seed_iterations_names[seed] = copy.deepcopy(iterations_names)

        return seed_iterations_names

    def read_logs_get_best_solution_path(self):
        print("Getting best solution path for each puzzle...")
        files = [f for f in listdir(self.log_folder) if isfile(join(self.log_folder, f))]
        files.sort()

        guide_file = files[0]  # Must be a file that has all the puzzles solved on it (cant be incomplete)
        print(guide_file)
        del files[0]

        puzzles_solutions = {}

        with open(join(self.log_folder, guide_file), "r") as stream:
            for line in stream:
                data = line.split('\n')
                if not len(line.split(', ')) == 1:
                    puzzle = data[0].split(', ')[0]
                    solution = data[0].split(', ')[3].split(' ')[:-1]
                    for f in files:
                        with open(join(self.log_folder, f), "r") as inner_stream:
                            for inner_line in inner_stream:
                                inner_data = inner_line.split('\n')
                                if not len(inner_line.split(', ')) == 1:
                                    inner_puzzle = inner_data[0].split(', ')[0]
                                    if puzzle == inner_puzzle:
                                        inner_solution = inner_data[0].split(', ')[3].split(' ')[:-1]
                                        if len(inner_solution) < len(solution):
                                            solution = inner_solution
                                        break

                    for i in range(len(solution)):
                        solution[i] = int(solution[i])
                    puzzles_solutions[puzzle] = solution

        return puzzles_solutions

    def compute_average_ordering(self, seed_iterations_names):
        print("Computing average ordering...")
        self._names_sum_iterations = {}
        puzzle_counter = {}  # Number of times each puzzle appeared, makes using incomplete ordering files possible

        for _, iterations_names in seed_iterations_names.items():
            puzzle_index = 1
            for _, names in iterations_names.items():
                for name in names:
                    if name not in self._names_sum_iterations.keys():
                        self._names_sum_iterations[name] = puzzle_index
                        puzzle_counter[name] = 1
                    else:
                        self._names_sum_iterations[name] += puzzle_index
                        puzzle_counter[name] += 1
                puzzle_index += 1

        for puzzle in self._names_sum_iterations.keys():
            self._names_sum_iterations[puzzle] = math.floor(self._names_sum_iterations[puzzle]/puzzle_counter[puzzle])

        iterations_names_avg = {}
        for name, sum_iterations in self._names_sum_iterations.items():
            if sum_iterations not in iterations_names_avg:
                iterations_names_avg[sum_iterations] = [name]
            else:
                iterations_names_avg[sum_iterations].append(name)

        return sorted(iterations_names_avg.items())

    def generate_average_ordering_log(self, avg_ordering, puzzles_solutions, log_name):
        print("Generating average log...")
        with open(join(self.log_folder + log_name + '_average_ordering'), 'a') as result_file:
            for i in range(len(avg_ordering)):
                result_file.write("{:d}".format(avg_ordering[i][0]))
                result_file.write('\n')
                for puzzle in avg_ordering[i][1]:
                    if puzzle in puzzles_solutions.keys():
                        result_file.write("{:s}, ".format(puzzle))
                        solution = puzzles_solutions[puzzle]
                        for action in solution:
                            result_file.write(("{:d} ".format(action)))
                        result_file.write('\n')

    def get_simple_ordering_from_file(self, ordering_file):
        with open(join(self.log_folder, ordering_file), "r") as stream:
            ordering = []
            for line in stream:
                data = line.split('\n')
                if not len(line.split(', ')) == 1:
                    puzzle = data[0].split(', ')[0]
                    ordering.append(puzzle)

        return ordering

    def generate_spread_curriculum(self, ordering_file, result_file, num_puzzles=9):
        if num_puzzles < 3:
            print("Number of puzzles for spread curriculum is too low.", num_puzzles)
        ordering = self.get_simple_ordering_from_file(ordering_file)
        size = len(ordering)
        middle = math.floor(size/2)
        self.spread_curriculum = [ordering[0], ordering[middle], ordering[size-1]]  # Spread starts with the first and last puzzles on ordering
        self.left_medians = []
        self.right_medians = []

        self.rec_ordering_median_left(ordering[:middle], self.left_medians)
        self.rec_ordering_median_right(ordering[middle:], self.right_medians)
        left_i = 0
        right_i = 0

        while len(self.spread_curriculum) < num_puzzles:
            if len(self.spread_curriculum) % 2 == 1:
                if self.left_medians[left_i] not in self.spread_curriculum:
                    self.spread_curriculum.append(self.left_medians[left_i])
                left_i += 1
            else:
                if self.right_medians[right_i] not in self.spread_curriculum:
                    self.spread_curriculum.append(self.right_medians[right_i])
                right_i += 1

        for i in range(len(ordering)):
            if ordering[i] in self.spread_curriculum:
                with open(result_file, 'a') as file:
                    file.write("{:s}".format(ordering[i]))
                    file.write("\n")

    def rec_ordering_median_left(self, ordering, medians_list):
        if len(ordering) == 1:
            medians_list.append(ordering[0])
            return medians_list

        middle = math.floor(len(ordering)/2)
        if ordering[middle] not in medians_list:
            medians_list.append(ordering[middle])
        self.rec_ordering_median_left(ordering[:middle], medians_list)
        self.rec_ordering_median_left(ordering[middle:], medians_list)

    def rec_ordering_median_right(self, ordering, medians_list):
        if len(ordering) == 1:
            medians_list.append(ordering[0])
            return medians_list

        middle = math.floor(len(ordering)/2)
        if ordering[middle] not in medians_list:
            medians_list.append(ordering[middle])
        self.rec_ordering_median_right(ordering[middle:], medians_list)
        self.rec_ordering_median_right(ordering[:middle], medians_list)


class TrajectoryGenerator:
    def __init__(self, ordering_file, states):
        self.ordering_file = ordering_file
        self.states = states

    def get_solved_blocks_and_solutions(self):
        solved_blocks = {}
        solutions = {}
        trajectories = {}
        with open(self.ordering_file, "r") as stream:
            for line in stream:
                data = line.split('\n')
                if len(line.split(', ')) == 1:
                    if trajectories:
                        solved_blocks[iteration] = trajectories
                    iteration = int(data[0])
                    trajectories = {}
                else:
                    puzzle = data[0].split(', ')[0]
                    solution = data[0].split(', ')[1].split(' ')[:-1]
                    for i in range(len(solution)):
                        solution[i] = int(solution[i])
                    trajectory = self.generate_trajectory(puzzle, solution)
                    trajectories[puzzle] = trajectory
                    solutions[puzzle] = solution

        return solved_blocks, solutions

    def generate_trajectory(self, puzzle, solution):
        bfs_planner = BFSLevin()
        state = self.states[puzzle]
        parent = None
        child = state
        p = 0
        depth = 1
        last_action = -1

        node = TreeNode(parent, child, p, depth, -1, last_action)

        for action in solution:
            child = copy.deepcopy(node.get_game_state())
            child.apply_action(action)

            parent = copy.deepcopy(node)
            depth = node.get_g() + 1
            last_action = action

            node = TreeNode(parent, child, p, depth, -1, last_action)

        trajectory = bfs_planner._store_trajectory_memory(node, 1)

        return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store', dest='log_folder',
                        help='Folder in which the logs are found (used for average ordering)')

    parser.add_argument('-o', action='store', dest='ordering_file',
                        help='Name of the resulting average ordering file (used for average ordering)')

    parser.add_argument('-po', action='store', dest='puzzles_ordering',
                        help='Name of the puzzles ordering file (used for spread generation)')

    parser.add_argument('-s', action='store', dest='spread_file',
                        help='Name of the resulting spread curriculum (used for spread generation)')

    parser.add_argument('-size', action='store', dest='size',
                        help='Desired size of spread curriculum (used for spread generation)')

    parser.add_argument('--spread', action='store_true', dest='spread_gen',
                        help='Sets the generate spread curriculum mode')
    
    parameters = parser.parse_args()

    if parameters.spread_gen:  # Spread generation
        log_reader = LogReader("")
        log_reader.generate_spread_curriculum(parameters.puzzles_ordering, parameters.spread_file, parameters.size)

    else:  # Average ordering generation
        log_reader = LogReader(parameters.log_folder)
        seed_iterations_names = log_reader.read_logs_iterations_names()
        puzzles_solutions = log_reader.read_logs_get_best_solution_path()
        avg_ordering = log_reader.compute_average_ordering(seed_iterations_names)
        log_reader.generate_average_ordering_log(avg_ordering, puzzles_solutions, parameters.ordering_file)


if __name__ == "__main__":
    main()
