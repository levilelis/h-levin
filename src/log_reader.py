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
                        help='Folder in which the logs are found')

    parser.add_argument('-o', action='store', dest='ordering_file',
                        help='Name of the resulting average ordering file')

    parameters = parser.parse_args()

    log_reader = LogReader(parameters.log_folder)
    seed_iterations_names = log_reader.read_logs_iterations_names()
    puzzles_solutions = log_reader.read_logs_get_best_solution_path()
    avg_ordering = log_reader.compute_average_ordering(seed_iterations_names)
    log_reader.generate_average_ordering_log(avg_ordering, puzzles_solutions, parameters.ordering_file)


if __name__ == "__main__":
    main()
