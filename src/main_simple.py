import os
import time
import argparse
import sys
import random
from os import listdir
from search.bfs_levin import BFSLevin
from search.a_star import AStar
from search.gbfs import GBFS
from search.bfs_levin_mult import BFSLevinMult
from search.puct import PUCT
from domains.simple import SimpleEnv
from models.simple import SimplePolicy

def main():
    planner = False
    planner_str = (sys.argv[1] if (len(sys.argv) > 1) else '').lower()
    if(planner_str == 'astar'):
      planner = AStar(use_heuristic=False, use_learned_heuristic=False, k_expansions=32)
    elif(planner_str == 'puct'):
      planner = PUCT(use_heuristic=False, use_learned_heuristic=False, k_expansions=32, cpuct=1.0)
    elif(planner_str == 'lts'):
      planner = BFSLevin(use_heuristic=False, use_learned_heuristic=False, k_expansions=32, estimated_probability_to_go=False)
    else:
      print('Expected one argument in ["astar", "puct", "lts"]. Exiting.')
      exit()

    branch = 2
    state = SimpleEnv(branch=branch, solution_path=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], printp=False)
    # state = SimpleEnv(branch=branch, solution_path=[random.randint(0, 1) for i in range(14)]) # can't compare with a random seed
    puzzle_name = "Simple1"
    model = SimplePolicy(branch=branch)
    budget = 100000
    start_time = time.time()
    time_limit_seconds = 100
    slack_time = 0  # 600

    data = [False] * 7
    data[0] = state
    data[1] = puzzle_name
    data[2] = model
    data[3] = budget
    data[4] = start_time
    data[5] = time_limit_seconds
    data[6] = slack_time
    solution_cost, expanded, generated, time_taken, puzzle_name_again = planner.search(data)
    # time_taken = time.time() - start_time
    print("solution_cost: {}, expanded: {}, generated: {}, time_taken: {}\n".format(solution_cost, expanded, generated, time_taken))
    return False

if __name__ == "__main__":
    main()
