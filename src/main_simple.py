import os
import time
import argparse
import sys
import random
from os import listdir
from search.a_star import AStar
from search.bfs_levin import BFSLevin
#from search.gbfs import GBFS
from search.puct import PUCT
from domains.simple import SimpleEnv
from models.simple import SimplePolicy

def main():
    planner = False
    planner_str = (sys.argv[1] if (len(sys.argv) > 1) else '').lower()
    if(planner_str == 'astar'):
      planner = AStar(use_heuristic=False, use_learned_heuristic=False, k_expansions=32)
    elif(planner_str == 'ucs'):
      planner = AStar(use_heuristic=False, use_learned_heuristic=False, k_expansions=32, weight=0)
    elif(planner_str == 'wastar'):
      planner = AStar(use_heuristic=False, use_learned_heuristic=False, k_expansions=32, weight=1.5)
    elif(planner_str == 'gbfs'):
      planner = AStar(use_heuristic=False, use_learned_heuristic=False, k_expansions=32, weight=-1)
    elif(planner_str == 'puct'):
      planner = PUCT(use_heuristic=False, use_learned_heuristic=False, k_expansions=32, cpuct=1.0)
    elif(planner_str == 'bflts'):
      planner = BFSLevin(use_heuristic=False, use_learned_heuristic=False, k_expansions=32, estimated_probability_to_go=False)
    else:
      print('Expected one argument in ["astar", "wastar", "ucs", "gbfs", "puct", "bflts"]. Exiting.')
      exit()

    branch = 2
    # solution = [1, 0, 0, 1]
    # solution = [1, 0, 0, 1, 0, 0, 1, 0, 1]
    # solution = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1]
    solution = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    #solution = [random.randint(0, 1) for i in range(14)]) # can't compare with a random seed
    print("solution [{}] = {}".format(len(solution), solution))
    state = SimpleEnv(branch=branch, solution_path=solution, printp=False)
    puzzle_name = "simple_1"
    model = SimplePolicy(branch=branch)
    budget = 100000 #1000
    start_time = time.time()
    time_limit_seconds = 100

    print("Searching...")
    data = {'state': state,
            'puzzle_name': puzzle_name,
            'nn_model': model,
            'node_budget': budget,
            'time_budget': time_limit_seconds}
    res = planner.search(data)
    
    print("status: {}, solution_cost: {}, expanded: {}, generated: {}, time_taken: {}"
        "\n".format(res['status'], res['solution_depth'], res['expanded'], res['generated'], res['time']))
    if res['status'] == 'solved':
        traj = res['trajectory']
        acts = traj.get_actions()
        print("actions [{}] = {}".format(len(acts), acts))
    return False

if __name__ == "__main__":
    main()
