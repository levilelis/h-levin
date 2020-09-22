import heapq
import numpy as np

from models.memory import Trajectory
import copy
import time

class AStarTreeNode:
    def __init__(self, parent, game_state, g, f, action):
        self._game_state = game_state
        self._g = g
        self._f = f
        self._action = action
        self._parent = parent
    
    def __eq__(self, other):
        """
        Verify if two tree nodes are identical by verifying the 
        game state in the nodes. 
        """
        return self._game_state == other._game_state
    
    def __lt__(self, other):
        """
        Function less-than used by the heap
        """
        return self._f < other._f    
    
    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self._game_state.__hash__()
    
    def get_g(self):
        """
        Returns the g-cost of a node
        """
        return self._g
    
    def get_game_state(self):
        """
        Returns the game state represented by the node
        """
        return self._game_state
    
    def get_parent(self):
        """
        Returns the parent of the node
        """
        return self._parent
    
    def get_action(self):
        """
        Returns the action taken to reach node stored in the node
        """
        return self._action
    
    def set_f_cost(self, f_cost):
        """
        Sets the f-value of a node
        """
        self._f = f_cost
        

class AStar():
    
    def __init__(self, use_heuristic=True, use_learned_heuristic=False, k_expansions=32, weight=1.0):
        """
        if weight=1 -> A*
        if weight=0 -> UCS
        if weight=-1 -> GBFS (kind of)
        otherwise weighted-A*
        """
        self._use_heuristic = use_heuristic
        self._use_learned_heuristic = use_learned_heuristic
        self._w = weight
        self._k = k_expansions
        if weight == -1:
            self._w = None # make sure to catch bugs
            self._m = 0
        elif weight != 0:
            self._m = 1. / weight
        else:
            self._m = None
        
        print('Weight used: {} inverse weight: {}'.format(self._w, self._m))
    
    def get_f_cost(self, state, g, predicted_h):       
        if self._use_learned_heuristic and self._use_heuristic:
            # return self._w * max(predicted_h, child.heuristic_value()) + g 
            return self._m * g + max(predicted_h, state.heuristic_value())
        if self._use_learned_heuristic:
            # return self._w * predicted_h + g
            return self._m * g + predicted_h
        elif self._use_heuristic:
            # return self._w * child.heuristic_value() + g
            return self._m * g + state.heuristic_value()
        return g
    
    # TODO: This is almost exactly shared with bfs_levin and puct -> merge
    def _store_trajectory_memory(self, tree_node, expanded):
        """
        Receives a tree node representing a solution to the problem. 
        Backtracks the path performed by search, collecting state-action pairs along the way. 
        The state-action pairs are stored alongside the number of nodes expanded in an object of type Trajectory,
        which is added to the variable memory. 
        """
        states = []
        actions = []
        solution_costs = []
        
        state = tree_node.get_parent()
        action = tree_node.get_action()
        cost = 1
        
        while not state.get_parent() is None:
            states.append(state.get_game_state())
            actions.append(action)
            solution_costs.append(cost)
            
            action = state.get_action()
            state = state.get_parent()
            cost += 1
            
        states.append(state.get_game_state())
        actions.append(action)
        solution_costs.append(cost)
        
        return Trajectory(states, actions, solution_costs, expanded)        
     
    # TODO: A lot of code shared with bfs_levin. Merge somehow?
    def search(self, data):
        """
        Performs A*-like search.
        
        Returns solution cost, number of nodes expanded, and generated
        """
        state = data['state']
        puzzle_name = data['puzzle_name']
        node_budget = data.get('node_budget', -1)
        time_budget = data.get('time_budget', -1)
        overall_start_time = data.get('overall_start_time', -1)
        overall_time_budget = data.get('overall_start_time', -1)
        nn_model = data['nn_model']

        start_time = time.time()

                
        _open = []
        _closed = set()
        
        expanded = 0
        generated = 0
        
        predicted_h = np.zeros(self._k)
        
        heapq.heappush(_open, AStarTreeNode(None, state, 0, 0, -1))
        _closed.add(state)
        
        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated = []
        
        def make_return_dict(status, traj, g):
            return {'status': status,
                    'trajectory': traj,
                    'solution_depth': -1 if traj is None else len(traj.get_actions()),
                    'g': g, # g-cost
                    'expanded': expanded,
                    'generated': generated,
                    'time': time.time() - start_time,
                    'puzzle_name': puzzle_name}

        while len(_open) > 0:
            node = heapq.heappop(_open)
            
            expanded += 1

            curr_time = time.time()
            
            # TODO: May be better to use constants 'solved', 'node_budget', etc.
            # to avoid typo bugs. (where should these constants be defined?)
            #
            # Node expansion budget exceeded for this problem
            if node_budget > 0 and expanded > node_budget:
                return make_return_dict('node_budget', None, -1)
            # Time budget exceeded for all problems
            if overall_time_budget > 0 and curr_time - start_overall_time + slack_time > overall_time_budget:
                return make_return_dict('overall_time_budget', None, -1)
            # Time budget exceeded for this problem
            if time_budget > 0 and curr_time - start_time > time_budget:
                return make_return_dict('time_budget', None, -1)
                            
            actions = node.get_game_state().successors_parent_pruning(node.get_action())             
                
            for a in actions:
                child = copy.deepcopy(node.get_game_state())
                child.apply_action(a)
                child_g = node.get_g() + 1
                
                generated += 1
                
                if child.is_solution():
                    trajectory = self._store_trajectory_memory(child_node, expanded)
                    return make_return_dict('solved', trajectory, child_g)

                child_node = AStarTreeNode(node, child, child_g, -1, a)
                
                children_to_be_evaluated.append(child_node)
                x_input_of_children_to_be_evaluated.append(child.get_image_representation())
                
            if len(children_to_be_evaluated) >= self._k or len(_open) == 0:
                if self._use_learned_heuristic:
                    predicted_h = nn_model.predict(np.array(x_input_of_children_to_be_evaluated))
                    
                for i in range(len(children_to_be_evaluated)):
            
                    f_cost = self.get_f_cost(children_to_be_evaluated[i].get_game_state(), 
                                             children_to_be_evaluated[i].get_g(),
                                             predicted_h[i])
                    children_to_be_evaluated[i].set_f_cost(f_cost)
                                    
                    if children_to_be_evaluated[i].get_game_state() not in _closed:
                        heapq.heappush(_open, children_to_be_evaluated[i])
                        _closed.add(children_to_be_evaluated[i].get_game_state())
                        
                children_to_be_evaluated.clear()
                x_input_of_children_to_be_evaluated.clear()
        print('Emptied Open list: ', puzzle_name)
