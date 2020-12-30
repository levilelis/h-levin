import heapq
import numpy as np

from models.memory import Trajectory
import copy
import time

class GBFSTreeNode:
    def __init__(self, parent, game_state, g, h, action):
        self._game_state = game_state
        self._h = h
        self._g = g
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
        return self._h < other._h    
    
    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self._game_state.__hash__()
    
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
    
    def get_g(self):
        """
        Returns the g cost of a node
        """
        return self._g
    
    def get_action(self):
        """
        Returns the action taken to reach node stored in the node
        """
        return self._action
    
    def set_h_cost(self, h):
        """
        Sets the f-value of a node
        """
        self._h = h
        

class GBFS():
    
    def __init__(self, use_heuristic=True, use_learned_heuristic=False, k_expansions=32):
        self._use_learned_heuristic = use_learned_heuristic
        self._use_heuristic = use_heuristic
        
        self._k = k_expansions
    
    def get_h_cost(self, child, predicted_h):
        if self._use_heuristic and self._use_learned_heuristic:
            return max(predicted_h, child.heuristic_value())
        elif self._use_learned_heuristic:
            return predicted_h
        else:
            return child.heuristic_value()
    
    def search(self, data):
        """
        Performs GBFS search. 
        
        Returns solution cost, number of nodes expanded, and generated
        """
        state = data[0] 
        puzzle_name = data[1]
        nn_model = data[2]
        budget = data[3]
        start_overall_time = data[4]
        time_limit = data[5]
        slack_time = data[6]
        
        _open = []
        _closed = set()
        
        start_time = time.time()
        
        if slack_time == 0:
            start_overall_time = time.time()
        
        expanded = 0
        generated = 0
        
        predicted_h = np.zeros(self._k)
        
        heapq.heappush(_open, GBFSTreeNode(None, state, 0, 0, -1))
        _closed.add(state)
        
        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated = []
        
        while len(_open) > 0:
            node = heapq.heappop(_open)
            
            expanded += 1               
            
            end_time = time.time()
            if (budget > 0 and expanded > budget) or end_time - start_overall_time + slack_time > time_limit:
                    return -1, expanded, generated, end_time - start_time, puzzle_name
            
            actions = node.get_game_state().successors_parent_pruning(node.get_action())             
                
            for a in actions:
                child = copy.deepcopy(node.get_game_state())
                child.apply_action(a)
                
                generated += 1
                
                if child.is_solution():
                    end_time = time.time() 
                    return node.get_g() + 1, expanded, generated, end_time - start_time, puzzle_name

                child_node = GBFSTreeNode(node, child, node.get_g() + 1, -1, a)
                
                children_to_be_evaluated.append(child_node)
                x_input_of_children_to_be_evaluated.append(child.get_image_representation())
                
            if len(children_to_be_evaluated) >= self._k or len(_open) == 0:
                if self._use_learned_heuristic:
                    predicted_h = nn_model.predict(np.array(x_input_of_children_to_be_evaluated))
                    
                for i in range(len(children_to_be_evaluated)):
            
                    h_cost = self.get_h_cost(children_to_be_evaluated[i].get_game_state(), 
                                            predicted_h[i])
                    children_to_be_evaluated[i].set_h_cost(h_cost)
                                    
                    if children_to_be_evaluated[i].get_game_state() not in _closed:
                        heapq.heappush(_open, children_to_be_evaluated[i])
                        _closed.add(children_to_be_evaluated[i].get_game_state())
                        
                children_to_be_evaluated.clear()
                x_input_of_children_to_be_evaluated.clear()
                
        print('Emptied Open list: ', puzzle_name)
        
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
     
    def search_for_learning(self, data):
        """
        Performs GBFS search bounded by a search budget.
        
        Returns Boolean indicating whether the solution was found,
        number of nodes expanded, and number of nodes generated
        """
        state = data[0]
        puzzle_name = data[1]
        budget = data[2]
        nn_model = data[3]
        
        _open = []
        _closed = set()
        
        expanded = 0
        generated = 0
        
        heapq.heappush(_open, GBFSTreeNode(None, state, 0, 0, -1))
        _closed.add(state)
        
        predicted_h = np.zeros(self._k)
        
        children_to_be_evaluated = []
        x_input_of_children_to_be_evaluated = []
        
        while len(_open) > 0:
            node = heapq.heappop(_open)
            
            expanded += 1
            
            if expanded >= budget:
                return False, None, expanded, generated, puzzle_name
            
            actions = node.get_game_state().successors_parent_pruning(node.get_action())             
                
            for a in actions:
                child = copy.deepcopy(node.get_game_state())
                child.apply_action(a)
                
                generated += 1
            
                child_node = GBFSTreeNode(node, child, node.get_g() + 1, -1, a)
                
                if child.is_solution(): 
                    trajectory = self._store_trajectory_memory(child_node, expanded)
                    return True, trajectory, expanded, generated, puzzle_name
                
                children_to_be_evaluated.append(child_node)
                x_input_of_children_to_be_evaluated.append(child.get_image_representation())
                
            if len(children_to_be_evaluated) >= self._k or len(_open) == 0:
                if self._use_learned_heuristic:
                    predicted_h = nn_model.predict(np.array(x_input_of_children_to_be_evaluated))
                    
                for i in range(len(children_to_be_evaluated)):
            
                    h_cost = self.get_h_cost(children_to_be_evaluated[i].get_game_state(), 
                                            predicted_h[i])
                    
                    children_to_be_evaluated[i].set_h_cost(h_cost)
                                    
                    if children_to_be_evaluated[i].get_game_state() not in _closed:
                        heapq.heappush(_open, children_to_be_evaluated[i])
                        _closed.add(children_to_be_evaluated[i].get_game_state())
                        
                children_to_be_evaluated.clear()
                x_input_of_children_to_be_evaluated.clear()
                
        print('Emptied Open list: ', puzzle_name)
            