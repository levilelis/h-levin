import heapq
import numpy as np

from models.memory import Trajectory
import copy
import math

class TreeNode:
    def __init__(self, parent, game_state, p, g, levin_cost, action):
        self._game_state = game_state
        self._p = p
        self._g = g
        self._levin_cost = levin_cost
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
        return self._levin_cost < other._levin_cost    
    
    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self._game_state.__hash__()
    
    def get_p(self):
        """
        Returns the pi cost of a node
        """
        return self._p
    
    def get_g(self):
        """
        Returns the pi cost of a node
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
        

class BFSLevin():
    
    def __init__(self, use_heuristic=True, use_learned_heuristic=False):
        self._use_heuristic = use_heuristic
        self._use_learned_heuristic = use_learned_heuristic
        
        self._k = 32
    
    def get_levin_cost(self, parent, child, p_action, predicted_h):
        if self._use_learned_heuristic:
            return (predicted_h + parent.get_g()) - (parent.get_p() + p_action)
        if self._use_heuristic:
            return (child.heuristic_value() + parent.get_g() + 1) - (parent.get_p() + p_action)
        return (parent.get_g() + 1) - (parent.get_p() + p_action)
    
    def search(self, data):
        """
        Performs Best-First LTS . 
        
        Returns solution cost. 
        """
        state = data[0] 
        nn_model = data[1]
        
        _open = []
        _closed = set()
        
        expanded = 0
        generated = 0
        
        heapq.heappush(_open, TreeNode(None, state, 1, 0, 0, None))
        _closed.add(state)
        
        predicted_h = np.zeros(self._k)
        
        while len(_open) > 0:
            
            nodes_to_be_expanded = []
            x_input_of_states_to_be_expanded = []
            
            while len(nodes_to_be_expanded) < self._k and len(_open) > 0:
                node = heapq.heappop(_open)
                nodes_to_be_expanded.append(node)
                x_input_of_states_to_be_expanded.append(node.get_game_state().get_image_representation())
            
            expanded += 1
            
            actions = node.get_game_state().successors()
        
            if self._use_learned_heuristic:
                action_distribution_log, predicted_h = nn_model.predict(np.array(x_input_of_states_to_be_expanded))
            else:
                action_distribution_log = nn_model.predict(np.array(x_input_of_states_to_be_expanded))
            
            for i in range(len(nodes_to_be_expanded)):
                expanded += 1
            
                actions = nodes_to_be_expanded[i].get_game_state().successors()                
                
                for a in actions:
                    child = copy.deepcopy(nodes_to_be_expanded[i].get_game_state())
                    child.apply_action(a)
                    
                    generated += 1
                    
                    levin_cost = math.log(self.get_levin_cost(nodes_to_be_expanded[i], 
                                                              child, 
                                                              action_distribution_log[i][a], 
                                                              predicted_h[i]))                
                    child_node = TreeNode(nodes_to_be_expanded[i],
                                          child, 
                                          nodes_to_be_expanded[i].get_p() + action_distribution_log[i][a], 
                                          nodes_to_be_expanded[i].get_g() + 1,
                                          levin_cost,
                                          a)
                    
                    if child.is_solution(): 
                        return nodes_to_be_expanded[i].get_g() + 1, expanded, generated
                    
                    if child not in _closed:
                        heapq.heappush(_open, child_node)
                        _closed.add(child)  
        
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
        Performs Best-First LTS bounded by a search budget.
        
        Returns Boolean indicating whether the solution was found,
        number of nodes expanded, and number of nodes generated
        """
        expanded = 0
        generated = 0
        
        state = data[0]
        puzzle_name = data[1]
        budget = data[2]
        nn_model = data[3]
        
        _open = []
        _closed = set()
        
        heapq.heappush(_open, TreeNode(None, state, 1, 0, 0, None))
        _closed.add(state)
        
        predicted_h = np.zeros(self._k)
        
        while len(_open) > 0:
            
            nodes_to_be_expanded = []
            x_input_of_states_to_be_expanded = []
            
            while len(nodes_to_be_expanded) < self._k and len(_open) > 0:
                node = heapq.heappop(_open)
                nodes_to_be_expanded.append(node)
                x_input_of_states_to_be_expanded.append(node.get_game_state().get_image_representation())
            
            if expanded >= budget:
                return False, None, expanded, generated, puzzle_name
                
            if self._use_learned_heuristic:
                action_distribution_log, predicted_h = nn_model.predict(np.array(x_input_of_states_to_be_expanded))
            else:
                action_distribution_log = nn_model.predict(np.array(x_input_of_states_to_be_expanded))
            
            for i in range(len(nodes_to_be_expanded)):
                expanded += 1
                
                actions = nodes_to_be_expanded[i].get_game_state().successors()                
                
                for a in actions:
                    child = copy.deepcopy(nodes_to_be_expanded[i].get_game_state())
                    child.apply_action(a)
                    
                    generated += 1
                    
                    levin_cost = math.log(self.get_levin_cost(nodes_to_be_expanded[i], 
                                                              child, 
                                                              action_distribution_log[i][a], 
                                                              predicted_h[i]))                
                    child_node = TreeNode(nodes_to_be_expanded[i],
                                          child, 
                                          nodes_to_be_expanded[i].get_p() + action_distribution_log[i][a], 
                                          nodes_to_be_expanded[i].get_g() + 1,
                                          levin_cost,
                                          a)
                    
                    if child.is_solution(): 
                        trajectory = self._store_trajectory_memory(child_node, expanded)
                        return True, trajectory, expanded, generated, puzzle_name
                    
                    if child not in _closed:
                        heapq.heappush(_open, child_node)
                        _closed.add(child)  
            
            
            