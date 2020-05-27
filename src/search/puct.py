import numpy as np

from models.memory import Trajectory
import copy

class PUCTTreeNode:
    def __init__(self, parent, game_state, action, action_probs, g_cost):
        self.c = 1
        self._game_state = game_state
        self._action = action
        self._parent = parent
        self._g = g_cost
        
        self._actions = game_state.successors()
        self._N = {}
        self._W = {}
        self._Q = {}
        self._P = {}
        self._children = {}
        
        self._N_total = 0
        
        for a in self._actions:
            self._N[a] = 0
            self._W[a] = 0
            self._Q[a] = 0
            self._children[a] = None
            self._P[a] = action_probs[a]
            
    def update_action_value(self, action, value):
        self._N_total += 1
        self._N[action] += 1
        self._W[action] += value
        self._Q[action] = self._W[action] / self._N[action]  
    
    def is_root(self):
        return self._parent is None
    
    def get_uct_values(self, max_q, min_q):
        normalized_Q = {}
        
        if max_q == min_q:
            max_q = 1
            min_q = 0
        
        for a, q in self._Q.items():
            normalized_Q[a] = (q - min_q) / (max_q - min_q)
                                        
        uct_values = {}
        for a in self._actions:
            uct_values[a] = normalized_Q[a] - self.c * self._P[a] * (np.math.sqrt(self._N_total)/(1 + self._N[a])) 
        
        return uct_values
    
    def argmin_uct_values(self, max_q, min_q):
        uct_values = self.get_uct_values(max_q, min_q)
        
        is_first = True
        min_value = 0
        min_action = None
        
        for action, value in uct_values.items():
            if is_first:
                min_action = action
                min_value = value
                is_first = False
            elif value < min_value:
                min_action = action
                min_value = value
        return min_action
    
    def get_child(self, action):
        return self._children[action]
    
    def is_leaf(self, action):
        return self._N[action] == 0
    
    def add_child(self, child, action):
        self._children[action] = child
        
    def get_g(self):
        """
        Returns the pi cost of a node
        """
        return self._g
    
    def __eq__(self, other):
        """
        Verify if two tree nodes are identical by verifying the 
        game state in the nodes. 
        """
        return self._game_state == other._game_state
    
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
    
    def get_actions(self):
        """
        Returns the actions available at the state represented by the node
        """
        return self._actions
        

class PUCT():
    
    def __init__(self, use_heuristic=True, use_learned_heuristic=False):
        self._use_heuristic = use_heuristic
        self._use_learned_heuristic = use_learned_heuristic
        
        self._max_q = None
        self._min_q = None
        
    def _get_v_value(self, child, predicted_h):
        if self._use_heuristic and self._use_learned_heuristic:
            return max(child.heuristic_value(), predicted_h)
        elif self._use_learned_heuristic:
            return predicted_h
        else:
            return child.heuristic_value()
        
    def _expand(self, root):
        current_node = root
        
        if self._max_q is None or self._min_q is None:
            max_q = 1
            min_q = 0
        else:
            max_q = self._max_q
            min_q = self._min_q        
        
        action = current_node.argmin_uct_values(max_q, min_q)
        
        while not current_node.is_leaf(action):
            current_node = current_node.get_child(action)
            action = current_node.argmin_uct_values(max_q, min_q)
            
            if action is None:
                return current_node, max_q
        
        child_state = copy.deepcopy(current_node.get_game_state())
        child_state.apply_action(action)
        
        predicted_h = 0
        if self._use_learned_heuristic:
            _, action_probs, predicted_h = self._nn_model.predict(np.array([child_state.get_image_representation()]))
        else:
            _, action_probs = self._nn_model.predict(np.array([child_state.get_image_representation()]))
        
        child_node = PUCTTreeNode(current_node, child_state, action, action_probs[0], current_node.get_g() + 1)
        current_node.add_child(child_node, action)
        v = self._get_v_value(child_state, predicted_h[0][0])
        
        if self._max_q is None or v > self._max_q:
            self._max_q = v
            
        if self._min_q is None or v < self._min_q:
            self._min_q = v
        
        return child_node, v
        
    def _backpropagate(self, leaf_node, v):
        node = leaf_node
        
        while not node.is_root():
            parent = node.get_parent()
            parent.update_action_value(node.get_action(), v)
            
            node = parent    
    
    def search_for_learning(self, data):
        """
        Performs PUCT search bounded by a search budget.
        
        Returns Boolean indicating whether the solution was found,
        number of nodes expanded, and number of nodes generated
        """
        state = data[0]
        puzzle_name = data[1]
        budget = data[2]
        self._nn_model = data[3]
        
        expanded = 0
        
        if self._use_learned_heuristic:
            _, action_probs, _ = self._nn_model.predict(np.array([state.get_image_representation()]))
        else:
            _, action_probs = self._nn_model.predict(np.array([state.get_image_representation()]))
        
        root = PUCTTreeNode(None, state, -1, action_probs[0], 0)
        
        while True:
            leaf_node, v = self._expand(root)
            expanded += 1
            
            if expanded >= budget:
                return False, None, expanded, 0, puzzle_name
            
            if leaf_node.get_game_state().is_solution():
                trajectory = self._store_trajectory_memory(leaf_node, expanded)
                return True, trajectory, expanded, 0, puzzle_name
            
            self._backpropagate(leaf_node, v)
        
   
        
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
     
    def search(self, data):
        """
        Performs PUCT search. 
        
        Returns solution cost, number of nodes expanded, and generated
        """
        state = data[0] 
        self._nn_model = data[1]
        
        expanded = 0
        
        if self._use_learned_heuristic:
            _, action_probs, _ = self._nn_model.predict(np.array([state.get_image_representation()]))
        else:
            _, action_probs = self._nn_model.predict(np.array([state.get_image_representation()]))
        
        root = PUCTTreeNode(None, state, -1, action_probs[0], 0)
                
        while True:
            leaf_node, v = self._expand(root)
            expanded += 1
            
            if leaf_node.get_game_state().is_solution():
                return leaf_node.get_g(), expanded, 0
            
            self._backpropagate(leaf_node, v)
            
            