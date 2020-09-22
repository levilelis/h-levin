import numpy as np

from models.memory import Trajectory
import time

class PUCTState:
    def __init__(self, game_state, action_probs):
        self._game_state = game_state
        self._actions = game_state.successors()
        
        self._N = {}
        self._W = {}
        self._Q = {}
        self._P = {}
        self._virtual_loss = {}        
        
        self._N_total = 0
        
        for a in self._actions:
            self._N[a] = 0
            self._W[a] = 0
            self._Q[a] = None
            self._virtual_loss[a] = 0
            self._P[a] = action_probs[a]
            
    def __eq__(self, other):
        """
        Verify if two tree nodes are identical by verifying the
        game state in the nodes.
        """
        return self._game_state == other._game_state
    
    def __hash__(self):
        """
        Hash function used in the closed list
        """
        return self._game_state.__hash__()
    
    def update_action_value(self, action, value):
        self._N_total += 1
        self._N[action] += 1

        self._W[action] += value
        self._Q[action] = self._W[action] / self._N[action]
        
    def remove_virtual_loss(self, action):
        self._virtual_loss[action] = 0
        
    def get_actions(self):
        return self._actions
    
    def get_virtual_loss(self, action):
        return self._virtual_loss[action]
    
    def get_probability_value(self, action):
        return self._P[action]
    
    def get_n_total(self):
        return self._N_total
    
    def get_n(self, action):
        return self._N[action]
    
    def add_virtual_loss(self, action, max_q):
        self._virtual_loss[action] += max_q
        
    def get_game_state(self):
        return self._game_state
    
    def get_q_values(self):
        return self._Q
    
        

class PUCTTreeNode:
    def __init__(self, parent, puct_state, action, g_cost, cpuct=1.0):
        self.c = cpuct
        self._puct_state = puct_state
        self._action = action
        self._parent = parent
        self._g = g_cost

        self._children = {}
        self._N = {}
        self._num_children_expanded = 0

        self._is_fully_expanded = False

        actions = self._puct_state.get_actions()
        for a in actions:
            self._children[a] = None
            self._N[a] = 0

    def update_action_value(self, action, value):
        self._N[action] += 1
        self._puct_state.update_action_value(action, value)
        
    def is_root(self):
        return self._parent is None

    def get_uct_values(self, max_q, min_q):
        normalized_Q = {}

        if max_q == min_q:
            max_q = 1
            min_q = 0

        for a, q in self._puct_state.get_q_values().items():
            normalized_Q[a] = (q + self._puct_state.get_virtual_loss(a) - min_q) / (max_q - min_q)

        uct_values = {}
        actions = self._puct_state.get_actions()
        for a in actions:
            uct_values[a] = normalized_Q[a] - self.c * self._puct_state.get_probability_value(a) * (np.math.sqrt(self._puct_state.get_n_total())/(1 + self._puct_state.get_n(a)))

        return uct_values

    def add_virtual_loss(self, action, max_q):
        self._puct_state.add_virtual_loss(action, max_q)

    def remove_virtual_loss(self, action):
        self._puct_state.remove_virtual_loss(action)

    def argmin_uct_values(self, max_q, min_q):
        if not self._is_fully_expanded:
            action_to_return = None
            actions = self._puct_state.get_actions()
            
            for i in range(len(actions)):
                if self._children[actions[i]] is None:
                    action_to_return = actions[i]
                    return action_to_return

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

        self._num_children_expanded += 1

        if self._num_children_expanded == len(self._puct_state.get_actions()):
            self._is_fully_expanded = True

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
        return self._puct_state.get_game_state()

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

    def __init__(self, use_heuristic=True, use_learned_heuristic=False, k_expansions=32, cpuct=1.0):
        self._use_heuristic = use_heuristic
        self._use_learned_heuristic = use_learned_heuristic
        self._k = k_expansions
        self._cpuct = cpuct
        
        self._states_list = {} 

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
            current_node.add_virtual_loss(action, max_q)

            current_node = current_node.get_child(action)
            action = current_node.argmin_uct_values(max_q, min_q)

            if action is None:
                self._backpropagate([current_node], [max_q])

                return current_node, action

        return current_node, action

    def _evaluate(self, nodes, actions):

        children = []
        children_image = []

        for i in range(len(nodes)):
            child_state = nodes[i].get_game_state().copy()
            child_state.apply_action(actions[i])
  
            children.append(child_state)
            children_image.append(child_state.get_image_representation())


        predicted_h = np.zeros(len(children))
        if self._use_learned_heuristic:
            _, action_probs, predicted_h = self._nn_model.predict(np.array(children_image))
        else:
            _, action_probs = self._nn_model.predict(np.array(children_image))

        child_nodes = []
        child_values = []

        for i in range(len(children)):
            child_state = self._states_list.get(children[i], None)
            if child_state is None:
                child_state = PUCTState(children[i], action_probs[i])
                self._states_list[children[i]] = child_state 
            
            child_node = PUCTTreeNode(nodes[i], child_state, actions[i], nodes[i].get_g() + 1, self._cpuct)
            nodes[i].add_child(child_node, actions[i])
            v = self._get_v_value(children[i], predicted_h[i])

            child_nodes.append(child_node)
            child_values.append(v)

            if self._max_q is None or v > self._max_q:
                self._max_q = v

            if self._min_q is None or v < self._min_q:
                self._min_q = v

        return child_nodes, child_values

    def _backpropagate(self, leaves, values):

        for i in range(len(leaves)):
            node = leaves[i]

            while not node.is_root():
                parent = node.get_parent()
                parent.update_action_value(node.get_action(), values[i])

                parent.remove_virtual_loss(node.get_action())

                node = parent

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
        state = data['state']
        puzzle_name = data['puzzle_name']
        node_budget = data.get('node_budget', -1)
        time_budget = data.get('time_budget', -1)
        overall_start_time = data.get('overall_start_time', -1)
        overall_time_budget = data.get('overall_start_time', -1)
        slack_time = data.get('slack_time', 0)
        nn_model = data['nn_model']

        self._nn_model = nn_model

        expanded = 0

        start_time = time.time()

        if self._use_learned_heuristic:
            _, action_probs, _ = self._nn_model.predict(np.array([state.get_image_representation()]))
        else:
            _, action_probs = self._nn_model.predict(np.array([state.get_image_representation()]))

        def make_return_dict(status, traj):
            return {'status': status,
                    'trajectory': traj,
                    'solution_depth': -1 if traj is None else len(traj.get_actions()),
                    'expanded': expanded,
                    'generated': 0,  # not sure what value to use here. expanded?
                    'time': time.time() - start_time,
                    'puzzle_name': puzzle_name}

        root_state = PUCTState(state, action_probs[0])
        self._states_list[state] = root_state
        root = PUCTTreeNode(None, root_state, -1, 0, self._cpuct)

        while True:
            nodes = []
            actions = []
            
            distinct_nodes = set()

            while len(nodes) == 0:

                for _ in range(self._k):
                    leaf_node, action = self._expand(root)

                    if action is None:
                        continue
                    
                    leaf_state = leaf_node.get_game_state()
                    if leaf_state in distinct_nodes:
                        continue
                    distinct_nodes.add(leaf_state)

                    nodes.append(leaf_node)
                    actions.append(action)

                    expanded += 1

                    # Time's up?
                    curr_time = time.time()
                    # Time budget exceeded for all problems.
                    if overall_time_budget > 0 and curr_time - overall_start_time + slack_time > overall_time_budget:
                        return make_return_dict('overall_time_budget', None)
                    # Time budget exceeded for this problem.
                    if time_budget > 0 and curr_time - start_time > time_budget:
                        return make_return_dict('time_budget', None)

            leaves, values = self._evaluate(nodes, actions)

            if node_budget > 0 and expanded > node_budget:
                return make_return_dict('node_budget', None)
            # This must be done after _evaluate so as to apply the last actions.
            for leaf_node in leaves:
                if leaf_node.get_game_state().is_solution():
                    trajectory = self._store_trajectory_memory(leaf_node, expanded)
                    return make_return_dict('solved', trajectory)

            self._backpropagate(leaves, values)
