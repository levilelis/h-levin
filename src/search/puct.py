import numpy as np

from models.memory import Trajectory
import time

class PUCTTreeNode:
    def __init__(self, parent, game_state, action, action_probs, g_cost, cpuct=1.0):
        self.c = cpuct
        self._game_state = game_state
        self._action = action
        self._parent = parent
        self._g = g_cost

        self._actions = game_state.successors()
        self._N = {}
        self._W = {}
        self._Q = {}
        self._P = {}
        self._virtual_loss = {}
        self._children = {}
        self._num_children_expanded = 0

        self._is_fully_expanded = False

        self._N_total = 0

        for a in self._actions:
            self._N[a] = 0
            self._W[a] = 0
            self._Q[a] = None
            self._virtual_loss[a] = 0
            self._children[a] = None
            self._P[a] = action_probs[a]

    def update_action_value(self, action, value):
        self._N_total += 1
        self._N[action] += 1

#         if self._Q[action] is None or value < self._Q[action]:
#             self._Q[action] = value

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
            normalized_Q[a] = (q + self._virtual_loss[a] - min_q) / (self._virtual_loss[a] + max_q - min_q)

        uct_values = {}
        for a in self._actions:
            uct_values[a] = normalized_Q[a] - self.c * self._P[a] * (np.math.sqrt(self._N_total)/(1 + self._N[a]))

        return uct_values

    def add_virtual_loss(self, action, max_q):
        self._virtual_loss[action] += max_q

    def remove_virtual_loss(self, action):
        self._virtual_loss[action] = 0

    def argmin_uct_values(self, max_q, min_q):
        if not self._is_fully_expanded:
            action_to_return = None
            for i in range(len(self._actions)):
                if self._children[self._actions[i]] is None:
                    action_to_return = self._actions[i]

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

        if self._num_children_expanded == len(self._actions):
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

    def __init__(self, use_heuristic=True, use_learned_heuristic=False, k_expansions=32, cpuct=1.0):
        self._use_heuristic = use_heuristic
        self._use_learned_heuristic = use_learned_heuristic
        self._k = k_expansions
        self._cpuct = cpuct

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
#                 self._backpropagate([current_node], [max_q])
                return current_node, action

        return current_node, action

    def _evaluate(self, nodes, actions, children):

#         children = []
        children_image = []

        for i in range(len(nodes)):
#             child_state = nodes[i].get_game_state().copy()
#             child_state.apply_action(actions[i])

#             children.append(child_state)
#             children_image.append(child_state.get_image_representation())
            children_image.append(children[i].get_image_representation())


        predicted_h = np.zeros(len(children))
        if self._use_learned_heuristic:
            _, action_probs, predicted_h = self._nn_model.predict(np.array(children_image))
        else:
#             print(np.array(children_image).shape)
            _, action_probs = self._nn_model.predict(np.array(children_image))

        child_nodes = []
        child_values = []

        for i in range(len(children)):
            child_node = PUCTTreeNode(nodes[i], children[i], actions[i], action_probs[i], nodes[i].get_g() + 1, self._cpuct)
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

        root = PUCTTreeNode(None, state, -1, action_probs[0], 0, self._cpuct)
        
        closed_list = set()
        closed_list.add(state)

        while True:
            nodes = []
            actions = []
            children_states = []
            number_trials = 0
            
            while len(nodes) == 0:

                for _ in range(self._k):
                    leaf_node, action = self._expand(root)
    
                    if number_trials % 100 == 0 and number_trials != 0:
                        print(number_trials)
    
                    if action is None:
                        number_trials += 1
                        continue
                    
                    child_state = leaf_node.get_game_state().copy()
                    child_state.apply_action(action)
                    
                    if child_state.is_solution():
                        print('Solved puzzle: ', puzzle_name)
                        trajectory = self._store_trajectory_memory(leaf_node, expanded)
                        return True, trajectory, expanded, 0, puzzle_name
    
                    if child_state not in closed_list:
                        closed_list.add(child_state)
    
                        nodes.append(leaf_node)
                        actions.append(action)
                        children_states.append(child_state)
    
                        expanded += 1

            leaves, values = self._evaluate(nodes, actions, children_states)

            if expanded >= budget:
                return False, None, expanded, 0, puzzle_name

            self._backpropagate(leaves, values)


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
        puzzle_name = data[1]
        self._nn_model = data[2]
        budget = data[3]
        start_overall_time = data[4]
        time_limit = data[5]
        slack_time = data[6]

        expanded = 0

        start_time = time.time()

        if self._use_learned_heuristic:
            _, action_probs, _ = self._nn_model.predict(np.array([state.get_image_representation()]))
        else:
            _, action_probs = self._nn_model.predict(np.array([state.get_image_representation()]))

        root = PUCTTreeNode(None, state, -1, action_probs[0], 0, self._cpuct)
        
        closed_list = set()
        closed_list.add(state)

        while True:
            nodes = []
            actions = []
            children_states = []

            while len(nodes) == 0:

                for _ in range(self._k):
                    leaf_node, action = self._expand(root)

                    if action is None:
                        continue
                    
                    child_state = leaf_node.get_game_state().copy()
                    child_state.apply_action(action)
                    
                    if child_state.is_solution():
                        end_time = time.time()
                        return leaf_node.get_g(), expanded, 0, end_time - start_time, puzzle_name

                    if child_state not in closed_list:
                        closed_list.add(child_state)

                        nodes.append(leaf_node)
                        actions.append(action)
                        children_states.append(child_state)

                        expanded += 1

                        end_time = time.time()
                        if (budget > 0 and expanded > budget) or end_time - start_overall_time + slack_time > time_limit:
                                return -1, expanded, 0, end_time - start_time, puzzle_name

            leaves, values = self._evaluate(nodes, actions, children_states)

#             for leaf_node in leaves:
#                 if leaf_node.get_game_state().is_solution():
#                     end_time = time.time()
#                     return leaf_node.get_g(), expanded, 0, end_time - start_time, puzzle_name

            self._backpropagate(leaves, values)
