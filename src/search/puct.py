import numpy as np

from models.memory import Trajectory
import time

### A custom implementation of PUCT
### Note that DeepMind's PUCT implementation currently does not feature 
### state duplication checks and merging as we do, hence this is already
### an improvement over the literature (AFAWCT)


class PUCTTreeNode:
    def __init__(self, parent, game_state, action, action_probs, g_cost, cpuct=1.0):
        self.c = cpuct
        self._game_state = game_state
        self._actions = game_state.successors()
        
        self._N = {}
        self._W = {}
        self._Q = {}
        self._P = {}
        self._children = {}
        self._num_children_expanded = 0

        self._is_fully_expanded = False
        self._N_total = 0
        
        for a in self._actions:
            self._N[a] = 0
            self._W[a] = 0
            self._Q[a] = None
            self._children[a] = None
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

        #print("Q items {}".format(self._Q.items()))
        for a, q in self._Q.items():
            if q is None:
                q = min_q  # optimistic
            normalized_Q[a] = (q - min_q) / (max_q - min_q)

        uct_values = {}
        actions = self._puct_state.get_actions()
        for a in actions:
            uct_values[a] = normalized_Q[a] - self.c * self._puct_state.get_probability_value(a) * (np.math.sqrt(self._puct_state.get_n_total())/(1 + self._puct_state.get_n(a)))

        return uct_values

    def argmin_uct_values(self, max_q, min_q):
        uct_values = self.get_uct_values(max_q, min_q)
        #print("uct values = {}".format(uct_values))

        min_value = 0
        min_action = None

        for action, value in uct_values.items():
            if (min_action is None) or (value < min_value):
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

    def _get_v_value(self, state, predicted_h):
        if self._use_heuristic and self._use_learned_heuristic:
            return max(state.heuristic_value(), predicted_h)
        elif self._use_learned_heuristic:
            return predicted_h
        else:
            return state.heuristic_value()

    def _descend(self, root):
        current_node = root

        if self._max_q is None or self._min_q is None:
            max_q = 1
            min_q = 0
        else:
            max_q = self._max_q
            min_q = self._min_q
            
        # From the root
        # choose an action
        # if action is none: return none
        # move to child

        action = current_node.argmin_uct_values(max_q, min_q)
        if action is None:
            self._backpropagate([current_node], [max_q])
            return None, None # just skip

        while not current_node.is_leaf(action):
            current_node = current_node.get_child(action)
            action = current_node.argmin_uct_values(max_q, min_q)

            if action is None:
                self._backpropagate([current_node], [max_q])
                return current_node, action
                
        return current_node, action

    def _evaluate(self, node, action, child_state):
        child_image = child_state.get_image_representation()

        predicted_hs = np.zeros(1)
        if self._use_learned_heuristic:
            #_, action_probs, predicted_h = self._nn_model.predict(np.array(children_image))
            _, action_probss, predicted_hs = self._nn_model.predict(np.array([child_image]))
        else:
            #_, action_probs = self._nn_model.predict(np.array(children_image))
            _, action_probss = self._nn_model.predict(np.array([child_image]))
            
        predicted_h = predicted_hs[0]
        action_probs = action_probss[0]

        child_node = PUCTTreeNode(node, child_state, action, action_probs, node.get_g() + 1, self._cpuct)
        node.add_child(child_node, action)
        child_value = self._get_v_value(child_state, predicted_h)

        if self._max_q is None or child_value > self._max_q:
            self._max_q = child_value

        if self._min_q is None or child_value < self._min_q:
            self._min_q = child_value

        return child_node, child_value

    def _backpropagate(self, leaf, value):
        node = leaf

        while not node.is_root():
            parent = node.get_parent()
            parent.update_action_value(node.get_action(), value)
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
            _, action_probs, _ = nn_model.predict(np.array([state.get_image_representation()]))
        else:
            _, action_probs = nn_model.predict(np.array([state.get_image_representation()]))

        root_state = PUCTState(state, action_probs[0])
        self._states_list[state] = root_state
        root = PUCTTreeNode(None, root_state, -1, 0, self._cpuct)

        closed_list = set()
        closed_list.add(state)

        def make_return_dict(status, traj):
            return {'status': status,
                    'trajectory': traj,
                    'solution_depth': -1 if traj is None else len(traj.get_actions()),
                    'expanded': expanded,
                    'generated': 0,  # not sure what value to use here. expanded?
                    'time': time.time() - start_time,
                    'puzzle_name': puzzle_name}

        # A dict of visited states -> canonical nodes
        # so as to share the state stats .
        # Still keep the children nodes when known to avoid a dict search.
        canonical_nodes = {}
        root_state = root.get_game_state()
        canonical_nodes[root_state] = root

        while True:
            # Tree descent. Restart from the root.
            current_node = root
            child_state = None
            child = root # just to get into the loop
            # Keep track of the visited states on the path to avoid loops.
            path_states = set()
            path = [] # sequence of (node, action) taken
            skip = False
            while child is not None:
                # Time's up?
                curr_time = time.time()
                # Time budget exceeded for all problems.
                if overall_time_budget > 0 and curr_time - overall_start_time + slack_time > overall_time_budget:
                    return make_return_dict('overall_time_budget', None)
                # Time budget exceeded for this problem.
                if time_budget > 0 and curr_time - start_time > time_budget:
                    return make_return_dict('time_budget', None)

                current_node = child
                current_state = current_node.get_game_state()
                if current_state in path_states:
                    # This state has already been visited on this path.
                    # Stop the loop, give a loss to the path, and
                    # restart from the root.
                    node, action = path[-1]  # can't be empty
                    # Backpropagate. WARNING: CODE DUPLICATION WITH BELOW
                    value = self._max_q  # worse value
                    for node_action in path:
                        node, action = node_action
                        node.update_action_value(action, value)
                    skip = True
                    break
                path_states.add(current_state)
                action = current_node.argmin_uct_values(self._max_q, self._min_q)
                path.append( (current_node, action) )
                #print("action: {}".format(action))
                child = current_node.get_child(action)
                # We found an non-expanded child.
                if child is None:
                    child_state = current_node.get_game_state().copy()
                    # Get the corresponding game state.
                    child_state.apply_action(action)
                    # Solution found?
                    if child_state.is_solution():
                        # WARNING: WRONG TRAJECTORY (uses canonical nodes' parents instead of the actual path)
                        #trajectory = self._store_trajectory_memory(child, expanded)
                        trajectory = Trajectory([], [p[1] for p in path], [], expanded)  # CHEAT
                        return make_return_dict('solved', trajectory)
                    # Check whether we end up in a known state,
                    # in which case we just link the node to the canonical child
                    # and continue the loop.
                    # Otherwise, we have found a leaf node and we exit the loop.
                    child = canonical_nodes.get(child_state, None)
                    if child is not None:
                        #print("Found canonical node")
                        current_node.add_child(child, action)

            # We count also one expansion if we 'skip', because
            # otherwise we may loop forever.
            expanded += 1
            #print("expanded = {} path len = {}".format(expanded, len(path)))
            # Node expansion budget exceeded for this problem.
            if node_budget > 0 and expanded > node_budget:
                return make_return_dict('node_budget', None)

            if not skip:
                            
                # Create a new node.
                child, child_value = self._evaluate(current_node, action, child_state)
                # Add a canonical node.
                canonical_nodes[child_state] = child
                
                # Backpropagate the value to the taken path.
                # We must propagate on the trajectory, so we can't use the 
                # canonical nodes' parents; this is why we keep track of
                # the trajectory.
                for node_action in path:
                    node, action = node_action
                    node.update_action_value(action, child_value)

