import copy
import numpy as np
import math
from models.conv_net import ConvNet
from models.memory import Memory, Trajectory

class DFSLevin():
    
    def __init__(self, loss_name = 'LevinLoss', model_name=None):
        """
        Contructor of planner. 
        
        Parameters:
            beta: entropy regularizer, which is currently not being used
            dropout: dropout rate used in the neural network; dropout=1.0 means no dropout
            batch: size of training batch
            model_folder and model_name: if empty, then a new model is created; model is loaded otherwise
        """
#         self.conv_nn = ConvNet(beta, dropout, model_folder, model_name)
        if model_name == None:
            self.nn = ConvNet((2, 2), 32, 4, loss_name)
        else:
            self.nn = ConvNet((2, 2), 32, 4, loss_name)
            self.nn.load_weights(model_name).expect_partial()
        
        self.train_positive_labels = []
        self.train_positive_images = []
        self._x = []
        self._y = []
        self.budget = 1
        self.has_trained = False
        
        self._memory = Memory()
        
        self.batch_size = 6
                
    def _get_images(self, states):
        """
        Method creates a numpy array of images of the states in the list states
        """
        images = [s.get_image_representation() for s in states]
        return np.array(images)
    
    def _get_labels(self, actions):
        """
        Returns a numpy array representing a collection of one-hot vectors with the labels of images.
        The labels are given by a set of actions, which is provided as input. 
        """
        path_array = np.array(actions)
        y = np.zeros((path_array.shape[0], 4))
        y[np.arange(path_array.shape[0]), path_array] = 1
        return y
    
    def save_model(self, model_name):
        """
        Saves the weight of the current neural network
        """
        self.nn.save_weights(model_name)
                
    def learn(self):        
        """
        Runs the training step of the neural network with the training data stored in
        the variables self._x (images) and self._y (image labels).
        """
        self.has_trained = True
        
        return self.nn.train_with_memory(self._memory)
    
    def increase_budget(self):
        """
        Increment the search budget by one. The budget is represented
        by the equation e^b, where b is the budget.
        """
        self.budget += 1
    
    def reset_budget(self):
        """
        Set the budget to the minimum value of 1
        """
        self.budget = 1
        
    def current_budget(self):
        """
        Returns the current budget
        """
        return self.budget
        
    def size_training_set(self):
        """
        Returns the size of the current training set
        """
        return len(self._x)
    
    def preprocess_data(self):
        """
        Generates training data from solved instances of the puzzles.
        """
#         self._x = self._get_images(self.train_positive_images)
        self._x = self.train_positive_images
        self._y = self._get_labels(self.train_positive_labels)
        
    def preprocess_data_augmentation(self):
        """
        Generates training data from solved instances of the puzzles. This is done by generating
        all possible reflection of the solved puzzles. The label of each image (which represents a state)
        if the action taken by the planner at that state while solving the problem.  
        """
        rotated_states = []
        rotated_actions = []
        
        swap_colors = [False, True]
        flip_up_down = [False, True]
        number_of_rotations = [0, 1, 2, 3]
        for swap in swap_colors:
            for flip in flip_up_down:
                for r in number_of_rotations:
                    for i in range(0, len(self.train_positive_images)):
                        r_state = copy.deepcopy(self.train_positive_images[i])
                        a = self.train_positive_labels[i]
                        for _ in range(r):
                            r_state.rotate90()
                            a = r_state.get_rotate90_action(a)
                        if flip:                        
                            r_state.flip_up_down()
                            a = r_state.get_flip_up_down_action(a)
                        if swap:
                            r_state.swap_colors()
                        rotated_states.append(r_state)
                        rotated_actions.append(a)
                self._x = self._get_images(rotated_states)
                self._y = self._get_labels(rotated_actions)
            
    def collect_training_data_from_last_solved(self):
        """
        Copy the states and actions taken at the states of solutions. These states and actions
        are stored in the lists self.train_positive_images and  self.train_positive_labels, respectively.
        
        Lists self.solution_states and self.solution_actions are set to empty for the next iteration of the system
        """
        for i in range(0, len(self.solution_states)):
            self.train_positive_images.append(copy.deepcopy(self.solution_states[i]))
            self.train_positive_labels.append(self.solution_actions[i])

        trajectory = Trajectory(self.solution_states, self.solution_actions, self.solution_f_values, self._states_expanded)
        self._memory.add_trajectory(trajectory)

        self.solution_states = []
        self.solution_actions = []
        self.solution_f_values = []
    


    def _dfs_lvn_budget_for_learning(self, state, p, depth, level, learning):
        """
        Recursive implementation of Depth-First LTS using a policy given by the 
        neural network self.conv_nn. 
        """
        v = math.log(depth + state.heuristic_value()) - p
        if v > level:
            if self.new_bound == -1 or v < self.new_bound:
                self.new_bound = v
            return False
        
        self._states_expanded += 1
        actions = state.successors()
        
        action_distribution_log, _, _ = self.nn(self._get_images(np.array([state])))
        action_distribution_log = action_distribution_log[0]
         
        for a in actions:
            child = copy.deepcopy(state)
            child.apply_action(a)
            self._states_generated += 1
             
            if child.is_solution(): 
                if learning:
                    self.solution_states.append(copy.deepcopy(state))
                    self.solution_actions.append(a)
                    self.solution_f_values.append(depth + child.heuristic_value())
                
                self.solution_depth = depth
                return True
                 
            if self._dfs_lvn_budget_for_learning(child, p + action_distribution_log[a], depth+1, level, learning):
                if learning:
                    self.solution_states.append(copy.deepcopy(state))
                    self.solution_actions.append(a)
                    self.solution_f_values.append(depth + child.heuristic_value())
                return True
        return False
    
    def get_solution_depth(self):
        """
        Returns the solution length (cost) of the last problem solved by the planner
        """
        return self.solution_depth
    
    def search(self, state):
        """
        Performs Depth-First LTS . 
        
        Returns solution cost. 
        """
        
        self.new_bound = -1
        self.solution_depth = -1
        self._states_expanded = 0
        self._states_generated = 0
        has_found_solution = False
        budget = 1
        
        while not has_found_solution:
            has_found_solution = self._dfs_lvn_budget_for_learning(state, 0, 1, budget, learning=False)        
            budget = max(budget + 1, math.ceil(self.new_bound))
    
        return self.solution_depth, self._states_expanded, self._states_generated
    
    def search_for_learning(self, state, budget):
        """
        Performs Depth-First LTS limited by budget. 
        
        Returns boolean value indicating whether the problem was solved, and a budget value
        (new_bound) that will allow LTS to expand at least one state that wasn't expanded in the 
        previous Depth-First LTS iteration performed. 
        """
        self.solution_states = []
        self.solution_actions = []
        self.solution_f_values = []
        
        self.leaf_states = []
        self.leaf_actions = []
        
        self.new_bound = -1
        self.solution_depth = -1
        self._states_expanded = 0
        self._states_generated = 0
        
        has_found_solution = self._dfs_lvn_budget_for_learning(state, 0, 1, budget, learning=True)
        
        if has_found_solution:
            self.collect_training_data_from_last_solved()
        
        if self.new_bound != -1:
            #self.new_bound = math.ceil(math.log2(self.new_bound))
            self.new_bound = math.ceil(self.new_bound)
        return has_found_solution, self.new_bound, self._states_expanded, self._states_generated
        