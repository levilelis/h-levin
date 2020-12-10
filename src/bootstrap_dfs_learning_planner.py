from conv_net import ConvNet
import copy
import numpy as np
import math

class BootstrapDFSLearningPlanner():
    
    def __init__(self, beta, dropout, batch, model_folder = '', model_name = ''):
        """
        Contructor of planner. 
        
        Parameters:
            beta: entropy regularizer, which is currently not being used
            dropout: dropout rate used in the neural network; dropout=1.0 means no dropout
            batch: size of training batch --- FD: how many puzzles are in the batch?
            model_folder and model_name: if empty, then a new model is created; model is loaded otherwise
        """
        self.conv_nn = ConvNet(beta, dropout, model_folder, model_name)
        
        self.train_positive_labels = []
        self.train_positive_images = []
        self._x = [] # FD: this is the training data (images)
        self._y = [] # FD: this is also the training data (image labels)
        self.budget = 1
        self.has_trained = False
        
        self.batch_size = batch
                
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
        FD: Hmmm? Sounds like supervised learning...
        """
        path_array = np.array(actions)
        y = np.zeros((path_array.shape[0], 4))
        y[np.arange(path_array.shape[0]), path_array] = 1
        return y
    
    def save_model(self, model_name, step):
        """
        Saves the weight of the current neural network
        """
        self.conv_nn.save_model(model_name, step)
                
    def learn(self):        
        """
        Runs the training step of the neural network with the training data stored in
        the variables self._x (images) and self._y (image labels).
        FD -- I am super confused about what happens from line 67 to line 73 (random_indices)
        """
        self.has_trained = True
        
        random_indices = np.random.permutation(len(self._y)) # FD: create some random permutation of integers 0 to len(self._y)
        # such that len(random_indices) == len(self._y)
        local_x = np.array(self._x)
        local_y = np.array(self._y)
        errors = []
        for i in range(len(self._y)):
            index_beg = i * self.batch_size # FD: why??
            index_end = (i+1) * self.batch_size + 1 # FD: why??
            if index_end > len(random_indices):
                index_end = len(random_indices)
            if index_beg < len(random_indices):
                x = local_x[random_indices[index_beg:index_end]]
                y = local_y[random_indices[index_beg:index_end]]
                error = self.conv_nn.batch_train_positive_examples(x, y) # This is where we get the error. Is the policy trained here?
                errors.append(error)
            else: # FD: once index_beg !< len(random_indices) means that len(random_indices) = len(random_indices) = index_end (line 70). Therefore, we are done training
                return np.mean(errors)
        # FD: when we call "conv_nn.batch_train_positive_examples(x, y)" we are allowing the conv_nn to train with
        # solution states and respective actions.
    
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
        Generates training data from solved instances of the puzzles. This is done by generating
        all possible reflection of the solved puzzles. The label of each image (which represents a state)
        if the action taken by the planner at that state while solving the problem.
        FD: What is meant by "reflection of the solved puzzles"? -- with swapped colors, and rotated images
        FD: the image represents a state of the puzzle, the label of the image is the action that the agent should take at that state.
        """
        rotated_states = []
        rotated_actions = []
        
        swap_colors = [False, True]
        flip_up_down = [False, True]
        number_of_rotations = [0, 1, 2, 3]
        for swap in swap_colors:
            for flip in flip_up_down:
                for r in number_of_rotations:
                    for i in range(0, len(self.train_positive_images)): # FD: iterate through all training images
                        r_state = copy.deepcopy(self.train_positive_images[i]) # FD: get one of the training images
                        a = self.train_positive_labels[i]  # FD: take its associated label
                        for _ in range(r):
                            r_state.rotate90() # FD: rotate the image
                            a = r_state.get_rotate90_action(a) # FD: get the action of the associated rotated image
                            # FD: but what do we do with these rotated images and their actions?
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
        are stored in the lists self.train_positive_images and self.train_positive_labels, respectively.
        FD: "states of solutions" are the states that are either the goal state of the puzzle, or that
         lead to the goal state. So, self.train_positive_images is a set of images that contain these
         "states of solutions". "self.train_positive_labels" are a set of the associated actions.

        Lists self.solution_states and self.solution_actions are set to empty for the next iteration of the system
        """
        for i in range(0, len(self.solution_states)):
            self.train_positive_images.append(copy.deepcopy(self.solution_states[i]))
            self.train_positive_labels.append(self.solution_actions[i])

        self.solution_states = []
        self.solution_actions = []
        # FD : at the end, we reset the "solution_states" and "solution_actions" lists to be empty.
    


    def _dfs_lvn_budget_for_learning(self, state, p, depth, level):
        """
        Recursive implementation of Depth-First LTS using a policy given by the 
        neural network self.conv_nn.
        FD: what is p?? Is p = pi?
        level = budget -- why do we pass the budget as the "level"?
        Why do we compare log(depth) - p to the budget?
        We make the new_bound = log(depth) - p. why?
        """
        v = math.log(depth) - p # FD: is this  = log(depth / pi) ?
        if v > level: # FD: if math.log(depth) - p > "current budget", then return false --> puzzle cannot be solved with given budget
            # FD: cap the new_bound (budget) to = math.log(depth) - p
            if self.new_bound == -1 or v < self.new_bound:
                self.new_bound = v
            return False
         
        actions = state.successors()
        action_distribution_log = self.conv_nn.classify_ylog(self._get_images(np.array([state])))[0]
         
        for a in actions:
            child = copy.deepcopy(state) # FD: is the child simply a copy of the current state?
            child.apply_action(a)
            if child.is_solution():
                self.solution_probability = math.exp(p)  # Adding the probability of finding this solution to difficulty ordering log
                self.solution_states.append(copy.deepcopy(state))
                self.solution_actions.append(a)
                self.solution_depth = depth + 1
                return True
                 
            if self._dfs_lvn_budget_for_learning(child, p + action_distribution_log[a], depth+1, level):
                self.solution_states.append(copy.deepcopy(state))
                self.solution_actions.append(a)
                return True
        return False
    
    def get_solution_depth(self):
        """
        Returns the solution length (cost) of the last problem solved by the planner
        """
        return self.solution_depth

    def get_probability_of_path(self):
        """
        Returns the probability of finding this solution from the last problem solved by the planner
        """
        return self.solution_probability
    
    def lvn_search_budget_for_learning(self, state, budget):
        """
        Performs Depth-First LTS limited by budget. 
        
        Returns boolean value indicating whether the problem was solved, and a budget value
        (new_bound) that will allow LTS to expand at least one state that wasn't expanded in the 
        previous Depth-First LTS iteration performed.
        # FD: What does it mean for a state to be expanded ?
        # FD: looks like new_bound is set to = log(depth) - p
        """
        self.solution_states = []
        self.solution_actions = []
        
        self.leaf_states = []
        self.leaf_actions = []
        
        self.new_bound = -1
        self.solution_depth = -1
        self.solution_probability = 0  # Probability of finding solution path
        
        has_found_solution = self._dfs_lvn_budget_for_learning(state, 0, 1, budget) # FD: state, p, depth, level
        # FD: my understanding is that we pass depth = 1, b/c we are starting at root = state.
        # FD: but why p=1?? Is it that p=log(pi)=0 -> pi = 1, b/c the probability of being at the root = 1 ??
        # FD : calls "_dfs_lvn_budget_for_learning", which is a recursive implementation of Depth-First LTS
        # using a policy given by the neural network "conv_nn".
        # FD: so it looks like we pass the current state, p=0, depth = 1 and the given budget to the LTS solver,
        # and the LTS solver tries to find the "solution states" (i.e., whether the puzzle is solvable)
        # starting at state = "state" and given budget = "budget"
        # FD: does this mean that this function (lvn_search_budget_for_learning) is only called once? yes

        if has_found_solution:
            self.collect_training_data_from_last_solved()
            # FD : this method copies all elements of self.solution_states and self.solution_actions to -
            # - self.train_positive_images and to self.train_positive_labels, respectively.
            # FD : Then this method resets self.solution_states and self.solution_actions to empty sets.
        
        if self.new_bound != -1: #FD: I think that the only case where new_bound would == -1 after calling
            # - the LTS solver is when the state you pass onto LTS is already a leaf node...?
            # - or if the LTS solver has called itself, but math.log(depth) - p <= budget at each iteration,
            # - and found the leaf node
            #self.new_bound = math.ceil(math.log2(self.new_bound))
            self.new_bound = math.ceil(self.new_bound)
        return has_found_solution, self.new_bound


# FD: what I don't understand is that, when the function "lvn_search_budget_for_learning" is executed, it makes
# self.new_bound = -1. Then, when it calls onto

# FD: look at: what does self.conv_nn.batch_train_positive_examples(x, y) and self.conv_nn.classify_ylog do?