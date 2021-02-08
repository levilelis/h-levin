import tensorflow as tf
import numpy as np

from abc import ABC
import math

class LossFunction(ABC):
    
    def compute_loss(self, trajectory, model):
        pass
    
    
class ImprovedLevinMSELoss(LossFunction):
    
    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]           
        actions_one_hot = tf.one_hot(trajectory.get_actions(), model.get_number_actions())
        _, _, logits_pi, logits_h  = model(np.array(images))
        
        weights = model.get_weights()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm(w, ord=2)
        
        loss = self.cross_entropy_loss(actions_one_hot, logits_pi)
        
        d = len(trajectory.get_actions()) + 1
        pi = trajectory.get_solution_pi()
        expanded = trajectory.get_non_normalized_expanded() + 1
        
        a = 0
        if pi < 1.0:
            a = (math.log((d+1)/expanded)) / math.log(pi)
        if a < 0:
            a = 0        
        
        loss *= tf.convert_to_tensor(expanded * a, dtype=tf.float64)
        
        solution_costs_tf = tf.expand_dims(tf.convert_to_tensor(trajectory.get_solution_costs(), dtype=tf.float64), 1)
        loss += self.mse(solution_costs_tf, logits_h) + model._reg_const * weights_l2_norm

        return loss
    
class ImprovedLevinLoss(LossFunction):
    
    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]           
        actions_one_hot = tf.one_hot(trajectory.get_actions(), model.get_number_actions())
        _, _, logits = model(np.array(images))
        
        weights = model.get_weights()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm(w, ord=2)
        
        loss = self.cross_entropy_loss(actions_one_hot, logits)
        
        d = len(trajectory.get_actions()) + 1
        pi = trajectory.get_solution_pi()
        expanded = trajectory.get_non_normalized_expanded() + 1 
        
        a = 0
        if pi < 1.0:
            a = (math.log((d+1)/expanded)) / math.log(pi)
        if a < 0:
            a = 0        
        
        loss *= tf.convert_to_tensor(expanded * a, dtype=tf.float64)
        
        loss += model._reg_const * weights_l2_norm

        return loss
    
class RegLevinMSELoss(LossFunction):
    
    def __init__(self):
        self.mse = tf.keras.losses.MeanSquaredError()
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]           
        actions_one_hot = tf.one_hot(trajectory.get_actions(), model.get_number_actions())
        _, probs_softmax, _, logits_h  = model(np.array(images))
        
        probs_used_on_path = tf.math.multiply(tf.cast(actions_one_hot, dtype=tf.float64), probs_softmax)
        probs_used_on_path = tf.math.reduce_sum(probs_used_on_path, axis=1)
        solution_prob = tf.math.reduce_prod(probs_used_on_path)   
        
        weights = model.get_weights()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm(w, ord=2)
        
        solution_costs = trajectory.get_solution_costs()
        solution_cost = solution_costs[len(solution_costs) - 1]
        
        loss = tf.math.divide(tf.convert_to_tensor(solution_cost, dtype=tf.float64), solution_prob)
                
        solution_costs_tf = tf.expand_dims(tf.convert_to_tensor(solution_costs, dtype=tf.float64), 1)
        loss += self.mse(solution_costs_tf, logits_h) + model._reg_const * weights_l2_norm

        return loss
    
class RegLevinLoss(LossFunction):
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]           
        actions_one_hot = tf.one_hot(trajectory.get_actions(), model.get_number_actions())
        _, probs_softmax, _ = model(np.array(images))
        
        probs_used_on_path = tf.math.multiply(tf.cast(actions_one_hot, dtype=tf.float64), probs_softmax)
        probs_used_on_path = tf.math.reduce_sum(probs_used_on_path, axis=1)
        solution_prob = tf.math.reduce_prod(probs_used_on_path)   
        
        weights = model.get_weights()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm(w, ord=2)
        
        solution_costs = trajectory.get_solution_costs()
        solution_cost = solution_costs[len(solution_costs) - 1]
        
        loss = tf.math.divide(tf.convert_to_tensor(solution_cost, dtype=tf.float64), solution_prob)
        loss += model._reg_const * weights_l2_norm

        return loss
    
class LevinLoss(LossFunction):
    
    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]           
        actions_one_hot = tf.one_hot(trajectory.get_actions(), model.get_number_actions())
        _, _, logits = model(np.array(images))
        
        weights = model.get_weights()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm(w, ord=2)
        
        loss = self.cross_entropy_loss(actions_one_hot, logits)
        
        loss *= tf.convert_to_tensor(trajectory.get_non_normalized_expanded(), dtype=tf.float64)
        loss += model._reg_const * weights_l2_norm

        return loss
    
class CrossEntropyLoss(LossFunction):
    
    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]
        actions_one_hot = tf.one_hot(trajectory.get_actions(), model.get_number_actions())
        _, _, logits = model(np.array(images))
        
        weights = model.get_weights()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm(w, ord=2)
        
        return self.cross_entropy_loss(actions_one_hot, logits) + model._reg_const * weights_l2_norm
    
class CrossEntropyMSELoss(LossFunction):
    
    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]
        actions_one_hot = tf.one_hot(trajectory.get_actions(), model.get_number_actions())
        _, _, logits_pi, logits_h  = model(np.array(images))
        
        weights = model.get_weights()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm(w, ord=2)
        
        loss = self.cross_entropy_loss(actions_one_hot, logits_pi) 
        
        solution_costs_tf = tf.expand_dims(tf.convert_to_tensor(trajectory.get_solution_costs(), dtype=tf.float64), 1)
        loss += self.mse(solution_costs_tf, logits_h) + model._reg_const * weights_l2_norm
        
        return loss
    
class LevinMSELoss(LossFunction):
    
    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]           
        actions_one_hot = tf.one_hot(trajectory.get_actions(), model.get_number_actions())
        _, _, logits_pi, logits_h  = model(np.array(images))
        
        weights = model.get_weights()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm(w, ord=2)
        
        loss = self.cross_entropy_loss(actions_one_hot, logits_pi)
        
        loss *= tf.convert_to_tensor(trajectory.get_non_normalized_expanded(), dtype=tf.float64)
        
        solution_costs_tf = tf.expand_dims(tf.convert_to_tensor(trajectory.get_solution_costs(), dtype=tf.float64), 1)
        loss += self.mse(solution_costs_tf, logits_h) + model._reg_const * weights_l2_norm

        return loss
    
class MSELoss(LossFunction):
    
    def __init__(self):
        self.mse = tf.keras.losses.MeanSquaredError()
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]           
        logits_h  = model(np.array(images))
        
        weights = model.get_weights()
        weights_l2_norm = 0
        for w in weights:
            weights_l2_norm += tf.norm(w, ord=2)
        
        solution_costs_tf = tf.expand_dims(tf.convert_to_tensor(trajectory.get_solution_costs(), dtype=tf.float64), 1)
        loss = self.mse(solution_costs_tf, logits_h) + model._reg_const * weights_l2_norm

        return loss