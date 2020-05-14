import tensorflow as tf
import numpy as np

from abc import ABC

class LossFunction(ABC):
    
    def compute_loss(self, trajectory, model):
        pass
    
class LevinLoss(LossFunction):
    
    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]           
        actions_one_hot = tf.one_hot(trajectory.get_actions(), model.get_number_actions())
        _, _, logits = model(np.array(images))
        loss = self.cross_entropy_loss(actions_one_hot, logits)
        
        loss *= tf.stop_gradient(tf.convert_to_tensor(trajectory.get_expanded(), dtype=tf.float64))

        return loss
    
class CrossEntropyLoss(LossFunction):
    
    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]
        actions_one_hot = tf.one_hot(trajectory.get_actions(), model.get_number_actions())
        _, _, logits = model(np.array(images))
        return self.cross_entropy_loss(actions_one_hot, logits)