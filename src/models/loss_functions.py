import tensorflow as tf
import numpy as np

from abc import ABC

class LossFunction(ABC):
    
    def compute_loss(self, trajectory, model):
        pass
    
class LevinLoss(LossFunction):
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]           
        actions_one_hot = tf.one_hot(trajectory.get_actions(), model.get_number_actions())
        y_log, _, _ = model(np.array(images))


        path_log_loss = tf.reduce_sum(tf.math.multiply(y_log, actions_one_hot), axis=1)
        
        cumsum_loss = -tf.cumsum(path_log_loss)
        log_f_values = tf.math.log(tf.convert_to_tensor(trajectory.get_f_values()[::-1], dtype=tf.float32))
        cumsum_loss += log_f_values
        return tf.reduce_sum(cumsum_loss)
    
class CrossEntropyLoss(LossFunction):
    
    def __init__(self):
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    def compute_loss(self, trajectory, model):
        images = [s.get_image_representation() for s in trajectory.get_states()]
        actions_one_hot = tf.one_hot(trajectory.get_actions(), model.get_number_actions())
        _, _, logits = model(np.array(images))
        return self.cross_entropy_loss(actions_one_hot, logits)