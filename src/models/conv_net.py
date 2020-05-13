import tensorflow as tf
import numpy as np
from models.loss_functions import LevinLoss, CrossEntropyLoss

class InvalidLossFunction(Exception):
    pass

class ConvNet(tf.keras.Model):
    
    def __init__(self, kernel_size, filters, number_actions, loss_name):
        tf.keras.backend.set_floatx('float64')
        
        super(ConvNet, self).__init__(name='')
                        
        self._number_actions = number_actions
        self._kernel_size = kernel_size
        self._filters = filters
        self._number_actions = number_actions
        self._loss_name = loss_name
        
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, name='conv1', activation='relu', dtype='float64')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1', dtype='float64')
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, name='conv2', activation='relu', dtype='float64')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2', dtype='float64')
        self.flatten = tf.keras.layers.Flatten(name='flatten1', dtype='float64')
        self.dense1 = tf.keras.layers.Dense(128, name='dense1', activation='relu', dtype='float64')
        self.dense2 = tf.keras.layers.Dense(number_actions, name='dense2', dtype='float64')
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        
        if loss_name == 'LevinLoss':
            self._loss_function = LevinLoss()
        elif loss_name == 'CrossEntropyLoss':
            self._loss_function = CrossEntropyLoss()
        else:
            raise InvalidLossFunction
        
    def predict(self, x):
#         with self.mutex:
        log_softmax, _, _ = self.call(x)
        return log_softmax[0]
        
    def call(self, input_tensor):
        
        x = self.conv1(input_tensor)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense2(x)
        x_softmax = tf.nn.softmax(logits)
#         x_log = tf.math.log(x_softmax)
        x_log_softmax = tf.nn.log_softmax(logits)
#         x_log2 = x_log / tf.math.log(tf.constant(2, dtype=x_log.dtype))
        
        return x_log_softmax, x_softmax, logits 
    
    def _cross_entropy_loss(self, states, y):
        images = [s.get_image_representation() for s in states]
        _, _, logits = self(np.array(images))
        return self.cross_entropy_loss(y, logits)
    
    def train_with_memory(self, memory):
        losses = []
        memory.shuffle_trajectories()
        for trajectory in memory.next_trajectory():
            
            with tf.GradientTape() as tape:
                loss = self._loss_function.compute_loss(trajectory, self)

            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            losses.append(loss)
        
        return np.mean(losses)
            
    def train(self, states, y):
        with tf.GradientTape() as tape:
            loss = self._cross_entropy_loss(states, y)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return loss
    
    def get_number_actions(self):
        return self._number_actions