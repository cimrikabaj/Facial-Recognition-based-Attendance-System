import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt


class PReLU(layers.Layer):
    def __init__(self, **kwargs):
        super(PReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(1,),
            initializer=tf.initializers.Constant(0.25),  # Initialize with 0.25 for example
            trainable=True,
            name='alpha'
        )
        super(PReLU, self).build(input_shape)

    def call(self, inputs):
        return tf.nn.relu(inputs) - self.alpha * tf.nn.relu(-inputs)