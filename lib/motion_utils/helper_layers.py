import tensorflow as tf


class MyUnSqueezeLayer(tf.keras.layers.Layer):
    def __init__(self, do_unsqueeze=True):
        super(MyUnSqueezeLayer, self).__init__()
        self.do_unsqueeze = do_unsqueeze

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return tf.expand_dims(inputs, 0) if self.do_unsqueeze else inputs

