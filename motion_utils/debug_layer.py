import tensorflow as tf


class MyPrintShapeLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(MyPrintShapeLayer, self).__init__()
    # self.num_outputs = num_outputs

  def build(self, input_shape):
      pass
    # self.kernel = self.add_weight("kernel",
    #                               shape=[int(input_shape[-1]),
    #                                      self.num_outputs])

  def call(self, inputs):
      print("printing layer!!!!",inputs.shape)
      return inputs
    # return tf.matmul(inputs, self.kernel)
