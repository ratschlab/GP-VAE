import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer, LayerNormalization


class MLP(Layer):
    def __init__(self, hdim=512, out_dim=256):
        super().__init__()
        self.hdim = hdim
        self.out_dim = out_dim

    def call(self, x):
        x = Dense(self.hdim, activation="linear")(x)
        x = tf.nn.gelu(x)
        x = Dense(self.out_dim, activation="linear")(x)

        return x

class MixerLayer(Layer):
    def __init__(self, hdim=128, image_size=(330,8), n_channels=8):
        super().__init__()
        self.image_size = image_size
        self.inp = Input(shape=[n_channels, image_size[0], image_size[1]])
        self.MLP1 = MLP(hdim, out_dim=image_size[0])
        self.MLP2 = MLP(hdim, out_dim=image_size[1])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

    def call(self, x):
        y = self.norm1(x)
        y = tf.transpose(y, [0, 2, 1])
        out_1 = self.MLP1(y)
        in_2 = tf.transpose(out_1, [0, 2, 1]) + x

        y = self.norm2(in_2)
        out_2 = self.MLP2(y) + in_2

        return out_2


