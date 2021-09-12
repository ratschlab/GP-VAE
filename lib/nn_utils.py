import tensorflow as tf
from nd_mlp_mixer import MLPMixer, DenseOnAxis
from nd_mlp_mixer.layers import ResidualLayers
from nd_mlp_mixer.mlp import MLPNormRes


from lib.motion_utils.helper_layers import MyUnSqueezeLayer

from lib.motion_utils.mixer_mlp import MixerLayer
from nd_mlp_mixer.nd_mixer import NdMixer, NdAutoencoder

''' NN utils '''


def make_mixer_nn(output_size, hidden_sizes, image_size=(330,8), reshape_size=(8, 2640),do_unsqueeze=True):
    layers = [MixerLayer(h,image_size=image_size)
              for h in hidden_sizes]

    layers.append(MyUnSqueezeLayer(do_unsqueeze=do_unsqueeze))
    layers.append(tf.keras.layers.Reshape(reshape_size, dtype=tf.float32))
    layers.append(tf.keras.layers.Dense(output_size, dtype=tf.float32))
    return tf.keras.Sequential(layers)


def make_mixer_nn2(repr_shape, num_mix_layers,out_shape=None, hidden_size=None):
    Net = lambda outsize, axis: MLPNormRes(outsize, axis, hidden_size)
    make_mixer = lambda: NdMixer(Net=Net, gate=True)
    repr_init = NdMixer(outshape=repr_shape,Net=DenseOnAxis, gate=False)
    mixed = ResidualLayers(num_mix_layers, make_mixer)
    model = tf.keras.Sequential([repr_init,mixed])
    if out_shape is not None:
        repr_final = NdMixer(out_shape, Net=DenseOnAxis, gate=False)
        model.add(repr_final)

    return model


def make_nn(output_size, hidden_sizes):
    """ Creates fully connected neural network
            :param output_size: output dimensionality
            :param hidden_sizes: tuple of hidden layer sizes.
                                 The tuple length sets the number of hidden layers.
    """
    layers = [tf.keras.layers.Dense(h, activation=tf.nn.leaky_relu, dtype=tf.float32)
              for h in hidden_sizes]
    layers.append(tf.keras.layers.Dense(output_size, dtype=tf.float32))
    return tf.keras.Sequential(layers)


def make_cnn(output_size, hidden_sizes, kernel_size=3):
    """ Construct neural network consisting of
          one 1d-convolutional layer that utilizes temporal dependences,
          fully connected network

        :param output_size: output dimensionality
        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param kernel_size: kernel size for convolutional layer
    """
    cnn_layer = [tf.keras.layers.Conv1D(hidden_sizes[0], kernel_size=kernel_size,
                                        padding="same", dtype=tf.float32)]
    layers = [tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
              for h in hidden_sizes[1:]]
    layers.append(tf.keras.layers.Dense(output_size, dtype=tf.float32))
    return tf.keras.Sequential(cnn_layer + layers)


def make_2d_cnn(output_size, hidden_sizes, kernel_size=3):
    """ Creates fully convolutional neural network.
        Used as CNN preprocessor for image data (HMNIST, SPRITES)

        :param output_size: output dimensionality
        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param kernel_size: kernel size for convolutional layers
    """
    layers = [tf.keras.layers.Conv2D(h, kernel_size=kernel_size, padding="same",
                                     activation=tf.nn.relu, dtype=tf.float32)
              for h in hidden_sizes + [output_size]]
    return tf.keras.Sequential(layers)

def make_3d_cnn(output_size, hidden_sizes, kernel_size=3):
    """
    3d conv tested for motion
        :param output_size: output dimensionality
        :param hidden_sizes: tuple of hidden layer sizes.
                             The tuple length sets the number of hidden layers.
        :param kernel_size: kernel size for convolutional layers
    """
    layers = [tf.keras.layers.Conv3D(h, kernel_size=kernel_size, padding="same",
                                     activation=tf.nn.relu, dtype=tf.float32)
              for h in hidden_sizes + [output_size]]
    return tf.keras.Sequential(layers)