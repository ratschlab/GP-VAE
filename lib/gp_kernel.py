import tensorflow as tf

''' 

GP kernel functions 

'''


def rbf_kernel(T, length_scale):
    xs = tf.range(T, dtype=tf.float32)
    xs_in = tf.expand_dims(xs, 0)
    xs_out = tf.expand_dims(xs, 1)
    distance_matrix = tf.math.squared_difference(xs_in, xs_out)
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = tf.math.exp(-distance_matrix_scaled)
    return kernel_matrix


def diffusion_kernel(T, length_scale):
    assert length_scale < 0.5, "length_scale has to be smaller than 0.5 for the "\
                               "kernel matrix to be diagonally dominant"
    sigmas = tf.ones(shape=[T, T]) * length_scale
    sigmas_tridiag = tf.linalg.band_part(sigmas, 1, 1)
    kernel_matrix = sigmas_tridiag + tf.eye(T)*(1. - length_scale)
    return kernel_matrix


def matern_kernel(T, length_scale):
    xs = tf.range(T, dtype=tf.float32)
    xs_in = tf.expand_dims(xs, 0)
    xs_out = tf.expand_dims(xs, 1)
    distance_matrix = tf.math.abs(xs_in - xs_out)
    distance_matrix_scaled = distance_matrix / tf.cast(tf.math.sqrt(length_scale), dtype=tf.float32)
    kernel_matrix = tf.math.exp(-distance_matrix_scaled)
    return kernel_matrix


def cauchy_kernel(T, sigma, length_scale):
    xs = tf.range(T, dtype=tf.float32)
    xs_in = tf.expand_dims(xs, 0)
    xs_out = tf.expand_dims(xs, 1)
    distance_matrix = tf.math.squared_difference(xs_in, xs_out)
    distance_matrix_scaled = distance_matrix / length_scale ** 2
    kernel_matrix = tf.math.divide(sigma, (distance_matrix_scaled + 1.))

    alpha = 0.001
    eye = tf.eye(num_rows=kernel_matrix.shape.as_list()[-1])
    return kernel_matrix + alpha * eye
