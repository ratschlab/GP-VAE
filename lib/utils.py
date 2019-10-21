import tensorflow as tf
import numpy as np


''' TF utils '''


def reduce_logmeanexp(x, axis, eps=1e-5):
    """Numerically-stable (?) implementation of log-mean-exp.
    Args:
        x: The tensor to reduce. Should have numeric type.
        axis: The dimensions to reduce. If `None` (the default),
              reduces all dimensions. Must be in the range
              `[-rank(input_tensor), rank(input_tensor)]`.
        eps: Floating point scalar to avoid log-underflow.
    Returns:
        log_mean_exp: A `Tensor` representing `log(Avg{exp(x): x})`.
    """
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    return tf.log(tf.reduce_mean(
            tf.exp(x - x_max), axis=axis, keepdims=True) + eps) + x_max


def multiply_tfd_gaussians(gaussians):
    """Multiplies two tfd.MultivariateNormal distributions."""
    mus = [gauss.mean() for gauss in gaussians]
    Sigmas = [gauss.covariance() for gauss in gaussians]
    mu_3, Sigma_3, _ = multiply_gaussians(mus, Sigmas)
    return tfd.MultivariateNormalFullCovariance(loc=mu_3, covariance_matrix=Sigma_3)


def multiply_inv_gaussians(mus, lambdas):
    """Multiplies a series of Gaussians that is given as a list of mean vectors and a list of precision matrices.
    mus: list of mean with shape [n, d]
    lambdas: list of precision matrices with shape [n, d, d]
    Returns the mean vector, covariance matrix, and precision matrix of the product
    """
    assert len(mus) == len(lambdas)
    batch_size = int(mus[0].shape[0])
    d_z = int(lambdas[0].shape[-1])
    identity_matrix = tf.reshape(tf.tile(tf.eye(d_z), [batch_size,1]), [-1,d_z,d_z])
    lambda_new = tf.reduce_sum(lambdas, axis=0) + identity_matrix
    mus_summed = tf.reduce_sum([tf.einsum("bij, bj -> bi", lamb, mu)
                                for lamb, mu in zip(lambdas, mus)], axis=0)
    sigma_new = tf.linalg.inv(lambda_new)
    mu_new = tf.einsum("bij, bj -> bi", sigma_new, mus_summed)
    return mu_new, sigma_new, lambda_new


def multiply_inv_gaussians_batch(mus, lambdas):
    """Multiplies a series of Gaussians that is given as a list of mean vectors and a list of precision matrices.
    mus: list of mean with shape [..., d]
    lambdas: list of precision matrices with shape [..., d, d]
    Returns the mean vector, covariance matrix, and precision matrix of the product
    """
    assert len(mus) == len(lambdas)
    batch_size = mus[0].shape.as_list()[:-1]
    d_z = lambdas[0].shape.as_list()[-1]
    identity_matrix = tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(d_z), axis=0), axis=0), batch_size+[1,1])
    lambda_new = tf.reduce_sum(lambdas, axis=0) + identity_matrix
    mus_summed = tf.reduce_sum([tf.einsum("bcij, bcj -> bci", lamb, mu)
                                for lamb, mu in zip(lambdas, mus)], axis=0)
    sigma_new = tf.linalg.inv(lambda_new)
    mu_new = tf.einsum("bcij, bcj -> bci", sigma_new, mus_summed)
    return mu_new, sigma_new, lambda_new


def multiply_gaussians(mus, sigmas):
    """Multiplies a series of Gaussians that is given as a list of mean vectors and a list of covariance matrices.
    mus: list of mean with shape [n, d]
    sigmas: list of covariance matrices with shape [n, d, d]
    Returns the mean vector, covariance matrix, and precision matrix of the product
    """
    assert len(mus) == len(sigmas)
    batch_size = [int(n) for n in mus[0].shape[0]]
    d_z = int(sigmas[0].shape[-1])
    identity_matrix = tf.reshape(tf.tile(tf.eye(d_z), [batch_size,1]), batch_size+[d_z,d_z])
    sigma_new = identity_matrix
    mu_new = tf.zeros((batch_size, d_z))
    for mu, sigma in zip(mus, sigmas):
        sigma_inv = tf.linalg.inv(sigma_new + sigma)
        sigma_prod = tf.matmul(tf.matmul(sigma_new, sigma_inv), sigma)
        mu_prod = (tf.einsum("bij,bj->bi", tf.matmul(sigma, sigma_inv), mu_new)
                   + tf.einsum("bij,bj->bi", tf.matmul(sigma_new, sigma_inv), mu))
        sigma_new = sigma_prod
        mu_new = mu_prod
    lambda_new = tf.linalg.inv(sigma_new)
    return mu_new, sigma_new, lambda_new
