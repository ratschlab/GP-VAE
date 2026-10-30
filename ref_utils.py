import numpy as np
import tensorflow as tf

"""
Define network configurations taken from the README and the cofigurations in train.py
"""

NAMES = ["physionet", "hmnist", "sprites"]

DEFAULTS = dict(kernel="cauchy", kernel_scales=1, M=1, K=1,
                cnn_sizes=[256], cnn_kernel_size=3, img_shape=None)


CONFIGS = {
    "physionet": dict(latent_dim=35, data_dim=35, time_length=48,
                      encoder_sizes=[128, 128], decoder_sizes=[256, 256],
                      window_size=24, sigma=1.005, length_scale=7.0, beta=0.2,
                      decoder="GaussianDecoder",
                      data_file="data/physionet/physionet.npz"),
    "hmnist": dict(latent_dim=256, data_dim=784, time_length=10,
                   encoder_sizes=[256, 256], decoder_sizes=[256, 256, 256],
                   window_size=3, sigma=1.0, length_scale=2.0, beta=0.8,
                   decoder="BernoulliDecoder", img_shape=(28, 28, 1),
                   data_file="data/hmnist/hmnist_mnar.npz"),
    "sprites": dict(latent_dim=256, data_dim=12288, time_length=8,
                    encoder_sizes=[32, 256, 256], decoder_sizes=[256, 256, 256],
                    window_size=3, sigma=1.0, length_scale=2.0, beta=0.1,
                    decoder="GaussianDecoder", img_shape=(64, 64, 3),
                    data_file="data/sprites/sprites.npz"),
}
 
 
def get_config(name):
    cfg = dict(DEFAULTS)
    cfg.update(CONFIGS[name])
    return cfg

def build_model(models, name):
    """
    Builds the regarding model as in train.py
    """
    cfg = get_config(name)

    preprocessor = None
    if cfg["img_shape"] is not None:
        preprocessor = models.ImagePreprocessor(cfg["img_shape"], list(cfg["cnn_sizes"]),
                                                cfg["cnn_kernel_size"])
        

    model = models.GP_VAE(
        latent_dim=cfg["latent_dim"], data_dim=cfg["data_dim"], time_length=cfg["time_length"],
        encoder_sizes=cfg["encoder_sizes"], encoder=models.BandedJointEncoder,
        decoder_sizes=cfg["decoder_sizes"], decoder=getattr(models, cfg["decoder"]),
        kernel=cfg["kernel"], sigma=cfg["sigma"], length_scale=cfg["length_scale"],
        kernel_scales=cfg["kernel_scales"], image_preprocessor=preprocessor,
        window_size=cfg["window_size"], beta=cfg["beta"], M=cfg["M"], K=cfg["K"],
        data_type=name)

    model.get_trainable_vars()
    return model

def nets(model):
    """
    TODO
    """
    out = [("enc", model.encoder.net), ("dec", model.decoder.net)]
    if model.preprocessor is not None:
        out.append(("pre", model.preprocessor.net))
    return out

def load_batch(name, n):
    """
    Creating the Data Batches as in train.py
    First n samples of the test datasets, no shuffeling
    Returns: x, m, x_full, m_mse  
    """
    data = np.load(get_config(name)["data_file"])

    x = data["x_train_miss"][:n]
    m = data["m_train_miss"][:n]
    x_full = data["x_train_full"][:n]

    if name == "physionet":
        m_mse = data["m_train_artificial"][:n]
    else: 
        m_mse = m

    return x.astype(np.float32), m, x_full.astype(np.float32), m_mse

def elbo_parts(model, x, m_mask, eps):
    """
    TODO
    Same computation as VAE._compute_loss for M = K = 1, but with the latent
    noise `eps` (shape [B, latent_dim, time_length]) passed in instead of being
    drawn inside `qz_x.sample()`.  Keep this in sync with `_compute_loss`.
 
    Returns (loss, nll, kl, z); mask convention as in the repo: 1 = missing.
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    m_mask = tf.cast(tf.convert_to_tensor(m_mask), tf.bool)
    eps = tf.convert_to_tensor(eps, dtype=tf.float32)
 
    pz = model._get_prior()
    qz_x = model.encode(x)
    # reparameterised sample: z = mean + L @ eps  (L = scale_tril of the banded posterior)
    z = qz_x.mean() + tf.linalg.matvec(qz_x.scale.to_dense(), eps)
    px_z = model.decode(z)
 
    nll = -px_z.log_prob(x)
    nll = tf.compat.v1.where(tf.math.is_finite(nll), nll, tf.zeros_like(nll))
    nll = tf.compat.v1.where(m_mask, tf.zeros_like(nll), nll)
    nll = tf.reduce_sum(nll, [1, 2])
 
    kl = model.kl_divergence(qz_x, pz)
    kl = tf.compat.v1.where(tf.math.is_finite(kl), kl, tf.zeros_like(kl))
    kl = tf.reduce_sum(kl, 1)
 
    elbo = -nll - model.beta * kl
    return -tf.reduce_mean(elbo), tf.reduce_mean(nll), tf.reduce_mean(kl), z

