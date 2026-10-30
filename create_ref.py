"""
RUN IN OLD ENVIRONMENT
Creates a reference file to later compare the old with the new model
"""
import os
import sys
 
import numpy as np
import tensorflow as tf

import lib.models as models  
import ref_utils as ult

tf.compat.v1.enable_eager_execution()


name = sys.argv[1]
 
assert name in ult.NAMES, f"dataset must be one of {ult.NAMES}"

N = 4 
cfg = ult.get_config(name)
 
# build model
model = ult.build_model(models, name)

# load fixed batch and create noise
x, m, x_full, m_mse = ult.load_batch(name, N)
eps = np.random.RandomState(27).randn(N, cfg["latent_dim"], cfg["time_length"]).astype(np.float32)
 
# old code outputs
qz = model.encode(tf.constant(x))
z_mean = qz.mean()
x_hat = model.decode(z_mean).mean()
mse = model.compute_mse(x, y=x_full, m_mask=m_mse, binary=(name == "hmnist")) 
loss, nll, kl, z = ult.elbo_parts(model, x, m, eps)
 
out = dict(
    x=x, m=m, x_full=x_full, m_mse=m_mse, eps=eps,
    z_mean=z_mean.numpy(), cov_tril=qz.scale.to_dense().numpy(), z=z.numpy(),
    x_hat=x_hat.numpy(), mse=mse.numpy(),
    prior_cov=model._get_prior().covariance().numpy(),
    loss=loss.numpy(), nll=nll.numpy(), kl=kl.numpy(),
)
for prefix, net in ult.nets(model):
    for i, w in enumerate(net.get_weights()):
        out[f"{prefix}_{i}"] = w
 
os.makedirs("reference", exist_ok=True)
path = f"reference/gpvae_{name}.npz"
np.savez_compressed(path, **out)
print(f"saved {path} ({os.path.getsize(path) / 1e6} MB)")
print(f"loss {out['loss']} | nll {out['nll']} | kl {out['kl']}")
