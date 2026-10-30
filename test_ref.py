
"""
Run in the NEW ENVIRONMENT
Test Old vs New version
"""
import os
 
import numpy as np
import pytest
 
import ref_utils as ult
import lib.models as models
import tensorflow as tf

#if tf.__version__.startswith("1."):      
#    tf.compat.v1.enable_eager_execution()

#tf.v1.disable_v2_behavior()

REF_DIR = os.path.join(os.getcwd(), "reference")

TOL = {"physionet": dict(rtol=1e-4, atol=1e-5),
       "hmnist": dict(rtol=1e-3, atol=1e-4),
       "sprites": dict(rtol=1e-3, atol=1e-4)}

@pytest.fixture(scope="module", params=ult.NAMES)
def case(request):
    name = request.param
    path = os.path.join(REF_DIR, f"gpvae_{name}.npz")
    if not os.path.exists(path):
        pytest.skip(f"no reference file for {name}")
    ref = np.load(path)
    model = ult.build_model(models, name)
    for prefix, net in ult.nets(model):
        n = len([k for k in ref.files if k.startswith(prefix + "_")])
        old = [ref[f"{prefix}_{i}"] for i in range(n)]
        assert [w.shape for w in old] == [w.shape for w in net.get_weights()], \
            f"{prefix}: weight shapes differ from the old model"
        net.set_weights(old)
    return name, model, ref
 
 
def test_prior_covariance_matches(case):
    name, model, ref = case
    # kernel + cauchy jitter + tiling over latent dimensions
    np.testing.assert_allclose(model._get_prior().covariance().numpy(), ref["prior_cov"], **TOL[name])
 
 
def test_encoder_matches(case):
    name, model, ref = case
    qz = model.encode(ref["x"])  # includes the CNN preprocessor for the image datasets
    np.testing.assert_allclose(qz.mean().numpy(), ref["z_mean"], **TOL[name])
    # banded posterior: precision -> covariance factor (sparse_add / triangular_solve path;
    # sigmoid for physionet, softplus for the image datasets)
    np.testing.assert_allclose(qz.scale.to_dense().numpy(), ref["cov_tril"], **TOL[name])
 
 
def test_decoder_matches(case):
    name, model, ref = case
    x_hat = model.decode(ref["z_mean"]).mean().numpy()
    np.testing.assert_allclose(x_hat, ref["x_hat"], **TOL[name])
 
 
def test_loss_terms_match(case):
    name, model, ref = case
    loss, nll, kl, z = ult.elbo_parts(model, ref["x"], ref["m"], ref["eps"])
    np.testing.assert_allclose(z.numpy(), ref["z"], **TOL[name])
    rtol = TOL[name]["rtol"]
    np.testing.assert_allclose(nll.numpy(), ref["nll"], rtol=rtol)
    np.testing.assert_allclose(kl.numpy(), ref["kl"], rtol=rtol)
    np.testing.assert_allclose(loss.numpy(), ref["loss"], rtol=rtol)
 
 
def test_mse_matches(case):
    name, model, ref = case
    mse = model.compute_mse(ref["x"], y=ref["x_full"], m_mask=ref["m_mse"],
                            binary=(name == "hmnist")).numpy()
    np.testing.assert_allclose(mse, ref["mse"], rtol=TOL[name]["rtol"])

