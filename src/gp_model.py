# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Gaussian Process
=========================

In this example we show how to use NUTS to sample from the posterior
over the hyperparameters of a gaussian process.

.. image:: ../_static/img/examples/gp.png
    :align: center
"""

import argparse
import os
import time
import pickle as pkl

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)

from scipy.spatial.distance import pdist, cdist, squareform

matplotlib.use("Agg")  # noqa: E402


# squared exponential kernel with diagonal noise term
def kernel(X, Y, alpha, length_scales, noise, jitter=1.0e-6, include_noise=True):
    dmat = (X[:, None, :] - Y[None, :, :]) / length_scales[None, None, :]
    k = alpha * jnp.exp(-0.5 * jnp.sum(dmat * dmat, axis=2))
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k



def model(X, Y):
    num_features = X.shape[1]
    # set uninformative log-normal priors on our three kernel hyperparameters

    intercept = numpyro.sample("intercept", dist.LogNormal(0.0, 10.0))
    alpha = numpyro.sample("kernel_var", dist.Gamma(1.5, 0.1))
    noise = numpyro.sample("kernel_noise", dist.HalfNormal(scale=1.0))
    with numpyro.plate("feature_params", num_features):
        length_scales = numpyro.sample("length_scale", dist.HalfNormal(0.0, 1))
        beta = numpyro.sample("beta_coef", dist.Normal(0.0, 10.0))

    # compute kernel
    k = kernel(X, X, alpha, length_scales, noise)

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=intercept + jnp.matmul(X, beta), covariance_matrix=k),
        obs=Y,
    )


def model_with_confounding(X, Y, M):
    K = X.shape[1]
    # set uninformative log-normal priors on our three kernel hyperparameters

    intercept = numpyro.sample("intercept", dist.LogNormal(0.0, 10.0))
    alpha = numpyro.sample("kernel_var", dist.LogNormal(0.0, 10.0))
    sigma_y = numpyro.sample("sigma_y", dist.HalfNormal(scale=10.0))
    sigma_x = numpyro.sample("sigma_x", dist.HalfNormal(scale=10.0))
    with numpyro.plate("feature_params", K):
        length_scales = numpyro.sample("length_scale", dist.LogNormal(0.0, 1.0))
        beta = numpyro.sample("beta_coef", dist.Normal(0.0, 10.0))

    # compute kernel
    gp_kernel = kernel(X, X, alpha, length_scales, jnp.square(sigma_y))

    r2 = numpyro.sample("r2", dist.Uniform())

    B = numpyro.sample("B", dist.Normal(0.0, 1.0), sample_shape=(K, int(M)))

    gamma_unnormalized = numpyro.sample("gamma_raw", dist.Normal(0.0, 1.0), sample_shape=(int(M),))

    cov_x = jnp.matmul(B, jnp.transpose(B)) + jnp.diag(jnp.square(sigma_x) * jnp.ones(K))
    cov_x_inv = jnp.linalg.inv(cov_x)
    cov_u_t = jnp.diag(jnp.ones(B.shape[1])) - jnp.transpose(B) @ cov_x_inv @ B
    root_u_t_inv = jnp.linalg.inv(jnp.linalg.cholesky(cov_u_t))


    gamma_dir = gamma_unnormalized / (jnp.linalg.norm(gamma_unnormalized))

    gamma = sigma_y * jnp.sqrt(r2) * jnp.matmul(root_u_t_inv, gamma_dir)

    bias = cov_x_inv @ B @  gamma

    # sample X according to factor model
    with numpyro.plate("data", X.shape[0]):
        numpyro.sample(
            "X",
            dist.MultivariateNormal(loc=0.0, covariance_matrix=cov_x),
            obs=X
        )
    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=intercept + jnp.matmul(X, (beta + bias)) , covariance_matrix=gp_kernel),
        obs=Y,
    )




# helper function for doing hmc inference
def run_inference(model, args, rng_key, X, Y, M):

    start = time.time()
    # demonstrate how to use different HMC initialization strategies
    if args.init_strategy == "value":
        init_strategy = init_to_value(
            values={"kernel_var": 1.0, "kernel_noise": 0.05, "kernel_length": 0.5}
        )
    elif args.init_strategy == "median":
        init_strategy = init_to_median(num_samples=10)
    elif args.init_strategy == "feasible":
        init_strategy = init_to_feasible()
    elif args.init_strategy == "sample":
        init_strategy = init_to_sample()
    elif args.init_strategy == "uniform":
        init_strategy = init_to_uniform(radius=1)
    kernel = NUTS(model_with_confounding, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thinning,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y, M)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for gaussian process predictions
def predict(rng_key, X, Y, X_test, var, length, noise, intercept, beta):
    # compute kernels between train and test data, etc.
    k_pp = kernel(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = kernel(X_test, X, var, length, noise, include_noise=False)
    k_XX = kernel(X, X, var, length, noise, include_noise=True)
    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(
        rng_key, X_test.shape[:1]
    )
    mean = intercept + jnp.matmul(X_test, beta) + jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y - intercept - jnp.matmul(X, beta)))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for gaussian process predictions
def causal_predict(rng_key, X, Y, X_test, var, length, sigma_y, intercept, beta):
    # compute kernels between train and test data, etc.
    noise = jnp.square(sigma_y)
    k_pp = kernel(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = kernel(X_test, X, var, length, noise, include_noise=False)
    k_XX = kernel(X, X, var, length, noise, include_noise=True)
    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(
        rng_key, X_test.shape[:1]
    )

    gp_component = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y - intercept - jnp.matmul(X, beta)))
    mean = intercept + jnp.matmul(X_test, beta) + gp_component
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise, gp_component



# create artificial regression dataset
def get_data(N=100, sigma_obs=0.15, N_test=400, num_features=5):
    np.random.seed(0)
    X = np.random.normal(size=(N, num_features))# jnp.linspace(-1, 1, N)
    Y = X[:, 1] + 0.2 * jnp.power(X[:, 1], 3.0) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
    Y += sigma_obs * np.random.randn(N)
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N, num_features)
    assert Y.shape == (N,)

    X_test = np.random.normal(size=(N_test, num_features)) # jnp.linspace(-1, 1, N)

    return X, Y, X_test


def main(args):

    X, Y, X_test = get_data(N=args.num_data, num_features=args.num_features)

    mice = pd.read_csv("data/micedata.csv")
    Y = mice['y']
    X = mice.loc[:, "Igfbp2":"Veph1"]
    X = X / np.std(X, axis=0)
    M = 3


    qtiles=[0, 0.025, 0.25, 0.5, 0.75, 0.975]
    X_test = np.repeat([np.median(X, axis=0)], repeats=len(qtiles)*X.shape[1], axis=0)
    for i in range(X_test.shape[1]):
        qs = np.quantile(X.iloc[:, i], q=qtiles)
        X_test[(len(qtiles)*i):(len(qtiles)*i+len(qtiles)), i] = qs

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, jnp.array(X), jnp.array(Y), M)

    # do predictino
    vmap_args = (
        random.split(rng_key_predict, samples["kernel_var"].shape[0]),
        samples["kernel_var"],
        samples["length_scale"],
        samples["sigma_y"],
        samples["intercept"],
        samples["beta_coef"],
    )
    means, predictions, gp_component = vmap(
        lambda rng_key, var, length, sigma_y, intercept, beta: causal_predict(
            rng_key, jnp.array(X), jnp.array(Y), jnp.array(X_test), var, length, sigma_y, intercept, beta
        )
    )(*vmap_args)

    mean_prediction = np.mean(means, axis=0)
    mean_percentiles = np.percentile(means, [2.5, 97.5], axis=0)

    results_dict = {"samples" : samples, "mean_prediction" : mean_prediction, "mean_percentiles" : mean_percentiles, "gp_component" : gp_component}
    pkl.dump(results_dict, open( "gp_samples.p", "wb" ))

if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.7.2")
    parser = argparse.ArgumentParser(description="Gaussian Process example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--thinning", nargs="?", default=2, type=int)
    parser.add_argument("--num-data", nargs="?", default=25, type=int)
    parser.add_argument("--num-features", nargs="?", default=5, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--init-strategy",
        default="median",
        type=str,
        choices=["median", "feasible", "value", "uniform", "sample"],
    )
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
