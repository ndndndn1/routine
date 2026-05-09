#!/usr/bin/env python3
"""
PSP-GEN Reference Implementation (Simplified)
==============================================
Stochastic Inversion of the Process-Structure-Property Chain
in Materials Design through Deep, Generative Probabilistic Modeling.

arXiv: 2408.01114

This minimal implementation demonstrates:
  1. A synthetic Process -> Structure -> Property (PSP) forward chain
     with stochastic process variation (latent noise z).
  2. A surrogate model for the Structure -> Property mapping.
  3. MCMC-based inverse design: given a target property Y*, sample
     process parameters X_proc whose forward distribution covers Y*.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# 1.  Synthetic PSP forward chain
# ---------------------------------------------------------------------------

def process_to_structure(x_proc, z=None, noise_std=0.15):
    """
    Process -> Structure mapping with stochastic latent variable z.

    Parameters
    ----------
    x_proc : ndarray, shape (d_proc,)
        Process parameters (e.g. temperature, pressure, time).
    z : ndarray or None
        Latent noise representing process variation.  If None, sampled.
    noise_std : float
        Scale of stochastic process variation.

    Returns
    -------
    structure : ndarray, shape (d_struct,)
        Microstructure descriptor (e.g. grain size, phase fraction).
    """
    d_proc = x_proc.shape[0]
    d_struct = d_proc + 1  # structure has one more dimension
    if z is None:
        z = np.random.randn(d_struct) * noise_std

    # Non-linear deterministic mapping + stochastic perturbation
    W = np.array([
        [0.8, -0.3, 0.5],
        [0.4,  0.9, -0.2],
        [-0.2, 0.6,  0.7],
    ])[:d_struct, :d_proc]
    structure = np.tanh(W @ x_proc) + z
    return structure


def structure_to_property(structure):
    """
    Structure -> Property mapping (deterministic surrogate).

    Parameters
    ----------
    structure : ndarray, shape (d_struct,)

    Returns
    -------
    prop : float
        Scalar material property (e.g. yield strength).
    """
    # Quadratic + linear combination
    prop = 0.5 * np.sum(structure ** 2) + 0.3 * np.sum(structure) + 0.1
    return float(prop)


def psp_forward(x_proc, z=None, noise_std=0.15):
    """Full forward chain: Process -> Structure -> Property."""
    s = process_to_structure(x_proc, z=z, noise_std=noise_std)
    y = structure_to_property(s)
    return y, s


# ---------------------------------------------------------------------------
# 2.  Surrogate model: simple RBF-kernel regression (GP-like)
# ---------------------------------------------------------------------------

class SimpleSurrogate:
    """Kernel-ridge surrogate for Structure -> Property mapping."""

    def __init__(self, length_scale=1.0, reg=1e-4):
        self.length_scale = length_scale
        self.reg = reg
        self.X_train = None
        self.y_train = None
        self.alpha = None

    def _kernel(self, X1, X2):
        sq_dist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * sq_dist / self.length_scale ** 2)

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        K = self._kernel(self.X_train, self.X_train)
        K_reg = K + self.reg * np.eye(len(K))
        self.alpha = np.linalg.solve(K_reg, self.y_train)
        self.K_inv = np.linalg.inv(K_reg)

    def predict(self, X):
        X = np.atleast_2d(X)
        K_star = self._kernel(X, self.X_train)
        mu = K_star @ self.alpha
        # Predictive variance (diagonal only)
        K_ss = np.ones(len(X))  # k(x,x) = 1 for RBF
        var = K_ss - np.sum((K_star @ self.K_inv) * K_star, axis=1)
        var = np.maximum(var, 1e-10)
        return mu, var


# ---------------------------------------------------------------------------
# 3.  MCMC inverse design
# ---------------------------------------------------------------------------

def log_likelihood(x_proc, y_target, surrogate, n_mc=10, noise_std=0.15):
    """
    Estimate log p(Y* | X_proc) via Monte-Carlo over latent z.

    For each z sample, run process->structure, then evaluate surrogate
    for structure->property.  Average likelihood over z samples.
    """
    log_likes = []
    d_struct = x_proc.shape[0] + 1
    Z_all = np.random.randn(n_mc, d_struct) * noise_std
    for i in range(n_mc):
        s = process_to_structure(x_proc, z=Z_all[i], noise_std=0.0)
        mu, var = surrogate.predict(s.reshape(1, -1))
        sigma = np.sqrt(var[0] + noise_std ** 2)
        ll = norm.logpdf(y_target, loc=mu[0], scale=sigma)
        log_likes.append(ll)
    # log-sum-exp for numerical stability
    max_ll = np.max(log_likes)
    return max_ll + np.log(np.mean(np.exp(np.array(log_likes) - max_ll)))


def mcmc_inverse_design(y_target, surrogate, d_proc=2, n_samples=500,
                        burn_in=200, proposal_std=0.15, noise_std=0.15):
    """
    Metropolis-Hastings MCMC to sample process parameters X_proc
    given a target property Y*.

    Returns
    -------
    samples : ndarray, shape (n_accepted, d_proc)
    """
    x_current = np.random.randn(d_proc) * 0.5
    ll_current = log_likelihood(x_current, y_target, surrogate,
                                noise_std=noise_std)

    samples = []
    n_accept = 0
    total_steps = n_samples + burn_in

    for step in range(total_steps):
        x_proposal = x_current + np.random.randn(d_proc) * proposal_std
        ll_proposal = log_likelihood(x_proposal, y_target, surrogate,
                                     noise_std=noise_std)

        log_alpha = ll_proposal - ll_current
        if np.log(np.random.rand()) < log_alpha:
            x_current = x_proposal
            ll_current = ll_proposal
            n_accept += 1

        if step >= burn_in:
            samples.append(x_current.copy())

    accept_rate = n_accept / total_steps
    return np.array(samples), accept_rate


# ---------------------------------------------------------------------------
# 4.  Demo
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    d_proc = 2
    noise_std = 0.15

    # --- Build training data for surrogate ---
    print("=" * 60)
    print("PSP-GEN Reference Implementation")
    print("=" * 60)

    n_train = 300
    print(f"\n[1] Generating {n_train} forward PSP samples for surrogate training...")
    structures = []
    properties = []
    for _ in range(n_train):
        x = np.random.randn(d_proc) * 0.8
        y, s = psp_forward(x, noise_std=noise_std)
        structures.append(s)
        properties.append(y)

    structures = np.array(structures)
    properties = np.array(properties)
    print(f"    Structure dim: {structures.shape[1]}, "
          f"Property range: [{properties.min():.3f}, {properties.max():.3f}]")

    # --- Train surrogate (Structure -> Property) ---
    print("\n[2] Training surrogate model (Structure -> Property)...")
    surrogate = SimpleSurrogate(length_scale=0.8, reg=1e-3)
    surrogate.fit(structures, properties)

    # Evaluate surrogate accuracy
    n_test = 50
    test_structs = []
    test_props = []
    for _ in range(n_test):
        x = np.random.randn(d_proc) * 0.8
        y, s = psp_forward(x, noise_std=noise_std)
        test_structs.append(s)
        test_props.append(y)
    test_structs = np.array(test_structs)
    test_props = np.array(test_props)
    pred_mu, pred_var = surrogate.predict(test_structs)
    rmse = np.sqrt(np.mean((pred_mu - test_props) ** 2))
    print(f"    Surrogate RMSE on test set: {rmse:.4f}")

    # --- Inverse design via MCMC ---
    y_target = 0.5
    print(f"\n[3] Inverse design: target property Y* = {y_target}")
    print("    Running MCMC to sample process parameters...")

    samples, accept_rate = mcmc_inverse_design(
        y_target, surrogate, d_proc=d_proc,
        n_samples=300, burn_in=150,
        proposal_std=0.12, noise_std=noise_std
    )
    print(f"    MCMC acceptance rate: {accept_rate:.2%}")
    print(f"    Number of posterior samples: {len(samples)}")

    # --- Validate inverse samples via forward chain ---
    print("\n[4] Validating inverse design samples via forward chain...")
    forward_props = []
    for x_proc in samples:
        # Average over multiple z realizations
        ys = [psp_forward(x_proc, noise_std=noise_std)[0] for _ in range(20)]
        forward_props.append(np.mean(ys))

    forward_props = np.array(forward_props)
    mean_prop = np.mean(forward_props)
    std_prop = np.std(forward_props)
    print(f"    Target property:           {y_target:.3f}")
    print(f"    Mean achieved property:    {mean_prop:.3f}")
    print(f"    Std of achieved property:  {std_prop:.3f}")
    print(f"    Mean absolute error:       {np.mean(np.abs(forward_props - y_target)):.4f}")

    # --- Show diversity of solutions ---
    print("\n[5] Diversity of inverse design solutions:")
    print(f"    Process parameter statistics (d_proc={d_proc}):")
    for d in range(d_proc):
        print(f"      x_proc[{d}]: mean={np.mean(samples[:, d]):.3f}, "
              f"std={np.std(samples[:, d]):.3f}, "
              f"range=[{np.min(samples[:, d]):.3f}, {np.max(samples[:, d]):.3f}]")

    # Pairwise diversity
    if len(samples) > 1:
        from scipy.spatial.distance import pdist
        diversity = np.mean(pdist(samples))
        print(f"    Mean pairwise distance:    {diversity:.4f}")

    print("\n" + "=" * 60)
    print("Done. PSP-GEN demonstrates that multiple process conditions")
    print("can achieve the same target property -- the inverse problem")
    print("is inherently one-to-many, captured by the MCMC posterior.")
    print("=" * 60)


if __name__ == "__main__":
    main()
