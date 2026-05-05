"""Minimal reference implementation: BO-guided Surrogate for Bayesian Inverse
Problems (arXiv:2602.04537).

Demonstrates adaptive surrogate construction via Bayesian optimization
followed by surrogate-based MCMC posterior sampling.

Run:
    python reference_impl.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Expensive forward model (ground truth) ------------------------------------
# ---------------------------------------------------------------------------
def forward_model(theta: np.ndarray) -> np.ndarray:
    """Expensive forward model G(theta) — 1D for demo."""
    return np.sin(1.5 * theta) + 0.2 * theta**2 - 0.5 * np.cos(3 * theta)


# ---------------------------------------------------------------------------
# GP Surrogate ---------------------------------------------------------------
# ---------------------------------------------------------------------------
class GPSurrogate:
    def __init__(self, length_scale: float = 0.6, noise: float = 1e-6):
        self.ls = length_scale
        self.noise = noise

    def _kernel(self, X1, X2):
        sq = cdist(X1.reshape(-1, 1), X2.reshape(-1, 1), "sqeuclidean")
        return np.exp(-0.5 * sq / self.ls**2)

    def fit(self, X, y):
        self.X = np.atleast_1d(X).flatten()
        self.y = np.atleast_1d(y).flatten()
        K = self._kernel(self.X, self.X) + self.noise * np.eye(len(self.X))
        self.K_inv = np.linalg.inv(K)
        self.alpha = self.K_inv @ self.y

    def predict(self, Xs):
        Xs = np.atleast_1d(Xs).flatten()
        Ks = self._kernel(Xs, self.X)
        mu = Ks @ self.alpha
        v = self._kernel(Xs, Xs) - Ks @ self.K_inv @ Ks.T
        var = np.maximum(np.diag(v), 0.0)
        return mu, var


# ---------------------------------------------------------------------------
# Phase 1: BO-guided adaptive surrogate construction -------------------------
# ---------------------------------------------------------------------------
def posterior_relevance(theta_grid, gp, y_obs, sigma_obs):
    """Approximate posterior weight for guiding acquisition."""
    mu, _ = gp.predict(theta_grid)
    log_lik = -0.5 * ((y_obs - mu) / sigma_obs)**2
    log_prior = -0.5 * (theta_grid / 2.0)**2  # N(0, 4)
    w = np.exp(log_lik + log_prior)
    return w / (w.sum() + 1e-30)


def bo_acquisition(theta_grid, gp, y_obs, sigma_obs):
    """alpha(theta) = sigma_GP(theta) * w(theta)."""
    _, var = gp.predict(theta_grid)
    sigma = np.sqrt(var)
    w = posterior_relevance(theta_grid, gp, y_obs, sigma_obs)
    return sigma * w


def build_surrogate_bo(y_obs, sigma_obs, n_init=3, n_bo=15, seed=42):
    """Phase 1: Build GP surrogate using BO acquisition."""
    rng = np.random.default_rng(seed)
    theta_train = rng.uniform(-4, 4, n_init)
    y_train = forward_model(theta_train)

    gp = GPSurrogate()
    gp.fit(theta_train, y_train)

    candidates = np.linspace(-4, 4, 300)

    for step in range(n_bo):
        acq = bo_acquisition(candidates, gp, y_obs, sigma_obs)
        idx = np.argmax(acq)
        theta_new = candidates[idx]
        y_new = forward_model(np.array([theta_new]))[0]

        theta_train = np.append(theta_train, theta_new)
        y_train = np.append(y_train, y_new)
        gp.fit(theta_train, y_train)

    return gp, theta_train


def build_surrogate_uniform(y_obs, sigma_obs, n_total=18, seed=42):
    """Baseline: uniform surrogate construction."""
    rng = np.random.default_rng(seed)
    theta_train = rng.uniform(-4, 4, n_total)
    y_train = forward_model(theta_train)
    gp = GPSurrogate()
    gp.fit(theta_train, y_train)
    return gp, theta_train


# ---------------------------------------------------------------------------
# Phase 2: Surrogate-based MCMC posterior sampling ---------------------------
# ---------------------------------------------------------------------------
def surrogate_mcmc(gp, y_obs, sigma_obs, n_samples=5000,
                   proposal_std=0.3, seed=123):
    """Metropolis-Hastings MCMC using GP surrogate likelihood."""
    rng = np.random.default_rng(seed)
    theta = 0.0
    samples = []
    accepted = 0

    mu_curr, var_curr = gp.predict(np.array([theta]))
    sigma_total = np.sqrt(sigma_obs**2 + var_curr[0])
    log_lik_curr = -0.5 * ((y_obs - mu_curr[0]) / sigma_total)**2
    log_prior_curr = -0.5 * (theta / 2.0)**2

    for _ in range(n_samples):
        theta_prop = theta + rng.normal(0, proposal_std)
        mu_prop, var_prop = gp.predict(np.array([theta_prop]))
        sigma_total_prop = np.sqrt(sigma_obs**2 + var_prop[0])
        log_lik_prop = -0.5 * ((y_obs - mu_prop[0]) / sigma_total_prop)**2
        log_prior_prop = -0.5 * (theta_prop / 2.0)**2

        log_alpha = (log_lik_prop + log_prior_prop) - (log_lik_curr + log_prior_curr)

        if np.log(rng.uniform()) < log_alpha:
            theta = theta_prop
            log_lik_curr = log_lik_prop
            log_prior_curr = log_prior_prop
            accepted += 1

        samples.append(theta)

    return np.array(samples), accepted / n_samples


# ---------------------------------------------------------------------------
# Demo: compare BO-guided vs uniform surrogate for inverse problem -----------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    theta_true = 1.5
    y_obs = forward_model(np.array([theta_true]))[0]
    sigma_obs = 0.1

    print("BO-guided Surrogate for Bayesian Inverse Problems")
    print("=" * 55)
    print(f"True theta = {theta_true:.3f}, y_obs = {y_obs:.3f}\n")

    # Phase 1: Build surrogates
    gp_bo, P_bo = build_surrogate_bo(y_obs, sigma_obs)
    gp_unif, P_unif = build_surrogate_uniform(y_obs, sigma_obs)

    # Phase 2: MCMC
    for label, gp in [("BO-guided", gp_bo), ("Uniform", gp_unif)]:
        samples, accept_rate = surrogate_mcmc(gp, y_obs, sigma_obs)
        burnin = len(samples) // 4
        posterior = samples[burnin:]

        mean = np.mean(posterior)
        std = np.std(posterior)
        median = np.median(posterior)
        ci_low, ci_high = np.percentile(posterior, [2.5, 97.5])

        print(f"[{label} surrogate]")
        print(f"  Forward model evals: {len(P_bo) if 'BO' in label else len(P_unif)}")
        print(f"  Posterior mean:   {mean:.3f}")
        print(f"  Posterior std:    {std:.3f}")
        print(f"  95% CI:           [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  |mean - true|:    {abs(mean - theta_true):.3f}")
        print(f"  MCMC accept rate: {accept_rate:.2%}")

        # Surrogate error at MAP
        mu_map, var_map = gp.predict(np.array([mean]))
        y_true_map = forward_model(np.array([mean]))[0]
        print(f"  Surrogate error at MAP: {abs(mu_map[0] - y_true_map):.4f}")
        print(f"  GP uncertainty at MAP:  {np.sqrt(var_map[0]):.4f}\n")
