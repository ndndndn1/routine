"""Minimal reference implementation for arXiv:2409.15307.

Reproduces:
  1) GP surrogate for forward model in Bayesian inverse problems.
  2) ILUES-based iterative refinement for multimodal posteriors.
  3) Bimodal posterior recovery demo.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm


# ---------------------------------------------------------------------------
# GP surrogate ---------------------------------------------------------------
# ---------------------------------------------------------------------------
class GP:
    def __init__(self, ls: float = 0.5, noise: float = 1e-3):
        self.ls, self.noise = ls, noise
        self.X = np.empty((0, 0))
        self.y = np.empty(0)

    def fit(self, X, y):
        self.X, self.y = X.copy(), y.copy()
        K = np.exp(-0.5 * cdist(X, X, "sqeuclidean") / self.ls ** 2) + self.noise * np.eye(len(y))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))

    def predict(self, Xs):
        if self.X.size == 0:
            return np.zeros(len(Xs)), np.ones(len(Xs))
        Ks = np.exp(-0.5 * cdist(Xs, self.X, "sqeuclidean") / self.ls ** 2)
        mu = Ks @ self._alpha
        v = np.linalg.solve(self._L, Ks.T)
        var = np.clip(1.0 - (v * v).sum(0), 1e-9, None)
        return mu, var


# ---------------------------------------------------------------------------
# Log-likelihood using GP surrogate ------------------------------------------
# ---------------------------------------------------------------------------
def log_likelihood_surrogate(theta: np.ndarray, y_obs: float, gp: GP,
                              sigma_obs: float = 0.1) -> float:
    mu, var = gp.predict(theta.reshape(1, -1))
    return float(-0.5 * ((mu[0] - y_obs) ** 2) / (sigma_obs ** 2 + var[0]))


# ---------------------------------------------------------------------------
# ILUES-like ensemble update -------------------------------------------------
# ---------------------------------------------------------------------------
def ilues_update(ensemble: np.ndarray, y_obs: float, gp: GP,
                 sigma_obs: float = 0.1, local_scale: float = 0.3) -> np.ndarray:
    n, d = ensemble.shape
    log_liks = np.array([log_likelihood_surrogate(e, y_obs, gp, sigma_obs)
                         for e in ensemble])
    weights = np.exp(log_liks - log_liks.max())
    weights /= weights.sum()

    new_ensemble = np.zeros_like(ensemble)
    rng = np.random.default_rng()
    for i in range(n):
        local_idx = np.argsort(cdist(ensemble[i:i + 1], ensemble).ravel())[:max(3, n // 5)]
        local_mean = np.average(ensemble[local_idx], weights=weights[local_idx], axis=0)
        local_cov = local_scale * np.cov(ensemble[local_idx].T) + 1e-6 * np.eye(d)
        new_ensemble[i] = rng.multivariate_normal(local_mean, local_cov)
    return new_ensemble


# ---------------------------------------------------------------------------
# Adaptive GP + ILUES iteration ----------------------------------------------
# ---------------------------------------------------------------------------
def adaptive_ilues_gp(forward_model, y_obs: float, bounds: np.ndarray,
                      n_ensemble: int = 100, n_iter: int = 8,
                      n_init_gp: int = 30, sigma_obs: float = 0.1,
                      seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = bounds.shape[0]

    X_gp = rng.uniform(bounds[:, 0], bounds[:, 1], (n_init_gp, d))
    y_gp = np.array([forward_model(x) for x in X_gp])
    gp = GP(ls=0.5)
    gp.fit(X_gp, y_gp)

    ensemble = rng.uniform(bounds[:, 0], bounds[:, 1], (n_ensemble, d))

    for it in range(n_iter):
        ensemble = ilues_update(ensemble, y_obs, gp, sigma_obs)
        ensemble = np.clip(ensemble, bounds[:, 0], bounds[:, 1])

        n_new = max(3, n_ensemble // 10)
        new_idx = rng.choice(n_ensemble, n_new, replace=False)
        X_new = ensemble[new_idx]
        y_new = np.array([forward_model(x) for x in X_new])
        X_gp = np.vstack([X_gp, X_new])
        y_gp = np.append(y_gp, y_new)
        gp.fit(X_gp, y_gp)

    return ensemble


# ---------------------------------------------------------------------------
# Demo: bimodal forward model -----------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def bimodal_forward(theta):
        return float(theta[0] ** 2 + 0.5 * theta[1] ** 2)

    y_obs = 1.0
    bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])

    posterior_samples = adaptive_ilues_gp(
        bimodal_forward, y_obs, bounds,
        n_ensemble=150, n_iter=10, n_init_gp=40, seed=42
    )

    residuals = np.array([abs(bimodal_forward(s) - y_obs) for s in posterior_samples])
    good = residuals < 0.5
    print(f"Samples within tolerance: {good.sum()}/{len(posterior_samples)}")
    if good.sum() > 0:
        good_samples = posterior_samples[good]
        print(f"Mean theta: {good_samples.mean(axis=0)}")
        print(f"Std theta:  {good_samples.std(axis=0)}")
        pos_x = good_samples[good_samples[:, 0] > 0]
        neg_x = good_samples[good_samples[:, 0] < 0]
        print(f"Mode 1 (x>0): {len(pos_x)} samples, mean={pos_x.mean(axis=0) if len(pos_x) else 'N/A'}")
        print(f"Mode 2 (x<0): {len(neg_x)} samples, mean={neg_x.mean(axis=0) if len(neg_x) else 'N/A'}")
    print(f"Total forward model calls: {40 + 10 * 15}")
