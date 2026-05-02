"""Minimal reference implementation for arXiv:2507.17713.

Reproduces:
  1) GP surrogate for PDE-to-observable map.
  2) Posterior-focused sequential design (IPVR acquisition).
  3) 1D Darcy-like inverse problem demo.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# GP surrogate ---------------------------------------------------------------
# ---------------------------------------------------------------------------
class GP:
    def __init__(self, ls: float = 0.4, noise: float = 1e-3):
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
# Simple 1D "Darcy" forward model -------------------------------------------
# ---------------------------------------------------------------------------
def darcy_forward(log_kappa: np.ndarray, n_grid: int = 20) -> np.ndarray:
    kappa = np.exp(log_kappa)
    dx = 1.0 / (n_grid + 1)
    f = np.ones(n_grid)
    A = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        k_left = kappa[i] if i > 0 else kappa[0]
        k_right = kappa[min(i + 1, len(kappa) - 1)]
        k_center = kappa[min(i, len(kappa) - 1)]
        A[i, i] = (k_left + 2 * k_center + k_right) / dx ** 2
        if i > 0:
            A[i, i - 1] = -(k_left + k_center) / (2 * dx ** 2)
        if i < n_grid - 1:
            A[i, i + 1] = -(k_center + k_right) / (2 * dx ** 2)
    u = np.linalg.solve(A, f)
    obs_idx = [n_grid // 4, n_grid // 2, 3 * n_grid // 4]
    return u[obs_idx]


# ---------------------------------------------------------------------------
# Posterior-weighted variance reduction (IPVR) acquisition -------------------
# ---------------------------------------------------------------------------
def ipvr_acquisition(gp: GP, X_cand: np.ndarray,
                     posterior_weights: np.ndarray) -> np.ndarray:
    _, var = gp.predict(X_cand)
    return var * posterior_weights


def approximate_posterior_weights(gp: GP, X_cand: np.ndarray, y_obs: np.ndarray,
                                  sigma_obs: float = 0.1) -> np.ndarray:
    mu, var = gp.predict(X_cand)
    log_lik = np.zeros(len(X_cand))
    for j in range(len(y_obs)):
        mu_j, var_j = gp.predict(X_cand)
        log_lik += -0.5 * ((mu_j - y_obs[j]) ** 2) / (sigma_obs ** 2 + var_j)
    weights = np.exp(log_lik - log_lik.max())
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Sequential Bayesian Design loop -------------------------------------------
# ---------------------------------------------------------------------------
def sbd_loop(forward_model, y_obs: np.ndarray, bounds: np.ndarray,
             n_init: int = 15, n_iter: int = 15, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = bounds.shape[0]

    X_train = rng.uniform(bounds[:, 0], bounds[:, 1], (n_init, d))
    y_train = np.array([forward_model(x) for x in X_train])
    y_scalar = np.array([np.sum(yt ** 2) for yt in y_train])

    gp = GP(ls=0.5)
    gp.fit(X_train, y_scalar)

    for it in range(n_iter):
        X_cand = rng.uniform(bounds[:, 0], bounds[:, 1], (200, d))
        weights = approximate_posterior_weights(gp, X_cand, y_obs)
        acq = ipvr_acquisition(gp, X_cand, weights)
        best_idx = int(np.argmax(acq))
        x_new = X_cand[best_idx]
        y_new = forward_model(x_new)
        y_new_scalar = np.sum(y_new ** 2)
        X_train = np.vstack([X_train, x_new])
        y_scalar = np.append(y_scalar, y_new_scalar)
        gp.fit(X_train, y_scalar)

    return {"X_train": X_train, "gp": gp, "n_forward_calls": n_init + n_iter}


# ---------------------------------------------------------------------------
# Demo -----------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    d = 3
    bounds = np.array([[-1.0, 1.0]] * d)

    theta_true = np.array([0.5, -0.3, 0.2])
    y_obs = darcy_forward(theta_true)
    print(f"True theta: {theta_true}")
    print(f"Observed y: {y_obs}")

    result = sbd_loop(darcy_forward, y_obs, bounds, n_init=20, n_iter=20, seed=42)
    gp = result["gp"]

    rng = np.random.default_rng(99)
    X_test = rng.uniform(bounds[:, 0], bounds[:, 1], (500, d))
    y_obs_scalar = np.sum(y_obs ** 2)
    mu, var = gp.predict(X_test)
    residuals = np.abs(mu - y_obs_scalar)
    best_idx = np.argmin(residuals)
    print(f"\nBest recovered theta: {X_test[best_idx]}")
    print(f"Residual: {residuals[best_idx]:.4f}")
    print(f"Forward model calls: {result['n_forward_calls']}")
