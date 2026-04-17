"""Minimal reference implementation: Gradient-based Active Learning for
Global Sensitivity Analysis (arXiv:2601.11790).

Demonstrates the sequential loop:
  1) Fit GP surrogate on current observations
  2) Estimate Sobol' indices via GP posterior
  3) Select next point by maximizing gradient-informed acquisition
  4) Evaluate simulator and augment dataset

Run:
    python reference_impl.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# GP surrogate with RBF kernel & gradient support ---------------------------
# ---------------------------------------------------------------------------
class GradientGP:
    def __init__(self, length_scales: np.ndarray, noise: float = 1e-4):
        self.ls = length_scales
        self.noise = noise
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self._L = None
        self._alpha = None

    def _k(self, A, B):
        d2 = cdist(A / self.ls, B / self.ls, metric="sqeuclidean")
        return np.exp(-0.5 * d2)

    def fit(self, X, y):
        self.X, self.y = X.copy(), y.copy()
        K = self._k(X, X) + self.noise * np.eye(len(y))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))

    def predict(self, Xs):
        Ks = self._k(Xs, self.X)
        mu = Ks @ self._alpha
        v = np.linalg.solve(self._L, Ks.T)
        var = np.clip(1.0 - (v * v).sum(0), 1e-9, None)
        return mu, var

    def gradient_variance(self, Xs):
        """Estimate variance of partial derivatives at Xs via GP posterior.
        For RBF: dk/dx_i = -1/l_i^2 * (x_i - x'_i) * k(x, x').
        Gradient variance indicates how uncertain the slope is — high values
        mean the sensitivity estimate for that dimension is unreliable."""
        n_pred, d = Xs.shape
        grad_var = np.zeros((n_pred, d))
        Ks = self._k(Xs, self.X)
        v = np.linalg.solve(self._L, Ks.T)
        for j in range(d):
            diff_j = (Xs[:, j:j+1] - self.X[:, j:j+1].T) / (self.ls[j] ** 2)
            dKs_j = -diff_j * Ks
            dv_j = np.linalg.solve(self._L, dKs_j.T)
            k_diag_deriv = 1.0 / (self.ls[j] ** 2)
            grad_var[:, j] = np.clip(
                k_diag_deriv - (dv_j * dv_j).sum(0) - (v * dv_j).sum(0) ** 2 / (1e-9 + (1.0 - (v * v).sum(0))),
                1e-9, None
            )
        return grad_var


# ---------------------------------------------------------------------------
# Sobol' index estimation via GP moments ------------------------------------
# ---------------------------------------------------------------------------
def estimate_sobol_indices(gp: GradientGP, bounds: np.ndarray,
                           n_mc: int = 5000, rng=None):
    """Monte Carlo estimate of first-order Sobol' indices from GP mean."""
    rng = rng or np.random.default_rng(0)
    d = bounds.shape[0]
    X_mc = rng.uniform(bounds[:, 0], bounds[:, 1], (n_mc, d))
    mu, _ = gp.predict(X_mc)
    var_total = mu.var()
    S = np.zeros(d)
    for i in range(d):
        unique_vals = np.linspace(bounds[i, 0], bounds[i, 1], 50)
        conditional_means = []
        for val in unique_vals:
            X_cond = X_mc.copy()
            X_cond[:, i] = val
            mu_cond, _ = gp.predict(X_cond)
            conditional_means.append(mu_cond.mean())
        S[i] = np.var(conditional_means) / (var_total + 1e-9)
    return S


# ---------------------------------------------------------------------------
# Gradient-informed acquisition function ------------------------------------
# ---------------------------------------------------------------------------
def gradient_acquisition(gp: GradientGP, candidates: np.ndarray) -> np.ndarray:
    """Sum of gradient variances across dimensions — high values indicate
    regions where sensitivity estimates are most uncertain."""
    grad_var = gp.gradient_variance(candidates)
    return grad_var.sum(axis=1)


# ---------------------------------------------------------------------------
# Active learning loop -------------------------------------------------------
# ---------------------------------------------------------------------------
def active_gsa(
    simulator,
    bounds: np.ndarray,
    n_init: int = 10,
    n_rounds: int = 20,
    n_candidates: int = 500,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    d = bounds.shape[0]
    ls = (bounds[:, 1] - bounds[:, 0]) / 3.0

    X = rng.uniform(bounds[:, 0], bounds[:, 1], (n_init, d))
    y = np.array([simulator(x) for x in X])

    gp = GradientGP(length_scales=ls)
    history = []

    for t in range(n_rounds):
        gp.fit(X, y)
        S = estimate_sobol_indices(gp, bounds, rng=rng)
        history.append(S.copy())

        candidates = rng.uniform(bounds[:, 0], bounds[:, 1], (n_candidates, d))
        acq = gradient_acquisition(gp, candidates)
        best = candidates[np.argmax(acq)]

        y_new = simulator(best)
        X = np.vstack([X, best.reshape(1, -1)])
        y = np.append(y, y_new)

        if (t + 1) % 5 == 0:
            print(f"  Round {t+1}: S = {S.round(3)}, n_obs = {len(y)}")

    gp.fit(X, y)
    S_final = estimate_sobol_indices(gp, bounds, rng=rng)
    return S_final, X, y, history


# ---------------------------------------------------------------------------
# Demo: Ishigami function ---------------------------------------------------
# ---------------------------------------------------------------------------
def ishigami(x, a=7.0, b=0.1):
    return np.sin(x[0]) + a * np.sin(x[1]) ** 2 + b * x[2] ** 4 * np.sin(x[0])


if __name__ == "__main__":
    bounds = np.array([[-np.pi, np.pi]] * 3)

    print("Active Learning for GSA on Ishigami function")
    print("True Sobol' indices ≈ S1=0.314, S2=0.442, S3=0.0")
    print("-" * 50)

    S, X, y, hist = active_gsa(ishigami, bounds, n_init=10, n_rounds=25)
    print(f"\nFinal estimated Sobol' indices: {S.round(3)}")
    print(f"Total simulator calls: {len(y)}")
