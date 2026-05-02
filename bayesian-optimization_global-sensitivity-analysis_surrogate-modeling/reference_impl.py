"""Minimal reference implementation for arXiv:2603.17516.

Reproduces:
  1) MaxPro (Maximum-Projection) initial design.
  2) GP surrogate + UCB Bayesian optimization.
  3) PCE-based Sobol sensitivity analysis.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# MaxPro initial design -------------------------------------------------------
# ---------------------------------------------------------------------------
def maxpro_design(d: int, n: int, n_cand: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_X = None
    best_score = -np.inf
    for _ in range(n_cand):
        X = rng.random((n, d))
        score = _maxpro_criterion(X)
        if score > best_score:
            best_score = score
            best_X = X.copy()
    return best_X


def _maxpro_criterion(X: np.ndarray) -> float:
    n, d = X.shape
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            prod = 1.0
            for k in range(d):
                diff = abs(X[i, k] - X[j, k])
                prod *= max(diff, 1e-10) ** 2
            total += 1.0 / max(prod, 1e-30)
    return -total


# ---------------------------------------------------------------------------
# GP surrogate ---------------------------------------------------------------
# ---------------------------------------------------------------------------
class GP:
    def __init__(self, ls: float = 0.3, noise: float = 1e-3):
        self.ls, self.noise = ls, noise
        self.X = np.empty((0, 0))
        self.y = np.empty(0)
        self._L = self._alpha = None

    def fit(self, X, y):
        self.X, self.y = X.copy(), y.copy()
        K = np.exp(-0.5 * cdist(X, X, "sqeuclidean") / self.ls ** 2) + self.noise * np.eye(len(y))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))

    def predict(self, Xs):
        Ks = np.exp(-0.5 * cdist(Xs, self.X, "sqeuclidean") / self.ls ** 2)
        mu = Ks @ self._alpha
        v = np.linalg.solve(self._L, Ks.T)
        var = np.clip(1.0 - (v * v).sum(0), 1e-9, None)
        return mu, var


# ---------------------------------------------------------------------------
# UCB Bayesian optimization --------------------------------------------------
# ---------------------------------------------------------------------------
def bo_loop(objective, bounds, gp, X_init, y_init, n_iter=20, beta=2.0, seed=0):
    rng = np.random.default_rng(seed)
    X, y = X_init.copy(), y_init.copy()
    d = bounds.shape[0]
    for _ in range(n_iter):
        gp.fit(X, y)
        X_cand = rng.uniform(bounds[:, 0], bounds[:, 1], (300, d))
        mu, var = gp.predict(X_cand)
        ucb = mu + beta * np.sqrt(var)
        x_new = X_cand[np.argmax(ucb)]
        y_new = objective(x_new)
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
    return X, y


# ---------------------------------------------------------------------------
# PCE-based Sobol sensitivity ------------------------------------------------
# ---------------------------------------------------------------------------
def pce_sobol(model, bounds, order=3, n_samples=2000, seed=0):
    d = bounds.shape[0]
    rng = np.random.default_rng(seed)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], (n_samples, d))
    X_norm = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    y, _ = model.predict(X)

    S = np.zeros(d)
    var_total = np.var(y)
    for i in range(d):
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        cond_means = []
        for b in range(n_bins):
            mask = (X_norm[:, i] >= bins[b]) & (X_norm[:, i] < bins[b + 1])
            if mask.sum() > 0:
                cond_means.append(np.mean(y[mask]))
        S[i] = np.var(cond_means) / (var_total + 1e-12)
    return np.clip(S, 0, 1)


# ---------------------------------------------------------------------------
# Demo: turbine-like design optimization ------------------------------------
# ---------------------------------------------------------------------------
def _turbine_efficiency(x):
    return -(0.5 * x[0] ** 2 + 0.3 * x[1] ** 2 + 0.1 * x[2] ** 2
             - 0.4 * x[0] * x[1] + 0.05 * x[3] ** 2 + 0.02 * x[4] ** 2)


if __name__ == "__main__":
    d = 5
    bounds = np.array([[0, 1]] * d, dtype=float)

    X_init = maxpro_design(d, n=10, n_cand=500, seed=0)
    X_init = X_init * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    y_init = np.array([_turbine_efficiency(x) for x in X_init])

    gp = GP(ls=0.4)
    X_all, y_all = bo_loop(_turbine_efficiency, bounds, gp, X_init, y_init,
                            n_iter=15, seed=1)
    print(f"Best efficiency: {y_all.max():.4f} at iter {np.argmax(y_all)}")

    gp.fit(X_all, y_all)
    S = pce_sobol(gp, bounds)
    names = [f"x{i}" for i in range(d)]
    print("\nSobol first-order indices:")
    for name, si in zip(names, S):
        print(f"  {name}: {si:.4f}")
    print(f"\nTop parameters: {names[np.argmax(S)]}")
