"""Minimal reference implementation: Sequential Bayesian Experimental Design
for Prediction (arXiv:2603.16756).

Demonstrates the KOH (Kennedy-O'Hagan) framework with MI-based sequential
experimental design to optimally place physical experiments by leveraging
a computer model surrogate.

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
    def __init__(self, ls: float = 0.3, noise: float = 1e-4):
        self.ls = ls
        self.noise = noise
        self.X = None
        self.y = None
        self._K_inv = None

    def _k(self, A, B):
        return np.exp(-0.5 * cdist(A, B, "sqeuclidean") / self.ls ** 2)

    def fit(self, X, y):
        self.X, self.y = X.copy(), y.copy()
        K = self._k(X, X) + self.noise * np.eye(len(y))
        self._K_inv = np.linalg.inv(K)

    def predict(self, Xs):
        Ks = self._k(Xs, self.X)
        mu = Ks @ self._K_inv @ self.y
        var = np.clip(1.0 - np.einsum("ij,jk,ik->i", Ks, self._K_inv, Ks),
                      1e-9, None)
        return mu, var

    def predict_cov(self, Xs):
        Ks = self._k(Xs, self.X)
        Kss = self._k(Xs, Xs)
        cov = Kss - Ks @ self._K_inv @ Ks.T
        return Ks @ self._K_inv @ self.y, cov


# ---------------------------------------------------------------------------
# Kennedy-O'Hagan model (simplified) ----------------------------------------
# ---------------------------------------------------------------------------
class KOHModel:
    """y_phys(x) = eta(x) + delta(x) + noise.
    eta: GP surrogate for computer model, delta: GP for discrepancy."""

    def __init__(self, ls_eta=0.3, ls_delta=0.5, noise=0.01):
        self.gp_eta = GP(ls=ls_eta, noise=1e-4)
        self.gp_delta = GP(ls=ls_delta, noise=1e-4)
        self.noise = noise

    def fit(self, X_sim, y_sim, X_phys, y_phys):
        self.gp_eta.fit(X_sim, y_sim)
        eta_at_phys, _ = self.gp_eta.predict(X_phys)
        residuals = y_phys - eta_at_phys
        self.gp_delta.fit(X_phys, residuals)

    def predict(self, Xs):
        mu_eta, var_eta = self.gp_eta.predict(Xs)
        mu_delta, var_delta = self.gp_delta.predict(Xs)
        mu = mu_eta + mu_delta
        var = var_eta + var_delta + self.noise
        return mu, var

    def predictive_entropy(self, Xs):
        _, var = self.predict(Xs)
        return 0.5 * np.log(2 * np.pi * np.e * var)


# ---------------------------------------------------------------------------
# MI-based acquisition -------------------------------------------------------
# ---------------------------------------------------------------------------
def mi_acquisition(koh: KOHModel, candidates: np.ndarray,
                   prediction_grid: np.ndarray) -> np.ndarray:
    """Approximate mutual information gain from observing at each candidate.
    MI(y*; y_new | D) ≈ H(y* | D) - E[H(y* | D ∪ {x_new, y_new})]
    Simplified: use variance reduction at prediction_grid as proxy."""
    _, var_prior = koh.predict(prediction_grid)
    H_prior = 0.5 * np.log(2 * np.pi * np.e * var_prior).sum()

    mi_scores = np.zeros(len(candidates))
    for i, x_new in enumerate(candidates):
        x_new_2d = x_new.reshape(1, -1)
        var_red_eta = _variance_reduction(koh.gp_eta, x_new_2d, prediction_grid)
        var_red_delta = _variance_reduction(koh.gp_delta, x_new_2d, prediction_grid)
        total_reduction = (var_red_eta + var_red_delta).sum()
        mi_scores[i] = total_reduction

    return mi_scores


def _variance_reduction(gp: GP, x_new: np.ndarray,
                        x_pred: np.ndarray) -> np.ndarray:
    """Predicted variance reduction at x_pred from adding observation at x_new."""
    if gp.X is None:
        return np.zeros(len(x_pred))
    Ks_new = gp._k(x_pred, x_new).flatten()
    k_new_new = 1.0 + gp.noise
    Ks_cross = gp._k(x_new, gp.X)
    v = Ks_cross @ gp._K_inv @ gp._k(gp.X, x_new)
    cond_var = k_new_new - v.item()
    if cond_var < 1e-9:
        return np.zeros(len(x_pred))
    Ks_pred = gp._k(x_pred, gp.X)
    proj = Ks_new - Ks_pred @ gp._K_inv @ gp._k(gp.X, x_new).flatten()
    return proj ** 2 / cond_var


# ---------------------------------------------------------------------------
# Sequential BED loop --------------------------------------------------------
# ---------------------------------------------------------------------------
def sequential_bed(
    computer_model,
    physical_system,
    bounds: np.ndarray,
    n_sim: int = 50,
    n_init_phys: int = 3,
    n_rounds: int = 15,
    n_candidates: int = 200,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]

    X_sim = rng.uniform(lo, hi, (n_sim, d))
    y_sim = np.array([computer_model(x) for x in X_sim])

    X_phys = rng.uniform(lo, hi, (n_init_phys, d))
    y_phys = np.array([physical_system(x) for x in X_phys])

    pred_grid = np.linspace(lo, hi, 30).reshape(-1, d) if d == 1 else \
        rng.uniform(lo, hi, (100, d))

    koh = KOHModel()
    rmse_history = []

    for t in range(n_rounds):
        koh.fit(X_sim, y_sim, X_phys, y_phys)

        mu_pred, _ = koh.predict(pred_grid)
        y_true = np.array([physical_system(x) for x in pred_grid])
        rmse = np.sqrt(((mu_pred - y_true) ** 2).mean())
        rmse_history.append(rmse)

        candidates = rng.uniform(lo, hi, (n_candidates, d))
        mi = mi_acquisition(koh, candidates, pred_grid)
        x_next = candidates[np.argmax(mi)]

        y_next = physical_system(x_next)
        X_phys = np.vstack([X_phys, x_next.reshape(1, -1)])
        y_phys = np.append(y_phys, y_next)

        if (t + 1) % 3 == 0:
            print(f"  Round {t+1}: RMSE={rmse:.4f}, n_phys={len(y_phys)}")

    koh.fit(X_sim, y_sim, X_phys, y_phys)
    return koh, X_phys, y_phys, rmse_history


# ---------------------------------------------------------------------------
# Demo -----------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def computer_model(x):
        return np.sin(3 * x[0]) + 0.5 * x[0]

    def physical_system(x):
        return np.sin(3 * x[0]) + 0.5 * x[0] + 0.2 * np.cos(5 * x[0])

    bounds = np.array([[0.0, 2.0]])

    print("Sequential BED with KOH model (MI criterion)")
    print("Computer model: sin(3x) + 0.5x")
    print("Physical system: sin(3x) + 0.5x + 0.2*cos(5x)  [discrepancy]")
    print("-" * 55)

    koh, X_phys, y_phys, rmse_hist = sequential_bed(
        computer_model, physical_system, bounds,
        n_sim=40, n_init_phys=3, n_rounds=15
    )
    print(f"\nFinal RMSE: {rmse_hist[-1]:.4f}")
    print(f"Total physical experiments: {len(y_phys)}")
    print(f"RMSE reduction: {rmse_hist[0]:.4f} → {rmse_hist[-1]:.4f}")
