"""Minimal reference implementation: Active Learning for Derivative-Based
Global Sensitivity Analysis (arXiv:2407.09739, NeurIPS 2024).

Demonstrates DGSM-UR acquisition for sequentially improving derivative-based
sensitivity estimates under a GP surrogate.

Run:
    python reference_impl_2407.09739.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# RBF Kernel GP with gradient posterior -------------------------------------
# ---------------------------------------------------------------------------
class GPWithGradient:
    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        self.ls = length_scale
        self.noise = noise
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.K_inv: np.ndarray | None = None

    def _kernel(self, X1, X2):
        sq = cdist(X1, X2, "sqeuclidean")
        return np.exp(-0.5 * sq / self.ls**2)

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
        self.alpha = self.K_inv @ y

    def predict(self, Xs):
        Ks = self._kernel(Xs, self.X)
        mu = Ks @ self.alpha
        v = self._kernel(Xs, Xs) - Ks @ self.K_inv @ Ks.T
        return mu, np.maximum(np.diag(v), 0.0)

    def gradient_variance(self, Xs):
        """Posterior variance of ∂f/∂x_i at Xs, for each input dimension."""
        n_test, d = Xs.shape
        grad_var = np.zeros((n_test, d))

        for j in range(n_test):
            xs = Xs[j:j+1]
            diff = xs - self.X  # (N, d)
            Ks = self._kernel(xs, self.X).flatten()  # (N,)

            for i in range(d):
                dk_dxi = -diff[:, i] / self.ls**2 * Ks  # (N,)
                d2k_dxi2 = (1.0 / self.ls**2) * (1.0 - diff[:, i]**2 / self.ls**2) * Ks

                prior_var = 1.0 / self.ls**2
                posterior_reduction = dk_dxi @ self.K_inv @ dk_dxi
                grad_var[j, i] = max(prior_var - posterior_reduction, 0.0)

        return grad_var


# ---------------------------------------------------------------------------
# True function: Ishigami (3D, but we use 2D for demo) ----------------------
# ---------------------------------------------------------------------------
def ishigami_2d(x: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    """Ishigami function (first 2 dims): f = sin(x1) + a*sin²(x2)."""
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1])**2


def true_dgsm(f, d: int, N: int = 50000, h: float = 1e-4, seed: int = 0):
    """Monte Carlo estimate of DGSM: ν_i = E[(∂f/∂x_i)²]."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-np.pi, np.pi, (N, d))
    dgsm = np.zeros(d)
    y0 = f(X)
    for i in range(d):
        X_h = X.copy()
        X_h[:, i] += h
        grad_i = (f(X_h) - y0) / h
        dgsm[i] = np.mean(grad_i**2)
    return dgsm


# ---------------------------------------------------------------------------
# DGSM-UR acquisition -------------------------------------------------------
# ---------------------------------------------------------------------------
def dgsm_ur_acquisition(candidates: np.ndarray, gp: GPWithGradient,
                        eval_grid: np.ndarray) -> np.ndarray:
    """DGSM uncertainty reduction: score each candidate by how much observing
    it would reduce posterior variance of gradient across the domain."""
    grad_var = gp.gradient_variance(eval_grid)  # (M, d)
    current_uncertainty = grad_var.sum()

    scores = np.zeros(len(candidates))
    for idx, xc in enumerate(candidates):
        Ks_c = gp._kernel(xc.reshape(1, -1), gp.X).flatten()
        _, var_c = gp.predict(xc.reshape(1, -1))
        var_c = var_c[0] + gp.noise

        if var_c < 1e-12:
            continue

        Ks_grid = gp._kernel(eval_grid, gp.X)
        k_cg = gp._kernel(eval_grid, xc.reshape(1, -1)).flatten()
        reduction = np.sum(k_cg**2) / var_c
        scores[idx] = reduction

    return scores


# ---------------------------------------------------------------------------
# Sequential DGSM estimation loop -------------------------------------------
# ---------------------------------------------------------------------------
def run_al_dgsm(d: int = 2, n_init: int = 5, n_seq: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    bounds = (-np.pi, np.pi)

    X_train = rng.uniform(*bounds, (n_init, d))
    y_train = ishigami_2d(X_train)

    gp = GPWithGradient(length_scale=1.5)
    gp.fit(X_train, y_train)

    eval_grid = rng.uniform(*bounds, (200, d))
    candidates = rng.uniform(*bounds, (500, d))
    true_nu = true_dgsm(ishigami_2d, d)

    print("Active Learning for DGSM (arXiv:2407.09739)")
    print("=" * 50)
    print(f"True DGSM: nu = [{', '.join(f'{v:.3f}' for v in true_nu)}]\n")

    for step in range(n_seq):
        scores = dgsm_ur_acquisition(candidates, gp, eval_grid)
        best_idx = np.argmax(scores)
        x_new = candidates[best_idx:best_idx+1]
        y_new = ishigami_2d(x_new)

        X_train = np.vstack([X_train, x_new])
        y_train = np.append(y_train, y_new)
        gp.fit(X_train, y_train)

        if (step + 1) % 5 == 0:
            est_nu = estimate_dgsm_from_gp(gp, d, bounds, N=2000, seed=step)
            rmse = np.sqrt(np.mean((est_nu - true_nu)**2))
            print(f"  Step {step+1:2d}: est_nu=[{', '.join(f'{v:.3f}' for v in est_nu)}], "
                  f"RMSE={rmse:.4f}, n={len(X_train)}")

    return gp, X_train


def estimate_dgsm_from_gp(gp: GPWithGradient, d: int, bounds, N: int = 2000,
                           seed: int = 0, h: float = 0.01):
    """Estimate DGSM from GP posterior mean via finite differences."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(bounds[0], bounds[1], (N, d))
    mu0, _ = gp.predict(X)
    dgsm = np.zeros(d)
    for i in range(d):
        X_h = X.copy()
        X_h[:, i] += h
        mu_h, _ = gp.predict(X_h)
        grad_i = (mu_h - mu0) / h
        dgsm[i] = np.mean(grad_i**2)
    return dgsm


# ---------------------------------------------------------------------------
# Compare AL vs random baseline ----------------------------------------------
# ---------------------------------------------------------------------------
def run_random_baseline(d: int = 2, n_total: int = 25, seed: int = 42):
    rng = np.random.default_rng(seed)
    bounds = (-np.pi, np.pi)
    X = rng.uniform(*bounds, (n_total, d))
    y = ishigami_2d(X)
    gp = GPWithGradient(length_scale=1.5)
    gp.fit(X, y)
    est_nu = estimate_dgsm_from_gp(gp, d, bounds, N=2000)
    return est_nu


if __name__ == "__main__":
    d = 2
    true_nu = true_dgsm(ishigami_2d, d)

    print("--- Active Learning (DGSM-UR) ---")
    gp_al, X_al = run_al_dgsm(d=d, n_init=5, n_seq=20)
    est_al = estimate_dgsm_from_gp(gp_al, d, (-np.pi, np.pi), N=5000)

    print("\n--- Random Baseline (same budget) ---")
    est_rand = run_random_baseline(d=d, n_total=25)
    print(f"  Random est_nu=[{', '.join(f'{v:.3f}' for v in est_rand)}]")

    print(f"\n--- Comparison (budget = 25) ---")
    print(f"  True DGSM:    [{', '.join(f'{v:.3f}' for v in true_nu)}]")
    print(f"  AL (DGSM-UR): [{', '.join(f'{v:.3f}' for v in est_al)}], "
          f"RMSE={np.sqrt(np.mean((est_al-true_nu)**2)):.4f}")
    print(f"  Random:       [{', '.join(f'{v:.3f}' for v in est_rand)}], "
          f"RMSE={np.sqrt(np.mean((est_rand-true_nu)**2)):.4f}")
