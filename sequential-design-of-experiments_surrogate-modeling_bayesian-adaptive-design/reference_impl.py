"""Minimal reference implementation: IP-SUR Sequential Design for
Bayesian Inverse Problems (arXiv:2402.16520).

Demonstrates posterior-weighted sequential design that adaptively refines a
GP surrogate in regions most relevant to the Bayesian inverse problem.

Run:
    python reference_impl.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# RBF Kernel GP surrogate ---------------------------------------------------
# ---------------------------------------------------------------------------
class GPSurrogate:
    def __init__(self, length_scale: float = 0.5, noise: float = 1e-6):
        self.ls = length_scale
        self.noise = noise
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.K_inv: np.ndarray | None = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq = cdist(X1, X2, "sqeuclidean")
        return np.exp(-0.5 * sq / self.ls**2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = X.copy()
        self.y_train = y.copy()
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X: np.ndarray):
        k_star = self._kernel(X, self.X_train)
        mu = k_star @ self.K_inv @ self.y_train
        v = self._kernel(X, X) - k_star @ self.K_inv @ k_star.T
        var = np.maximum(np.diag(v), 0.0)
        return mu, var


# ---------------------------------------------------------------------------
# True forward model (expensive black-box) ----------------------------------
# ---------------------------------------------------------------------------
def forward_model(theta: np.ndarray) -> np.ndarray:
    """G(theta): 1D toy forward model."""
    return np.sin(2 * theta) + 0.3 * theta


# ---------------------------------------------------------------------------
# Approximate posterior weight -----------------------------------------------
# ---------------------------------------------------------------------------
def posterior_weight(theta_grid: np.ndarray, gp: GPSurrogate,
                    y_obs: float, sigma_obs: float) -> np.ndarray:
    """w_n(theta) proportional to approximate posterior."""
    mu, _ = gp.predict(theta_grid)
    log_lik = -0.5 * ((y_obs - mu) / sigma_obs) ** 2
    log_prior = -0.5 * theta_grid.flatten() ** 2  # N(0,1) prior
    w = np.exp(log_lik + log_prior)
    w /= w.sum() + 1e-30
    return w


# ---------------------------------------------------------------------------
# IP-SUR acquisition function -----------------------------------------------
# ---------------------------------------------------------------------------
def ip_sur_acquisition(theta_cand: np.ndarray, gp: GPSurrogate,
                       theta_grid: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Expected posterior-weighted variance reduction for each candidate."""
    _, var_grid = gp.predict(theta_grid)
    scores = np.zeros(len(theta_cand))

    for i, tc in enumerate(theta_cand):
        k_cand_grid = gp._kernel(tc.reshape(1, -1), theta_grid).flatten()
        _, var_cand = gp.predict(tc.reshape(1, -1))
        var_c = var_cand[0] + gp.noise
        if var_c < 1e-12:
            continue
        delta_var = k_cand_grid**2 / var_c
        scores[i] = np.sum(w * delta_var)
    return scores


# ---------------------------------------------------------------------------
# Sequential Design loop ----------------------------------------------------
# ---------------------------------------------------------------------------
def sequential_design(y_obs: float, sigma_obs: float,
                      n_init: int = 3, n_seq: int = 12,
                      n_grid: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    domain = np.linspace(-3, 3, n_grid).reshape(-1, 1)

    X_train = rng.uniform(-3, 3, (n_init, 1))
    y_train = forward_model(X_train).flatten()

    gp = GPSurrogate(length_scale=0.5)
    gp.fit(X_train, y_train)

    print("IP-SUR Sequential Design for Bayesian Inverse Problem")
    print("=" * 55)
    print(f"Observation y_obs = {y_obs:.3f}, noise sigma = {sigma_obs:.3f}")
    print(f"Initial design: {n_init} points, sequential budget: {n_seq}\n")

    for step in range(n_seq):
        w = posterior_weight(domain, gp, y_obs, sigma_obs)
        _, var_grid = gp.predict(domain)
        H_n = np.sum(w * var_grid)

        scores = ip_sur_acquisition(domain, gp, domain, w)
        idx_best = np.argmax(scores)
        theta_new = domain[idx_best].reshape(1, -1)
        y_new = forward_model(theta_new).flatten()

        X_train = np.vstack([X_train, theta_new])
        y_train = np.append(y_train, y_new)
        gp.fit(X_train, y_train)

        if (step + 1) % 4 == 0 or step == 0:
            print(f"  Step {step+1:2d}: theta_new={theta_new[0,0]:+.3f}, "
                  f"H_n={H_n:.6f}, n_total={len(X_train)}")

    w_final = posterior_weight(domain, gp, y_obs, sigma_obs)
    _, var_final = gp.predict(domain)
    H_final = np.sum(w_final * var_final)
    theta_map = domain[np.argmax(w_final), 0]

    print(f"\nFinal: H_n = {H_final:.6f}, MAP estimate = {theta_map:.3f}")
    print(f"True G(MAP) = {forward_model(np.array([[theta_map]]))[0,0]:.3f}, "
          f"y_obs = {y_obs:.3f}")
    return gp, X_train, theta_map


# ---------------------------------------------------------------------------
# Compare IP-SUR vs random design -------------------------------------------
# ---------------------------------------------------------------------------
def random_design(y_obs: float, sigma_obs: float,
                  n_total: int = 15, n_grid: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    domain = np.linspace(-3, 3, n_grid).reshape(-1, 1)

    X_train = rng.uniform(-3, 3, (n_total, 1))
    y_train = forward_model(X_train).flatten()
    gp = GPSurrogate(length_scale=0.5)
    gp.fit(X_train, y_train)

    w = posterior_weight(domain, gp, y_obs, sigma_obs)
    _, var = gp.predict(domain)
    H = np.sum(w * var)
    theta_map = domain[np.argmax(w), 0]
    return H, theta_map


if __name__ == "__main__":
    y_obs = 0.8
    sigma_obs = 0.1

    gp, X_seq, map_seq = sequential_design(y_obs, sigma_obs)

    H_rand, map_rand = random_design(y_obs, sigma_obs, n_total=15)
    print(f"\n--- Comparison (same budget = 15 evals) ---")
    print(f"Random design:  H = {H_rand:.6f}, MAP = {map_rand:.3f}")

    domain = np.linspace(-3, 3, 200).reshape(-1, 1)
    w = posterior_weight(domain, gp, y_obs, sigma_obs)
    _, var = gp.predict(domain)
    H_seq = np.sum(w * var)
    print(f"IP-SUR design:  H = {H_seq:.6f}, MAP = {map_seq:.3f}")
    print(f"Uncertainty reduction ratio: {H_rand / (H_seq + 1e-30):.1f}x")
