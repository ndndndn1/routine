"""Minimal reference implementation: Sequential Bayesian Design for Locally
Accurate Surrogates (SBD-LAS) for inverse problems (arXiv:2507.17713).

Demonstrates the core loop of building a locally accurate GP surrogate
via sequential Bayesian design, targeting high-posterior-probability regions
in a 1D Darcy-like inverse problem.

Run:
    python 2507.17713_reference_impl.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# GP surrogate
# ---------------------------------------------------------------------------
class GP:
    def __init__(self, ls: float = 0.3, noise: float = 1e-4):
        self.ls = ls
        self.noise = noise
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self._K_inv: np.ndarray | None = None

    def _k(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * cdist(A, B, "sqeuclidean") / self.ls ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X, self.y = X.copy(), y.copy()
        K = self._k(X, X) + self.noise * np.eye(len(y))
        self._K_inv = np.linalg.inv(K)

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Ks = self._k(Xs, self.X)
        mu = Ks @ self._K_inv @ self.y
        var = np.clip(
            1.0 - np.einsum("ij,jk,ik->i", Ks, self._K_inv, Ks), 1e-9, None
        )
        return mu, var

    def add_point(self, x_new: np.ndarray, y_new: float) -> None:
        X_aug = np.vstack([self.X, x_new.reshape(1, -1)])
        y_aug = np.append(self.y, y_new)
        self.fit(X_aug, y_aug)


# ---------------------------------------------------------------------------
# Synthetic forward model (1D Darcy-like)
# ---------------------------------------------------------------------------
def forward_model(u: np.ndarray) -> np.ndarray:
    return np.sin(3.0 * u) + 0.5 * u ** 2


# ---------------------------------------------------------------------------
# Unnormalised log-posterior
# ---------------------------------------------------------------------------
def log_posterior(
    u_grid: np.ndarray, y_obs: float, surrogate: GP, sigma_obs: float = 0.1
) -> np.ndarray:
    mu, _ = surrogate.predict(u_grid)
    log_lik = -0.5 * ((mu - y_obs) / sigma_obs) ** 2
    log_prior = -0.5 * u_grid.ravel() ** 2
    return log_lik + log_prior


# ---------------------------------------------------------------------------
# SBD-LAS acquisition: posterior-weighted predictive variance
# ---------------------------------------------------------------------------
def acquisition_sbd(
    u_grid: np.ndarray, surrogate: GP, y_obs: float, sigma_obs: float = 0.1
) -> np.ndarray:
    _, var = surrogate.predict(u_grid)
    log_post = log_posterior(u_grid, y_obs, surrogate, sigma_obs)
    post_weights = np.exp(log_post - log_post.max())
    post_weights /= post_weights.sum() + 1e-12
    return var * post_weights


# ---------------------------------------------------------------------------
# Main: sequential design loop
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(42)
    y_obs = 0.8

    u_grid = np.linspace(-2, 2, 200).reshape(-1, 1)

    X_init = rng.uniform(-2, 2, size=(4, 1))
    y_init = forward_model(X_init).ravel()

    surr = GP(ls=0.5)
    surr.fit(X_init, y_init)

    n_steps = 10
    print("SBD-LAS sequential design loop")
    print("=" * 55)
    for step in range(n_steps):
        acq = acquisition_sbd(u_grid, surr, y_obs)
        best_idx = np.argmax(acq)
        u_new = u_grid[best_idx].reshape(1, -1)
        y_new = forward_model(u_new).ravel()[0]

        surr.add_point(u_new, y_new)

        log_p = log_posterior(u_grid, y_obs, surr)
        pw = np.exp(log_p - log_p.max())
        pw /= pw.sum()
        post_var = np.sum(pw * (u_grid.ravel() - np.sum(pw * u_grid.ravel())) ** 2)

        print(
            f"  step {step + 1:2d} | u_new={u_new.item():+.3f} | "
            f"posterior_var={post_var:.4f} | n_train={len(surr.y)}"
        )

    print("\nFinal posterior variance:", f"{post_var:.4f}")
    print("Training points used:", len(surr.y))


if __name__ == "__main__":
    main()
