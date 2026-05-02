"""Minimal reference implementation for arXiv:2409.09141.

Reproduces:
  1) Derivative-informed dimension reduction for sensitivity-aware latent space.
  2) GP surrogate as neural operator proxy.
  3) EIG-based sequential experimental design.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Simple forward model with Jacobian ----------------------------------------
# ---------------------------------------------------------------------------
def advection_diffusion_forward(theta: np.ndarray, xi: float,
                                 n_grid: int = 20) -> np.ndarray:
    dx = 1.0 / (n_grid + 1)
    kappa = 0.01
    v = theta[0]
    source_loc = int(np.clip(theta[1] * n_grid, 0, n_grid - 1))
    source_amp = theta[2] if len(theta) > 2 else 1.0
    f = np.zeros(n_grid)
    f[source_loc] = source_amp / dx

    A = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        A[i, i] = 2 * kappa / dx ** 2
        if i > 0:
            A[i, i - 1] = -kappa / dx ** 2 - v / (2 * dx)
        if i < n_grid - 1:
            A[i, i + 1] = -kappa / dx ** 2 + v / (2 * dx)
    u = np.linalg.solve(A + 1e-6 * np.eye(n_grid), f)
    obs_loc = int(np.clip(xi * n_grid, 0, n_grid - 1))
    return u[max(0, obs_loc - 1):min(n_grid, obs_loc + 2)]


def numerical_jacobian(theta: np.ndarray, xi: float, delta: float = 1e-5) -> np.ndarray:
    y0 = advection_diffusion_forward(theta, xi)
    J = np.zeros((len(y0), len(theta)))
    for i in range(len(theta)):
        tp = theta.copy()
        tp[i] += delta
        J[:, i] = (advection_diffusion_forward(tp, xi) - y0) / delta
    return J


# ---------------------------------------------------------------------------
# Derivative-informed dimension reduction ------------------------------------
# ---------------------------------------------------------------------------
def derivative_informed_subspace(thetas: np.ndarray, xi: float,
                                  n_components: int = 2) -> np.ndarray:
    J_sum = None
    for theta in thetas:
        J = numerical_jacobian(theta, xi)
        JtJ = J.T @ J
        if J_sum is None:
            J_sum = JtJ
        else:
            J_sum += JtJ
    J_sum /= len(thetas)
    eigvals, eigvecs = np.linalg.eigh(J_sum)
    idx = np.argsort(eigvals)[::-1][:n_components]
    return eigvecs[:, idx]


# ---------------------------------------------------------------------------
# GP surrogate ---------------------------------------------------------------
# ---------------------------------------------------------------------------
class GP:
    def __init__(self, ls=0.4, noise=1e-3):
        self.ls, self.noise = ls, noise
        self.X = np.empty((0, 0))
        self.y = np.empty(0)

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
# EIG-based sequential design ------------------------------------------------
# ---------------------------------------------------------------------------
def approximate_eig(gp: GP, theta_samples: np.ndarray, xi: float,
                    sigma_obs: float = 0.1) -> float:
    mu, var = gp.predict(theta_samples)
    log_lik_var = var + sigma_obs ** 2
    eig = 0.5 * np.mean(np.log(log_lik_var + 1e-12) - np.log(sigma_obs ** 2))
    return float(eig)


def sequential_boed(forward_model, bounds: np.ndarray, xi_candidates: np.ndarray,
                    n_init: int = 10, n_design: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = bounds.shape[0]

    X_train = rng.uniform(bounds[:, 0], bounds[:, 1], (n_init, d))
    xi_history = []

    for step in range(n_design):
        theta_pool = rng.uniform(bounds[:, 0], bounds[:, 1], (50, d))
        best_eig, best_xi = -np.inf, xi_candidates[0]

        for xi in xi_candidates:
            y_train = np.array([forward_model(x, xi).mean() for x in X_train])
            gp = GP(ls=0.5)
            gp.fit(X_train, y_train)
            eig = approximate_eig(gp, theta_pool, xi)
            if eig > best_eig:
                best_eig, best_xi = eig, xi

        xi_history.append(best_xi)
        x_new = rng.uniform(bounds[:, 0], bounds[:, 1], (3, d))
        X_train = np.vstack([X_train, x_new])

    return {"xi_history": xi_history, "n_experiments": len(xi_history)}


# ---------------------------------------------------------------------------
# Demo -----------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    d = 3
    bounds = np.array([[0.1, 1.0], [0.1, 0.9], [0.5, 2.0]])
    xi_candidates = np.linspace(0.1, 0.9, 9)

    result = sequential_boed(advection_diffusion_forward, bounds, xi_candidates,
                              n_init=15, n_design=5, seed=42)
    print(f"Selected experiment locations: {result['xi_history']}")

    rng = np.random.default_rng(7)
    thetas = rng.uniform(bounds[:, 0], bounds[:, 1], (20, d))
    W = derivative_informed_subspace(thetas, xi=0.5, n_components=2)
    print(f"\nDerivative-informed subspace (top-2 eigenvectors):")
    for i in range(d):
        print(f"  theta_{i}: [{W[i, 0]:.3f}, {W[i, 1]:.3f}]")
    print(f"Sensitivity ranking: {np.argsort(np.abs(W[:, 0]))[::-1]}")
