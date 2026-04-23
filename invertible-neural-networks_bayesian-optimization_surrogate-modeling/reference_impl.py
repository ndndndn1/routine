"""
Reference implementation: Latent Bayesian Optimization via Normalizing Flows
Based on arXiv:2504.14889 (ICLR 2025)

Core idea: use an invertible normalizing flow to map structured input space
to a latent space with zero reconstruction gap, then run GP-based BO in
the latent space.

Minimal NumPy demo with a toy sequence optimization problem.
"""

import numpy as np
from scipy.linalg import cho_solve, cho_factor
from typing import Tuple, List


class ToyAffineFlow:
    """Minimal 1D affine coupling flow (invertible) for demonstration.

    Maps x in R^d <-> z in R^d with exact invertibility.
    """

    def __init__(self, dim: int, n_layers: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.scales = [rng.normal(0, 0.3, dim) for _ in range(n_layers)]
        self.shifts = [rng.normal(0, 0.3, dim) for _ in range(n_layers)]
        self.masks = [np.arange(dim) % 2 == (i % 2) for i in range(n_layers)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x.copy()
        for s, t, m in zip(self.scales, self.shifts, self.masks):
            z_masked = z * m
            scale = np.exp(s * z_masked)
            shift = t * z_masked
            z = z * (~m) * scale + z * (~m) * 0 + z_masked
            z = np.where(m, z, z * scale + shift)
        return z

    def inverse(self, z: np.ndarray) -> np.ndarray:
        x = z.copy()
        for s, t, m in reversed(list(zip(self.scales, self.shifts, self.masks))):
            x_masked = x * m
            scale = np.exp(s * x_masked)
            shift = t * x_masked
            x = np.where(m, x, (x - shift) / scale)
        return x


class GPSurrogate:
    """GP surrogate for BO in latent space."""

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-3):
        self.ls = length_scale
        self.noise = noise
        self.Z = None
        self.y = None

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-0.5 * sq / self.ls**2)

    def fit(self, Z: np.ndarray, y: np.ndarray):
        self.Z = Z
        self.y = y
        K = self._kernel(Z, Z) + self.noise * np.eye(len(Z))
        self._L = cho_factor(K)

    def predict(self, Z_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ks = self._kernel(self.Z, Z_new)
        Kss_diag = np.ones(len(Z_new))
        alpha = cho_solve(self._L, self.y)
        mu = Ks.T @ alpha
        v = cho_solve(self._L, Ks)
        var = Kss_diag - np.einsum("ij,ij->j", Ks, v)
        return mu, np.clip(var, 1e-10, None)

    def ucb(self, Z_new: np.ndarray, beta: float = 2.0) -> np.ndarray:
        mu, var = self.predict(Z_new)
        return mu + beta * np.sqrt(var)


def toy_objective(x: np.ndarray) -> float:
    """Multi-modal toy objective in original space."""
    return -np.sum((x - 0.5) ** 2) + 0.5 * np.sin(5 * np.pi * np.sum(x))


def run_nf_bo(
    dim: int = 4,
    n_init: int = 10,
    n_iter: int = 30,
    n_candidates: int = 200,
    seed: int = 42,
) -> dict:
    """Run NF-BO: Bayesian optimization in normalizing flow latent space."""
    rng = np.random.default_rng(seed)
    flow = ToyAffineFlow(dim, n_layers=3, seed=seed)
    gp = GPSurrogate(length_scale=1.0, noise=1e-3)

    X_init = rng.uniform(0, 1, (n_init, dim))
    y_init = np.array([toy_objective(x) for x in X_init])

    X_all = X_init.copy()
    y_all = y_init.copy()
    best_history = [np.max(y_all)]

    for it in range(n_iter):
        Z_all = np.array([flow.forward(x) for x in X_all])
        gp.fit(Z_all, y_all)

        Z_cand = rng.normal(0, 1, (n_candidates, dim))
        acq = gp.ucb(Z_cand, beta=2.0)
        best_idx = np.argmax(acq)
        z_new = Z_cand[best_idx]

        x_new = flow.inverse(z_new)
        y_new = toy_objective(x_new)

        X_all = np.vstack([X_all, x_new.reshape(1, -1)])
        y_all = np.append(y_all, y_new)
        best_history.append(np.max(y_all))

    best_idx = np.argmax(y_all)
    return {
        "best_x": X_all[best_idx],
        "best_y": y_all[best_idx],
        "best_history": best_history,
        "n_evals": n_init + n_iter,
    }


if __name__ == "__main__":
    result = run_nf_bo(dim=4, n_init=10, n_iter=25, seed=0)
    print(f"Best objective: {result['best_y']:.4f}")
    print(f"Best x: {np.round(result['best_x'], 3)}")
    print(f"Total evaluations: {result['n_evals']}")
    print(f"Improvement trajectory: {[f'{v:.3f}' for v in result['best_history'][::5]]}")
