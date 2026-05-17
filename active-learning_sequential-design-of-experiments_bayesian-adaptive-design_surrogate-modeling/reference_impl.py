"""
Reference implementation: Simulation Experiment Design for Calibration via AL
Based on arXiv:2407.18885 (Surer, 2024)

Core idea: use GP emulator + active learning criteria (posterior variance
minimization or KL-divergence reduction) to sequentially select simulation
inputs that improve calibration accuracy.

Minimal NumPy/SciPy demo on a 1D simulator calibration problem.
"""

import numpy as np
from scipy.linalg import cho_solve, cho_factor
from typing import Tuple


class GPEmulator:
    """Minimal GP emulator with RBF kernel."""

    def __init__(self, length_scale: float = 0.3, noise: float = 1e-4):
        self.ls = length_scale
        self.noise = noise
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self._L = None

    def _rbf(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-0.5 * sq / self.ls**2)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        K = self._rbf(X, X) + self.noise * np.eye(len(X))
        self._L = cho_factor(K)

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ks = self._rbf(self.X, X_new)
        Kss = self._rbf(X_new, X_new)
        alpha = cho_solve(self._L, self.y)
        mu = Ks.T @ alpha
        v = cho_solve(self._L, Ks)
        var = np.diag(Kss - Ks.T @ v)
        return mu, np.clip(var, 1e-10, None)


def simulator(x: float, theta: float) -> float:
    """Toy simulator: eta(x, theta) = sin(theta*x) + 0.3*theta*cos(x)."""
    return np.sin(theta * x) + 0.3 * theta * np.cos(x)


def calibration_posterior(
    gp: GPEmulator,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    theta_grid: np.ndarray,
    obs_noise: float = 0.1,
) -> np.ndarray:
    """Compute unnormalized log-posterior over theta grid."""
    log_post = np.zeros(len(theta_grid))
    for i, th in enumerate(theta_grid):
        X_pred = np.column_stack([x_obs, np.full(len(x_obs), th)])
        mu, var = gp.predict(X_pred)
        total_var = var + obs_noise**2
        log_post[i] = -0.5 * np.sum((y_obs - mu) ** 2 / total_var + np.log(total_var))
    log_post -= np.max(log_post)
    return np.exp(log_post) / np.exp(log_post).sum()


def al_criterion_posterior_variance(
    gp: GPEmulator,
    candidates: np.ndarray,
    theta_post: np.ndarray,
    theta_grid: np.ndarray,
) -> np.ndarray:
    """Criterion 1: expected posterior predictive variance."""
    scores = np.zeros(len(candidates))
    for i, (xc, thc) in enumerate(candidates):
        X_q = np.array([[xc, thc]])
        _, var = gp.predict(X_q)
        th_idx = np.argmin(np.abs(theta_grid - thc))
        scores[i] = var[0] * theta_post[th_idx]
    return scores


def run_sequential_calibration(
    n_init: int = 5,
    n_seq: int = 10,
    n_theta: int = 50,
    obs_noise: float = 0.1,
    seed: int = 42,
) -> dict:
    """Run sequential experiment design for simulator calibration."""
    rng = np.random.default_rng(seed)
    theta_true = 2.0
    theta_grid = np.linspace(0.5, 4.0, n_theta)

    x_obs = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    y_obs = np.array([simulator(x, theta_true) + rng.normal(0, obs_noise) for x in x_obs])

    X_init_x = rng.uniform(0, 5, n_init)
    X_init_th = rng.uniform(0.5, 4.0, n_init)
    X_train = np.column_stack([X_init_x, X_init_th])
    y_train = np.array([simulator(x, th) for x, th in X_train])

    gp = GPEmulator(length_scale=0.5, noise=1e-3)
    gp.fit(X_train, y_train)

    posterior_history = []
    designs_history = []

    for step in range(n_seq):
        post = calibration_posterior(gp, x_obs, y_obs, theta_grid, obs_noise)
        posterior_history.append(post.copy())

        x_cands = rng.uniform(0, 5, 50)
        th_cands = rng.choice(theta_grid, 50, p=post)
        candidates = np.column_stack([x_cands, th_cands])

        scores = al_criterion_posterior_variance(gp, candidates, post, theta_grid)
        best_idx = np.argmax(scores)
        x_new, th_new = candidates[best_idx]
        designs_history.append((x_new, th_new))

        y_new = simulator(x_new, th_new)
        X_train = np.vstack([X_train, [[x_new, th_new]]])
        y_train = np.append(y_train, y_new)
        gp.fit(X_train, y_train)

    final_post = calibration_posterior(gp, x_obs, y_obs, theta_grid, obs_noise)
    posterior_history.append(final_post)

    theta_map = theta_grid[np.argmax(final_post)]
    return {
        "theta_true": theta_true,
        "theta_map": theta_map,
        "theta_grid": theta_grid,
        "posterior_history": posterior_history,
        "designs": designs_history,
        "n_simulations": n_init + n_seq,
    }


if __name__ == "__main__":
    result = run_sequential_calibration(n_init=8, n_seq=15, seed=0)
    print(f"True theta: {result['theta_true']:.2f}")
    print(f"MAP estimate: {result['theta_map']:.2f}")
    print(f"Total simulations: {result['n_simulations']}")
    print(f"Posterior peak width (std): {np.sqrt(np.average((result['theta_grid'] - result['theta_map'])**2, weights=result['posterior_history'][-1])):.3f}")
