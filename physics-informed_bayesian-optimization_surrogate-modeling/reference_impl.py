"""
Reference implementation: Physics-Informed Uncertainty for AI-driven Design
Based on arXiv:2601.18638 (Xue et al., 2026)

Core idea: use the degree to which a surrogate's prediction violates
physical laws as a cheap proxy for predictive uncertainty, then integrate
this Physics-Informed Uncertainty (PIU) into BO for reliable inverse design.

Minimal NumPy/SciPy demo on a toy electromagnetic filter design problem.
"""

import numpy as np
from scipy.linalg import cho_solve, cho_factor
from typing import Tuple


def forward_simulator(params: np.ndarray) -> np.ndarray:
    """Toy EM filter: maps 3 design params to transmission spectrum (10 freqs).

    Mimics a frequency-selective surface with resonance peaks.
    """
    f = np.linspace(20, 30, 10)
    w1, w2, d = params[0], params[1], params[2]
    S21 = (
        0.8 * np.exp(-((f - 22 * w1) ** 2) / (2 * d**2))
        + 0.6 * np.exp(-((f - 27 * w2) ** 2) / (2 * d**2))
    )
    return np.clip(S21, 0, 1)


def physics_constraint(params: np.ndarray, spectrum: np.ndarray) -> float:
    """Physics constraint: energy conservation + causality proxy.

    Returns violation magnitude (0 = perfectly physical).
    """
    energy_violation = max(0, np.sum(spectrum) - 8.0)
    monotonicity_violation = np.sum(np.maximum(0, np.diff(spectrum, n=2)))
    passivity = np.sum(np.maximum(0, spectrum - 1.0))
    return energy_violation + 0.5 * monotonicity_violation + 10.0 * passivity


class Surrogate:
    """DNN-like surrogate (implemented as GP for simplicity)."""

    def __init__(self, length_scale: float = 0.5, noise: float = 1e-3):
        self.ls = length_scale
        self.noise = noise
        self.X = None
        self.Y = None

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-0.5 * sq / self.ls**2)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        self._L = cho_factor(K)

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ks = self._kernel(self.X, X_new)
        alpha = cho_solve(self._L, self.Y)
        mu = Ks.T @ alpha
        return mu, None


def physics_informed_uncertainty(
    surrogate: Surrogate,
    params: np.ndarray,
) -> float:
    """PIU: physics violation of surrogate prediction as uncertainty proxy."""
    spectrum_pred, _ = surrogate.predict(params.reshape(1, -1))
    return physics_constraint(params, spectrum_pred[0])


def target_spectrum() -> np.ndarray:
    """Target: bandpass at 24-26 GHz."""
    f = np.linspace(20, 30, 10)
    return np.exp(-((f - 25) ** 2) / (2 * 1.0**2))


def design_objective(params: np.ndarray, surrogate: Surrogate) -> float:
    """Negative MSE between predicted and target spectrum."""
    pred, _ = surrogate.predict(params.reshape(1, -1))
    tgt = target_spectrum()
    return -np.mean((pred[0] - tgt) ** 2)


def run_piu_bo(
    n_init: int = 15,
    n_iter: int = 30,
    n_candidates: int = 200,
    piu_threshold: float = 0.5,
    seed: int = 42,
) -> dict:
    """BO with Physics-Informed Uncertainty filtering."""
    rng = np.random.default_rng(seed)
    bounds = np.array([[0.8, 1.2], [0.8, 1.2], [0.5, 2.0]])

    X_init = rng.uniform(bounds[:, 0], bounds[:, 1], (n_init, 3))
    Y_init = np.array([forward_simulator(x) for x in X_init])

    surrogate = Surrogate(length_scale=0.5, noise=1e-3)
    surrogate.fit(X_init, Y_init)

    X_all = X_init.copy()
    Y_all = Y_init.copy()
    best_history = []
    piu_history = []
    n_filtered = 0

    for it in range(n_iter):
        candidates = rng.uniform(bounds[:, 0], bounds[:, 1], (n_candidates, 3))

        piu_scores = np.array([
            physics_informed_uncertainty(surrogate, c) for c in candidates
        ])
        valid_mask = piu_scores < piu_threshold
        n_filtered += np.sum(~valid_mask)

        if not np.any(valid_mask):
            valid_mask = piu_scores < np.percentile(piu_scores, 30)

        valid_candidates = candidates[valid_mask]
        objectives = np.array([
            design_objective(c, surrogate) for c in valid_candidates
        ])
        best_idx = np.argmax(objectives)
        x_new = valid_candidates[best_idx]

        y_new = forward_simulator(x_new)
        X_all = np.vstack([X_all, x_new.reshape(1, -1)])
        Y_all = np.vstack([Y_all, y_new.reshape(1, -1)])
        surrogate.fit(X_all, Y_all)

        tgt = target_spectrum()
        mse = np.mean((y_new - tgt) ** 2)
        best_history.append(mse)
        piu_history.append(float(piu_scores[valid_mask][best_idx]))

    best_idx = np.argmin(best_history)
    return {
        "best_params": X_all[n_init + best_idx],
        "best_mse": best_history[best_idx],
        "mse_history": best_history,
        "piu_history": piu_history,
        "n_filtered": n_filtered,
        "total_evals": n_init + n_iter,
    }


if __name__ == "__main__":
    result = run_piu_bo(n_init=15, n_iter=25, seed=0)
    print(f"Best MSE: {result['best_mse']:.6f}")
    print(f"Best params: {np.round(result['best_params'], 3)}")
    print(f"Candidates filtered by PIU: {result['n_filtered']}")
    print(f"Total evaluations: {result['total_evals']}")

    pred = forward_simulator(result["best_params"])
    tgt = target_spectrum()
    print(f"Spectrum match (correlation): {np.corrcoef(pred, tgt)[0,1]:.4f}")
