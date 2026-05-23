"""Minimal reference implementation: Bayesian-guided inverse design of
microstructures via MOGP + active learning (arXiv:2603.15917).

Demonstrates uncertainty-driven active learning with a multi-output GP
surrogate for inverse design under a severely limited oracle budget.

Run:
    python 2603.15917_reference_impl.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Multi-output GP surrogate (independent outputs sharing kernel)
# ---------------------------------------------------------------------------
class MultiOutputGP:
    def __init__(self, n_outputs: int = 2, ls: float = 0.4, noise: float = 1e-4):
        self.n_outputs = n_outputs
        self.ls = ls
        self.noise = noise
        self.X: np.ndarray | None = None
        self.Y: np.ndarray | None = None
        self._K_inv: np.ndarray | None = None

    def _k(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * cdist(A, B, "sqeuclidean") / self.ls ** 2)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X, self.Y = X.copy(), Y.copy()
        K = self._k(X, X) + self.noise * np.eye(len(X))
        self._K_inv = np.linalg.inv(K)

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Ks = self._k(Xs, self.X)
        W = Ks @ self._K_inv
        mu = W @ self.Y
        var_per_point = np.clip(
            1.0 - np.einsum("ij,jk,ik->i", Ks, self._K_inv, Ks), 1e-9, None
        )
        var = np.tile(var_per_point[:, None], (1, self.n_outputs))
        return mu, var

    def add_point(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        X_aug = np.vstack([self.X, x_new.reshape(1, -1)])
        Y_aug = np.vstack([self.Y, y_new.reshape(1, -1)])
        self.fit(X_aug, Y_aug)


# ---------------------------------------------------------------------------
# Synthetic oracle: microstructure descriptors -> constitutive params
# ---------------------------------------------------------------------------
def oracle(x: np.ndarray) -> np.ndarray:
    c1 = np.sin(2.0 * x[:, 0]) * np.cos(x[:, 1]) + 0.2 * x[:, 0]
    c2 = 0.5 * x[:, 0] ** 2 - np.sin(x[:, 1])
    return np.column_stack([c1, c2])


# ---------------------------------------------------------------------------
# Active learning loop
# ---------------------------------------------------------------------------
def active_learning_loop(
    candidate_pool: np.ndarray,
    n_initial: int = 5,
    n_budget: int = 15,
    seed: int = 42,
) -> MultiOutputGP:
    rng = np.random.default_rng(seed)
    n_candidates = len(candidate_pool)

    init_idx = rng.choice(n_candidates, n_initial, replace=False)
    X_train = candidate_pool[init_idx]
    Y_train = oracle(X_train)
    labelled = set(init_idx.tolist())

    gp = MultiOutputGP(n_outputs=2)
    gp.fit(X_train, Y_train)

    print("Active learning loop (uncertainty-driven)")
    print("=" * 55)

    for step in range(n_budget):
        _, var = gp.predict(candidate_pool)
        total_var = var.sum(axis=1)
        for idx in labelled:
            total_var[idx] = -1.0

        best_idx = int(np.argmax(total_var))
        labelled.add(best_idx)

        x_new = candidate_pool[best_idx]
        y_new = oracle(x_new.reshape(1, -1)).ravel()
        gp.add_point(x_new, y_new)

        _, var_all = gp.predict(candidate_pool)
        mean_var = var_all.mean()
        print(
            f"  step {step + 1:2d} | query_idx={best_idx:4d} | "
            f"mean_var={mean_var:.4f} | n_labelled={len(labelled)}"
        )

    return gp


# ---------------------------------------------------------------------------
# Inverse design
# ---------------------------------------------------------------------------
def inverse_design(
    gp: MultiOutputGP, candidate_pool: np.ndarray, target: np.ndarray
) -> tuple[int, np.ndarray]:
    mu, var = gp.predict(candidate_pool)
    dist = np.sum((mu - target) ** 2, axis=1)
    best_idx = int(np.argmin(dist))
    return best_idx, mu[best_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(0)
    candidate_pool = rng.uniform(-2, 2, size=(2000, 2))

    gp = active_learning_loop(candidate_pool, n_initial=5, n_budget=15)

    print(f"\nTotal oracle calls: {len(gp.X)} / {len(candidate_pool)} "
          f"({100 * len(gp.X) / len(candidate_pool):.1f}%)")

    target = np.array([0.5, -0.3])
    best_idx, pred = inverse_design(gp, candidate_pool, target)
    true_val = oracle(candidate_pool[best_idx].reshape(1, -1)).ravel()

    print(f"\nInverse design target: {target}")
    print(f"Surrogate prediction:  {pred}")
    print(f"Oracle ground truth:   {true_val}")
    print(f"Design vector:         {candidate_pool[best_idx]}")


if __name__ == "__main__":
    main()
