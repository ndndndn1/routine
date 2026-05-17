"""Minimal reference implementation: Physics-Informed Bayesian Optimization
(arXiv:2503.00420).

Demonstrates a GP surrogate with physics-informed mean function, MESA
initialization, and sequential BO via Expected Improvement.

Run:
    python reference_impl.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Toy physics model (analytical prior) + expensive simulator -----------------
# ---------------------------------------------------------------------------
def physics_prior(x: np.ndarray) -> np.ndarray:
    """Cheap analytical physics model (e.g., magnetic circuit approximation).
    Captures the main trend but misses fine details."""
    return 2.0 * np.sin(x[:, 0]) * np.cos(0.5 * x[:, 1]) + x[:, 0] * 0.3


def expensive_simulator(x: np.ndarray) -> np.ndarray:
    """Expensive ground-truth (e.g., FEM simulation).
    physics_prior + nonlinear residual + noise."""
    base = physics_prior(x)
    residual = 0.5 * np.sin(3 * x[:, 0]) * np.cos(2 * x[:, 1])
    noise = 0.02 * np.random.default_rng(hash(x.tobytes()) % 2**31).standard_normal(len(x))
    return base + residual + noise


# ---------------------------------------------------------------------------
# Physics-Informed GP -------------------------------------------------------
# ---------------------------------------------------------------------------
class PhysicsInformedGP:
    def __init__(self, length_scale: float = 0.8, noise: float = 1e-4):
        self.ls = length_scale
        self.noise = noise

    def _kernel(self, X1, X2):
        sq = cdist(X1, X2, "sqeuclidean")
        return np.exp(-0.5 * sq / self.ls**2)

    def fit(self, X, y):
        self.X = X.copy()
        self.residuals = y - physics_prior(X)
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
        self.alpha = self.K_inv @ self.residuals

    def predict(self, Xs):
        m_phys = physics_prior(Xs)
        Ks = self._kernel(Xs, self.X)
        mu_residual = Ks @ self.alpha
        mu = m_phys + mu_residual

        v = self._kernel(Xs, Xs) - Ks @ self.K_inv @ Ks.T
        var = np.maximum(np.diag(v), 0.0)
        return mu, var


# ---------------------------------------------------------------------------
# Standard GP (no physics prior) for comparison ------------------------------
# ---------------------------------------------------------------------------
class StandardGP:
    def __init__(self, length_scale: float = 0.8, noise: float = 1e-4):
        self.ls = length_scale
        self.noise = noise

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
        var = np.maximum(np.diag(v), 0.0)
        return mu, var


# ---------------------------------------------------------------------------
# MESA: Maximum Entropy Sampling Algorithm -----------------------------------
# ---------------------------------------------------------------------------
def mesa_init(n_init: int, d: int, n_candidates: int = 500,
              length_scale: float = 0.8, bounds=(-3, 3), seed: int = 42):
    """Select initial points by maximizing GP prior entropy."""
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(bounds[0], bounds[1], (n_candidates, d))

    def kernel(X1, X2):
        sq = cdist(X1, X2, "sqeuclidean")
        return np.exp(-0.5 * sq / length_scale**2)

    selected = [rng.integers(n_candidates)]
    for _ in range(n_init - 1):
        X_sel = candidates[selected]
        K_sel = kernel(X_sel, X_sel) + 1e-6 * np.eye(len(selected))
        K_inv = np.linalg.inv(K_sel)
        best_score, best_idx = -np.inf, 0
        for j in range(n_candidates):
            if j in selected:
                continue
            xc = candidates[j:j+1]
            k_vec = kernel(xc, X_sel).flatten()
            cond_var = 1.0 - k_vec @ K_inv @ k_vec
            if cond_var > best_score:
                best_score = cond_var
                best_idx = j
        selected.append(best_idx)

    return candidates[selected]


# ---------------------------------------------------------------------------
# Expected Improvement -------------------------------------------------------
# ---------------------------------------------------------------------------
def expected_improvement(mu, sigma, f_best, xi=0.01):
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - f_best - xi) / (sigma + 1e-30)
    ei = (mu - f_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-10] = 0.0
    return ei


# ---------------------------------------------------------------------------
# BO loop -------------------------------------------------------------------
# ---------------------------------------------------------------------------
def run_bo(gp_class, label: str, n_init: int = 5, n_seq: int = 20,
           use_mesa: bool = False, seed: int = 42):
    d = 2
    bounds = (-3.0, 3.0)
    rng = np.random.default_rng(seed)

    if use_mesa:
        X_train = mesa_init(n_init, d, bounds=bounds, seed=seed)
    else:
        X_train = rng.uniform(bounds[0], bounds[1], (n_init, d))
    y_train = expensive_simulator(X_train)

    gp = gp_class()
    candidates = rng.uniform(bounds[0], bounds[1], (1000, d))

    for step in range(n_seq):
        gp.fit(X_train, y_train)
        mu, var = gp.predict(candidates)
        sigma = np.sqrt(var)
        f_best = y_train.max()
        ei = expected_improvement(mu, sigma, f_best)
        idx = np.argmax(ei)
        x_new = candidates[idx:idx+1]
        y_new = expensive_simulator(x_new)
        X_train = np.vstack([X_train, x_new])
        y_train = np.append(y_train, y_new)

    best_idx = np.argmax(y_train)
    return y_train[best_idx], X_train[best_idx], len(y_train)


if __name__ == "__main__":
    print("Physics-Informed Bayesian Optimization (PIBO)")
    print("=" * 50)

    n_trials = 5
    results = {}
    for label, gp_cls, mesa in [
        ("PIBO + MESA", PhysicsInformedGP, True),
        ("PIBO + LHS",  PhysicsInformedGP, False),
        ("Standard BO",  StandardGP, False),
    ]:
        bests = []
        for trial in range(n_trials):
            best_y, best_x, n_eval = run_bo(gp_cls, label, use_mesa=mesa,
                                             seed=42 + trial)
            bests.append(best_y)
        results[label] = bests
        print(f"\n[{label}]")
        print(f"  Best f (mean ± std): {np.mean(bests):.3f} ± {np.std(bests):.3f}")
        print(f"  Total evaluations: {n_eval}")

    print("\n--- Summary ---")
    for label, bests in results.items():
        print(f"  {label:20s}: {np.mean(bests):.3f} ± {np.std(bests):.3f}")
