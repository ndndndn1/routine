"""Minimal reference implementation for arXiv:2512.13354.

Reproduces the core pipeline:
  1) Gradient-boosted surrogate for a CVD-like process.
  2) Sobol global sensitivity analysis via surrogate.
  3) Approximate Bayesian Computation (ABC-SMC) for inverse UQ.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Simple gradient-boosted stump ensemble (surrogate) -------------------------
# ---------------------------------------------------------------------------
class GBTSurrogate:
    def __init__(self, n_estimators: int = 50, lr: float = 0.1, seed: int = 0):
        self.n_est = n_estimators
        self.lr = lr
        self.seed = seed
        self.stumps: list[tuple[int, float, float, float]] = []
        self.intercept = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.intercept = float(np.mean(y))
        residual = y - self.intercept
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_est):
            feat = rng.integers(X.shape[1])
            thresh = float(np.median(X[:, feat]) + rng.normal() * np.std(X[:, feat]) * 0.5)
            left = residual[X[:, feat] <= thresh]
            right = residual[X[:, feat] > thresh]
            val_l = float(np.mean(left)) if len(left) > 0 else 0.0
            val_r = float(np.mean(right)) if len(right) > 0 else 0.0
            self.stumps.append((feat, thresh, val_l, val_r))
            pred = np.where(X[:, feat] <= thresh, val_l, val_r)
            residual -= self.lr * pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = np.full(len(X), self.intercept)
        for feat, thresh, vl, vr in self.stumps:
            y += self.lr * np.where(X[:, feat] <= thresh, vl, vr)
        return y


# ---------------------------------------------------------------------------
# Sobol sensitivity (first-order, Monte Carlo) --------------------------------
# ---------------------------------------------------------------------------
def sobol_first_order(model: GBTSurrogate, bounds: np.ndarray,
                      n_samples: int = 4000, seed: int = 0) -> np.ndarray:
    d = bounds.shape[0]
    rng = np.random.default_rng(seed)
    A = rng.uniform(bounds[:, 0], bounds[:, 1], (n_samples, d))
    B = rng.uniform(bounds[:, 0], bounds[:, 1], (n_samples, d))
    fA = model.predict(A)
    fB = model.predict(B)
    var_total = np.var(np.concatenate([fA, fB]))
    S = np.zeros(d)
    for i in range(d):
        AB_i = B.copy()
        AB_i[:, i] = A[:, i]
        fAB_i = model.predict(AB_i)
        S[i] = np.mean(fB * (fAB_i - fA)) / (var_total + 1e-12)
    return np.clip(S, 0, 1)


# ---------------------------------------------------------------------------
# ABC-SMC for inverse UQ ---------------------------------------------------
# ---------------------------------------------------------------------------
def abc_smc(model: GBTSurrogate, y_obs: float, bounds: np.ndarray,
            n_particles: int = 500, n_rounds: int = 5,
            epsilon_schedule: list[float] | None = None,
            seed: int = 0) -> np.ndarray:
    d = bounds.shape[0]
    rng = np.random.default_rng(seed)
    if epsilon_schedule is None:
        epsilon_schedule = [2.0, 1.0, 0.5, 0.25, 0.1]
    particles = rng.uniform(bounds[:, 0], bounds[:, 1], (n_particles, d))
    weights = np.ones(n_particles) / n_particles

    for rnd in range(min(n_rounds, len(epsilon_schedule))):
        eps = epsilon_schedule[rnd]
        new_particles = []
        while len(new_particles) < n_particles:
            idx = rng.choice(n_particles, p=weights)
            proposal = particles[idx] + rng.normal(0, 0.05, d) * (bounds[:, 1] - bounds[:, 0])
            proposal = np.clip(proposal, bounds[:, 0], bounds[:, 1])
            y_sim = model.predict(proposal.reshape(1, -1))[0]
            if abs(y_sim - y_obs) < eps:
                new_particles.append(proposal)
        particles = np.array(new_particles)
        weights = np.ones(n_particles) / n_particles
    return particles


# ---------------------------------------------------------------------------
# Demo: synthetic CVD-like process ------------------------------------------
# ---------------------------------------------------------------------------
def _cvd_simulator(x: np.ndarray) -> float:
    temp, flow, pressure = x[0], x[1], x[2]
    thickness = 0.5 * temp + 0.3 * flow - 0.2 * pressure + 0.1 * temp * flow / 100
    return thickness + np.random.normal(0, 0.05)


if __name__ == "__main__":
    bounds = np.array([[300, 600], [50, 200], [1, 10]], dtype=float)
    param_names = ["temperature", "flow_rate", "pressure"]

    rng = np.random.default_rng(42)
    n_train = 200
    X_train = rng.uniform(bounds[:, 0], bounds[:, 1], (n_train, 3))
    y_train = np.array([_cvd_simulator(x) for x in X_train])

    model = GBTSurrogate(n_estimators=80, lr=0.1, seed=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    r2 = 1 - np.sum((y_train - y_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    print(f"Surrogate R²: {r2:.4f}")

    S = sobol_first_order(model, bounds)
    print("\nSobol first-order indices:")
    for name, si in zip(param_names, S):
        print(f"  {name}: {si:.4f}")

    y_target = 250.0
    posterior = abc_smc(model, y_target, bounds, n_particles=300, seed=1)
    print(f"\nABC-SMC posterior (target coating={y_target}):")
    for i, name in enumerate(param_names):
        print(f"  {name}: mean={posterior[:, i].mean():.1f}, "
              f"std={posterior[:, i].std():.1f}, "
              f"95%CI=[{np.percentile(posterior[:, i], 2.5):.1f}, "
              f"{np.percentile(posterior[:, i], 97.5):.1f}]")
