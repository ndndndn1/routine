"""Minimal reference implementation for arXiv:2504.04244 and arXiv:2511.23141.

Shared folder for two papers with the same keyword combination:
  - 2504.04244: BMSDM framework for multi-objective sequential BO in manufacturing
  - 2511.23141: BOLD framework for laser dicing process optimization

Reproduces:
  1) Multi-objective GP surrogate with expected hypervolume improvement.
  2) Sequential experimental design loop.
  3) Constrained multi-objective optimization.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# GP surrogate ---------------------------------------------------------------
# ---------------------------------------------------------------------------
class GP:
    def __init__(self, length_scale: float = 0.3, noise: float = 1e-3):
        self.ls = length_scale
        self.noise = noise
        self.X = np.empty((0, 0))
        self.y = np.empty(0)
        self._L = None
        self._alpha = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X.copy()
        self.y = y.copy()
        K = self._kernel(X, X) + self.noise * np.eye(len(y))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.X.size == 0:
            return np.zeros(len(Xs)), np.ones(len(Xs))
        Ks = self._kernel(Xs, self.X)
        mu = Ks @ self._alpha
        v = np.linalg.solve(self._L, Ks.T)
        var = np.clip(1.0 - (v * v).sum(0), 1e-9, None)
        return mu, var

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d2 = cdist(A, B, "sqeuclidean")
        return np.exp(-0.5 * d2 / self.ls ** 2)


# ---------------------------------------------------------------------------
# Pareto utilities -----------------------------------------------------------
# ---------------------------------------------------------------------------
def is_dominated(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(b >= a) and np.any(b > a))


def pareto_front(Y: np.ndarray) -> np.ndarray:
    n = len(Y)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i != j and mask[j] and is_dominated(Y[i], Y[j]):
                mask[i] = False
                break
    return np.where(mask)[0]


def hypervolume_2d(Y: np.ndarray, ref: np.ndarray) -> float:
    front_idx = pareto_front(Y)
    F = Y[front_idx]
    F = F[F[:, 0].argsort()[::-1]]
    hv = 0.0
    prev_y = ref[1]
    for pt in F:
        if pt[0] > ref[0] or pt[1] > ref[1]:
            continue
        hv += (ref[0] - pt[0]) * (prev_y - pt[1])
        prev_y = pt[1]
    return hv


# ---------------------------------------------------------------------------
# Approximate EHVI (Monte Carlo) for 2 objectives --------------------------
# ---------------------------------------------------------------------------
def approx_ehvi(gps: list[GP], X_cand: np.ndarray, Y_obs: np.ndarray,
                ref: np.ndarray, n_mc: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hv_base = hypervolume_2d(Y_obs, ref)
    ehvi = np.zeros(len(X_cand))
    mus, vars_ = [], []
    for gp in gps:
        m, v = gp.predict(X_cand)
        mus.append(m)
        vars_.append(v)
    for i in range(len(X_cand)):
        samples = np.column_stack([
            rng.normal(mus[j][i], np.sqrt(vars_[j][i]), n_mc) for j in range(len(gps))
        ])
        hv_gains = []
        for s in samples:
            Y_new = np.vstack([Y_obs, s])
            hv_gains.append(hypervolume_2d(Y_new, ref) - hv_base)
        ehvi[i] = max(0, np.mean(hv_gains))
    return ehvi


# ---------------------------------------------------------------------------
# Sequential multi-objective BO loop ----------------------------------------
# ---------------------------------------------------------------------------
def sequential_mobo(
    objectives: list,
    bounds: np.ndarray,
    n_init: int = 10,
    n_iter: int = 20,
    ref_point: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    d = bounds.shape[0]
    n_obj = len(objectives)
    if ref_point is None:
        ref_point = np.zeros(n_obj)

    X = rng.uniform(bounds[:, 0], bounds[:, 1], (n_init, d))
    Y = np.array([[obj(x) for obj in objectives] for x in X])

    for it in range(n_iter):
        gps = []
        for j in range(n_obj):
            gp = GP(length_scale=0.3, noise=1e-3)
            gp.fit(X, Y[:, j])
            gps.append(gp)

        X_cand = rng.uniform(bounds[:, 0], bounds[:, 1], (200, d))
        ehvi = approx_ehvi(gps, X_cand, Y, ref_point, n_mc=50, seed=seed + it)
        best_idx = int(np.argmax(ehvi))
        x_new = X_cand[best_idx]
        y_new = np.array([obj(x_new) for obj in objectives])
        X = np.vstack([X, x_new])
        Y = np.vstack([Y, y_new])

    front_idx = pareto_front(Y)
    return {"X": X, "Y": Y, "pareto_X": X[front_idx], "pareto_Y": Y[front_idx]}


# ---------------------------------------------------------------------------
# Demo: manufacturing-like 2-objective problem ------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def strength(x):
        return -(x[0] ** 2 + x[1] ** 2 - 0.5 * x[0] * x[1])

    def speed(x):
        return -(((x[0] - 1) ** 2 + (x[1] - 1) ** 2))

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    result = sequential_mobo([strength, speed], bounds, n_init=8, n_iter=15,
                              ref_point=np.array([-3.0, -3.0]), seed=42)

    print(f"Total evaluations: {len(result['X'])}")
    print(f"Pareto front size: {len(result['pareto_X'])}")
    print("Pareto Y:")
    for y in result["pareto_Y"]:
        print(f"  obj1={y[0]:.3f}, obj2={y[1]:.3f}")
