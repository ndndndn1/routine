"""Minimal reference implementation of ALABI (arXiv:2603.18259).

ALABI accelerates Bayesian inference when the log-posterior is expensive
by learning a GP surrogate over log-posterior values and picking the next
query via the BAPE acquisition, which targets points that simultaneously
(a) have high surrogate posterior mass and (b) have large surrogate variance.

The official package (ALABI) plugs into emcee / dynesty / multinest /
ultranest. This reference script keeps scope tight: GP + BAPE sequential
loop + a tiny slice sampler on the trained surrogate, to illustrate how
variance-driven sequential design tames a bimodal inverse problem.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Lightweight GP over log-posterior ----------------------------------------
# ---------------------------------------------------------------------------
@dataclass
class GP:
    length_scale: float = 0.5
    signal_var: float = 1.0
    noise: float = 1e-4

    def __post_init__(self):
        self.X = np.empty((0, 0))
        self.y = np.empty((0,))
        self._L = None
        self._alpha = None

    def _kernel(self, A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return self.signal_var * np.exp(-0.5 * d2 / self.length_scale ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X, self.y = X, y
        K = self._kernel(X, X) + self.noise * np.eye(len(y))
        self._L = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Ks = self._kernel(Xs, self.X)
        mu = Ks @ self._alpha
        v = np.linalg.solve(self._L, Ks.T)
        var = np.clip(
            self.signal_var - (v * v).sum(0), 1e-9, None
        )
        return mu, var


# ---------------------------------------------------------------------------
# BAPE acquisition ----------------------------------------------------------
# ---------------------------------------------------------------------------
def bape(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Bayesian Active Posterior Estimation score.

    BAPE(θ) = exp(2 μ + σ²) · (exp(σ²) - 1)
    """
    return np.exp(2.0 * mu + var) * (np.exp(var) - 1.0)


# ---------------------------------------------------------------------------
# ALABI sequential-design loop ---------------------------------------------
# ---------------------------------------------------------------------------
def alabi(
    log_posterior: Callable[[np.ndarray], float],
    bounds: np.ndarray,                 # shape (d, 2)
    n_init: int = 16,
    n_iter: int = 40,
    n_cand: int = 4096,
    seed: int = 0,
) -> GP:
    rng = np.random.default_rng(seed)
    d = bounds.shape[0]

    def sample_box(n):
        lo, hi = bounds[:, 0], bounds[:, 1]
        return rng.uniform(lo, hi, size=(n, d))

    X = sample_box(n_init)
    y = np.array([log_posterior(x) for x in X])

    gp = GP(length_scale=0.3 * np.mean(bounds[:, 1] - bounds[:, 0]))
    gp.fit(X, y)

    for t in range(n_iter):
        cand = sample_box(n_cand)
        mu, var = gp.predict(cand)
        # BAPE can overflow — shift mu by max for numerical safety
        mu_shift = mu - mu.max()
        score = np.exp(2.0 * mu_shift + var) * (np.exp(var) - 1.0)
        x_next = cand[int(np.argmax(score))]
        y_next = log_posterior(x_next)
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        gp.fit(X, y)

    return gp


# ---------------------------------------------------------------------------
# Slice-sample the trained surrogate to recover posterior ------------------
# ---------------------------------------------------------------------------
def sample_surrogate(gp: GP, bounds, n_samples=2000, seed=1):
    rng = np.random.default_rng(seed)
    d = bounds.shape[0]
    x = rng.uniform(bounds[:, 0], bounds[:, 1])
    logp_x = gp.predict(x.reshape(1, -1))[0][0]
    chain = np.zeros((n_samples, d))
    for i in range(n_samples):
        prop = x + 0.1 * rng.standard_normal(d) * (bounds[:, 1] - bounds[:, 0])
        prop = np.clip(prop, bounds[:, 0], bounds[:, 1])
        logp_prop = gp.predict(prop.reshape(1, -1))[0][0]
        if math.log(rng.uniform()) < logp_prop - logp_x:
            x, logp_x = prop, logp_prop
        chain[i] = x
    return chain


# ---------------------------------------------------------------------------
# Demo: bimodal inverse problem --------------------------------------------
# ---------------------------------------------------------------------------
def _bimodal_logpost(theta: np.ndarray) -> float:
    """log-posterior with two well-separated modes.

    This mimics a one-to-many inverse problem: two distinct parameter
    settings explain the observation equally well.
    """
    m1 = np.array([-1.5, -1.5])
    m2 = np.array([+1.5, +1.5])
    s2 = 0.35 ** 2
    p = (
        math.exp(-0.5 * ((theta - m1) ** 2).sum() / s2)
        + math.exp(-0.5 * ((theta - m2) ** 2).sum() / s2)
    )
    return math.log(p + 1e-300)


if __name__ == "__main__":
    bounds = np.array([[-3.0, 3.0], [-3.0, 3.0]])
    gp = alabi(_bimodal_logpost, bounds, n_init=20, n_iter=60)

    chain = sample_surrogate(gp, bounds, n_samples=4000)
    mean = chain.mean(0)
    # report whether both modes were visited
    near_m1 = ((chain - np.array([-1.5, -1.5])) ** 2).sum(1) < 0.5
    near_m2 = ((chain - np.array([+1.5, +1.5])) ** 2).sum(1) < 0.5
    print(f"chain mean = {mean}")
    print(f"fraction near mode 1 = {near_m1.mean():.3f}")
    print(f"fraction near mode 2 = {near_m2.mean():.3f}")
    print(f"surrogate training evaluations = {len(gp.y)}")
