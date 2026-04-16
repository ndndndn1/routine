"""Minimal reference implementation of ALMAB-DC (arXiv:2603.21180).

Reproduces the event-driven asynchronous loop that unifies:
  1) a Gaussian-Process surrogate (posterior mean / variance),
  2) an active-learning acquisition (UCB, EI, Max-Variance, Thompson),
  3) a multi-armed-bandit (UCB1) allocator across K candidate regions,
  4) asynchronous parallel workers via threading.

This is a *teaching* reference: numerical constants and kernel are simple,
not production-grade. It demonstrates how variance-driven acquisition tames
the one-to-many ambiguity of inverse search by concentrating budget on
high-variance / high-reward regions.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Tiny GP surrogate with RBF kernel -----------------------------------------
# ---------------------------------------------------------------------------
class GP:
    def __init__(self, length_scale: float = 0.25, noise: float = 1e-3):
        self.ls = length_scale
        self.noise = noise
        self.X: np.ndarray = np.empty((0, 0))
        self.y: np.ndarray = np.empty((0,))
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        self._lock = threading.Lock()

    def _k(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / self.ls ** 2)

    def condition(self, x: np.ndarray, y: float) -> None:
        with self._lock:
            x = x.reshape(1, -1)
            if self.X.size == 0:
                self.X = x
                self.y = np.array([y])
            else:
                self.X = np.vstack([self.X, x])
                self.y = np.append(self.y, y)
            K = self._k(self.X, self.X) + self.noise * np.eye(len(self.y))
            self._L = np.linalg.cholesky(K)
            self._alpha = np.linalg.solve(
                self._L.T, np.linalg.solve(self._L, self.y)
            )

    def posterior(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self.X.size == 0:
                mu = np.zeros(len(Xs))
                var = np.ones(len(Xs))
                return mu, var
            Ks = self._k(Xs, self.X)
            Kss = np.ones(len(Xs))  # k(x,x)=1 for RBF
            mu = Ks @ self._alpha
            v = np.linalg.solve(self._L, Ks.T)
            var = np.clip(Kss - (v * v).sum(0), 1e-9, None)
            return mu, var


# ---------------------------------------------------------------------------
# Acquisition functions ------------------------------------------------------
# ---------------------------------------------------------------------------
def acq_ucb(mu, var, beta=2.0, **_):
    return mu + beta * np.sqrt(var)


def acq_max_variance(mu, var, **_):
    return var


def acq_ei(mu, var, f_best=0.0, **_):
    sigma = np.sqrt(var)
    imp = mu - f_best
    z = imp / (sigma + 1e-9)
    from scipy.stats import norm  # local import keeps top clean
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def acq_thompson(mu, var, rng: np.random.Generator | None = None, **_):
    rng = rng or np.random.default_rng()
    return rng.normal(mu, np.sqrt(var))


ACQ = {"ucb": acq_ucb, "mv": acq_max_variance, "ei": acq_ei, "ts": acq_thompson}


# ---------------------------------------------------------------------------
# UCB1 bandit over K regions ------------------------------------------------
# ---------------------------------------------------------------------------
@dataclass
class UCB1:
    k: int
    counts: np.ndarray = field(init=False)
    means: np.ndarray = field(init=False)
    t: int = 0

    def __post_init__(self):
        self.counts = np.zeros(self.k)
        self.means = np.zeros(self.k)

    def select(self) -> int:
        self.t += 1
        # initial round: play each arm once
        for i in range(self.k):
            if self.counts[i] == 0:
                return i
        ucb = self.means + np.sqrt(2 * math.log(self.t) / self.counts)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] += (reward - self.means[arm]) / n


# ---------------------------------------------------------------------------
# ALMAB-DC main loop ---------------------------------------------------------
# ---------------------------------------------------------------------------
def almab_dc(
    objective: Callable[[np.ndarray], float],
    regions: list[tuple[np.ndarray, np.ndarray]],
    budget: int = 40,
    n_workers: int = 4,
    acquisition: str = "ucb",
    eval_latency: Callable[[], float] = lambda: 0.02,
    seed: int = 0,
) -> dict:
    """Asynchronous GP-bandit sequential experimental design.

    Args:
        objective:    black-box f(x) -> float, maximized.
        regions:      list of K (lo, hi) boxes acting as bandit arms.
        budget:       total evaluations.
        n_workers:    parallel asynchronous evaluators.
        acquisition:  one of {"ucb","mv","ei","ts"}.
    """
    rng = np.random.default_rng(seed)
    gp = GP()
    mab = UCB1(k=len(regions))
    af = ACQ[acquisition]

    results_q: "queue.Queue[tuple[int, np.ndarray, float]]" = queue.Queue()
    pending = 0
    history: list[tuple[np.ndarray, float, int]] = []

    def worker(worker_id: int, x: np.ndarray) -> None:
        time.sleep(eval_latency())  # simulate expensive evaluation
        y = objective(x)
        results_q.put((worker_id, x, y))

    def propose(arm: int) -> np.ndarray:
        lo, hi = regions[arm]
        cand = rng.uniform(lo, hi, size=(256, len(lo)))
        mu, var = gp.posterior(cand)
        f_best = float(gp.y.max()) if gp.y.size else 0.0
        score = af(mu, var, beta=2.0, f_best=f_best, rng=rng)
        return cand[int(np.argmax(score))]

    # prime workers
    for wid in range(n_workers):
        arm = mab.select()
        x = propose(arm)
        threading.Thread(
            target=worker, args=(wid, x), daemon=True
        ).start()
        pending += 1
        history.append((x, float("nan"), arm))

    done = 0
    while done < budget:
        wid, x, y = results_q.get()
        gp.condition(x, y)
        # reward ← normalized y improvement
        f_best = float(gp.y.max())
        arm_done = history[-pending][2] if pending else 0
        mab.update(arm_done, reward=y / (abs(f_best) + 1e-9))
        done += 1
        pending -= 1
        if done + pending < budget:
            arm = mab.select()
            x_next = propose(arm)
            threading.Thread(
                target=worker, args=(wid, x_next), daemon=True
            ).start()
            pending += 1
            history.append((x_next, float("nan"), arm))

    return {
        "best_x": gp.X[int(np.argmax(gp.y))],
        "best_y": float(gp.y.max()),
        "history_X": gp.X.copy(),
        "history_y": gp.y.copy(),
        "mab_counts": mab.counts.copy(),
    }


# ---------------------------------------------------------------------------
# Demo: inverse recovery of a latent mode ----------------------------------
# ---------------------------------------------------------------------------
def _branin(x: np.ndarray) -> float:
    a, b, c = 1.0, 5.1 / (4 * math.pi ** 2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
    x1, x2 = x[0], x[1]
    val = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s
    return -val  # maximize


if __name__ == "__main__":
    # Branin has 3 global minima — a canonical one-to-many / multimodal
    # inverse-design toy. Variance-driven AL should spread budget to all three
    # modes while UCB1 re-allocates to the most promising region.
    regions = [
        (np.array([-5.0, 0.0]), np.array([0.0, 15.0])),
        (np.array([0.0, 0.0]), np.array([5.0, 15.0])),
        (np.array([5.0, 0.0]), np.array([10.0, 15.0])),
    ]
    out = almab_dc(_branin, regions, budget=30, n_workers=3, acquisition="ucb")
    print("best_x =", out["best_x"])
    print("best_y =", out["best_y"])
    print("arm allocation =", out["mab_counts"])
