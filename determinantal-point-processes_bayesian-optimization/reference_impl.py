"""
Reference implementation: PDBO — Pareto-front Diverse Batch Optimization
(arXiv:2406.08799)

Core idea: DPP-based diverse batch selection for multi-objective Bayesian
optimization. The L-ensemble DPP kernel combines acquisition quality scores
with GP covariance to select batches that are high-quality AND diverse.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


# ---------------------------------------------------------------------------
# Gaussian Process surrogate (minimal, single-objective)
# ---------------------------------------------------------------------------

class GP:
    def __init__(self, length_scale=1.0, noise=1e-6):
        self.ls = length_scale
        self.noise = noise
        self.X = self.y = self.L = self.alpha = None

    def _kernel(self, X1, X2):
        sq = np.sum(X1**2, 1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2**2, 1)
        return np.exp(-0.5 * sq / self.ls**2)

    def fit(self, X, y):
        self.X, self.y = X, y
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        self.L = cho_factor(K)
        self.alpha = cho_solve(self.L, y)

    def predict(self, Xs):
        Ks = self._kernel(Xs, self.X)
        mu = Ks @ self.alpha
        v = cho_solve(self.L, Ks.T)
        var = np.maximum(1.0 - np.sum(Ks.T * v, 0), 0.0)
        return mu, var


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------

def ucb(mu, var, beta=2.0):
    return mu + beta * np.sqrt(var)


def ei(mu, var, best):
    from scipy.stats import norm
    s = np.sqrt(var + 1e-12)
    z = (mu - best) / s
    return s * (z * norm.cdf(z) + norm.pdf(z))


# ---------------------------------------------------------------------------
# DPP sampling (L-ensemble, exact eigendecomposition-based)
# ---------------------------------------------------------------------------

def build_dpp_kernel(quality, cov_matrix):
    """L_ij = q_i * K_ij * q_j"""
    q = quality[:, None]
    return (q @ q.T) * cov_matrix


def sample_dpp(L, k):
    """Sample a size-k subset from an L-ensemble DPP."""
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals = np.maximum(eigvals, 0)

    # phase 1: select eigenvectors
    probs = eigvals / (eigvals + 1)
    selected = np.where(np.random.rand(len(probs)) < probs)[0]
    if len(selected) == 0:
        return np.random.choice(L.shape[0], size=k, replace=False)

    V = eigvecs[:, selected].copy()
    n = L.shape[0]
    chosen = []

    # phase 2: iteratively sample items
    for _ in range(min(k, len(selected))):
        probs_i = np.sum(V**2, axis=1)
        probs_i = np.maximum(probs_i, 0)
        probs_i /= probs_i.sum() + 1e-12
        idx = np.random.choice(n, p=probs_i)
        chosen.append(idx)

        # project out chosen direction
        j = np.argmax(np.abs(V[idx]))
        Vj = V[:, j: j + 1].copy()
        V -= (V[idx: idx + 1, :] / (Vj[idx] + 1e-12)) * Vj
        V = np.delete(V, j, axis=1)

    while len(chosen) < k:
        remaining = list(set(range(n)) - set(chosen))
        chosen.append(remaining[np.random.randint(len(remaining))])

    return np.array(chosen[:k])


# ---------------------------------------------------------------------------
# PDBO main loop
# ---------------------------------------------------------------------------

def pdbo(objectives, bounds, n_init=5, n_iter=20, batch_size=4):
    """
    Parameters
    ----------
    objectives : list of callables, each f: (d,) -> float
    bounds : (d, 2) array of [lo, hi]
    """
    d = bounds.shape[0]
    n_obj = len(objectives)

    # Latin hypercube init
    X = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(n_init, d)
    Y = np.column_stack([np.array([f(x) for x in X]) for f in objectives])

    acq_fns = [ucb, lambda mu, var: ei(mu, var, best=np.max(Y[:, 0]))]
    acq_rewards = np.ones(len(acq_fns))

    for it in range(n_iter):
        gps = []
        for j in range(n_obj):
            gp = GP()
            gp.fit(X, Y[:, j])
            gps.append(gp)

        # MAB: epsilon-greedy acquisition selection
        if np.random.rand() < 0.1:
            acq_idx = np.random.randint(len(acq_fns))
        else:
            acq_idx = np.argmax(acq_rewards)

        # generate candidate set via random search (cheap MOO proxy)
        n_cand = 500
        X_cand = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(n_cand, d)

        mus, vars_ = [], []
        for gp in gps:
            m, v = gp.predict(X_cand)
            mus.append(m)
            vars_.append(v)

        # quality = sum of acquisition values across objectives
        quality = np.zeros(n_cand)
        for j in range(n_obj):
            if acq_idx == 0:
                quality += ucb(mus[j], vars_[j])
            else:
                quality += ei(mus[j], vars_[j], best=np.max(Y[:, j]))
        quality = np.maximum(quality, 1e-8)

        # GP covariance for DPP kernel
        cov = gps[0]._kernel(X_cand, X_cand)
        L = build_dpp_kernel(quality, cov)
        batch_idx = sample_dpp(L, batch_size)
        X_new = X_cand[batch_idx]

        Y_new = np.column_stack([np.array([f(x) for x in X_new]) for f in objectives])

        # update MAB reward (hypervolume improvement proxy)
        reward = np.mean(np.max(Y_new, axis=0))
        acq_rewards[acq_idx] = 0.9 * acq_rewards[acq_idx] + 0.1 * reward

        X = np.vstack([X, X_new])
        Y = np.vstack([Y, Y_new])
        print(f"iter {it+1}/{n_iter}  |X|={len(X)}  best_obj0={np.max(Y[:,0]):.4f}")

    return X, Y


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def f1(x):
        return -np.sum((x - 0.3) ** 2)

    def f2(x):
        return -np.sum((x - 0.7) ** 2)

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    X, Y = pdbo([f1, f2], bounds, n_init=8, n_iter=10, batch_size=3)
    print(f"Final dataset size: {len(X)}")
