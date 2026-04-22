"""
Sequence-Aware PDBO — extension of arXiv:2406.08799 (PDBO).

Setting: experiments are organized into S sequences (e.g., recipes, wafers,
process flows). Each BO iteration must produce a DOE (Design of Experiments)
of m_s parameter-sweep points per sequence s.

Extension vs. base PDBO (see reference_impl.py):

  1. Multi-task GP per objective with ICM (Intrinsic Coregionalization) kernel

         k((s,x),(s',x')) = B[s,s'] * k_rbf(x,x'),
         B = (1 - rho) * I + rho * 1 1^T,   rho in [0, 1).

     rho is a single shared-variance knob: rho = 0 recovers independent GPs
     per sequence; rho -> 1 is a fully pooled surrogate. diag(B) = 1 so the
     prior variance k((s,x),(s,x)) = 1 is preserved (matches base GP).

  2. Partitioned batch selection: the L-ensemble DPP is run *per sequence*
     over that sequence's candidate pool with quality scored under the
     joint MT-GP. Cross-sequence information transfer happens inside the
     surrogate via B; within-sequence DOE diversity comes from the DPP;
     cross-sequence diversity is guaranteed by the partition (exact m_s
     quota per sequence).

  3. MAB acquisition selection (UCB/EI) and the L-ensemble DPP sampler are
     unchanged from the base PDBO (copied verbatim to keep this file
     self-contained, matching repo convention).

A joint-DPP variant (single DPP over all S * n_cand candidates with B in the
similarity kernel, plus a greedy repair step to enforce quotas) is a natural
stretch goal; kept out of scope here for teaching clarity.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Task kernel B
# ---------------------------------------------------------------------------

def build_task_kernel(S, rho):
    """B = (1 - rho) * I + rho * 1 1^T; PSD for rho in [0, 1)."""
    assert 0.0 <= rho < 1.0, "rho must be in [0, 1)"
    B = (1.0 - rho) * np.eye(S) + rho * np.ones((S, S))
    return B


# ---------------------------------------------------------------------------
# Multi-task GP surrogate (ICM kernel, single output)
# ---------------------------------------------------------------------------

class MTGP:
    """ICM multi-task GP. Mirrors reference_impl.GP but with a sequence id."""

    def __init__(self, S, rho=0.3, length_scale=1.0, noise=1e-6):
        self.S = S
        self.rho = rho
        self.B = build_task_kernel(S, rho)
        self.ls = length_scale
        self.noise = noise
        self.S_tr = self.X_tr = self.y = self.L = self.alpha = None

    def _rbf(self, X1, X2):
        sq = np.sum(X1**2, 1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2**2, 1)
        return np.exp(-0.5 * sq / self.ls**2)

    def _full_kernel(self, S1, X1, S2, X2):
        return self.B[np.ix_(S1, S2)] * self._rbf(X1, X2)

    def fit(self, S_train, X_train, y):
        self.S_tr = np.asarray(S_train, dtype=int)
        self.X_tr = np.asarray(X_train, dtype=float)
        self.y = np.asarray(y, dtype=float)
        K = self._full_kernel(self.S_tr, self.X_tr, self.S_tr, self.X_tr)
        K += self.noise * np.eye(len(self.y))
        self.L = cho_factor(K)
        self.alpha = cho_solve(self.L, self.y)

    def predict(self, S_test, X_test):
        S_test = np.asarray(S_test, dtype=int)
        X_test = np.asarray(X_test, dtype=float)
        Ks = self._full_kernel(S_test, X_test, self.S_tr, self.X_tr)
        mu = Ks @ self.alpha
        v = cho_solve(self.L, Ks.T)
        # diag(B) = 1 and k_rbf(x,x) = 1 → prior variance = 1 for every (s,x)
        var = np.maximum(1.0 - np.sum(Ks.T * v, axis=0), 0.0)
        return mu, var


# ---------------------------------------------------------------------------
# Acquisition functions (copied from reference_impl.py)
# ---------------------------------------------------------------------------

def ucb(mu, var, beta=2.0):
    return mu + beta * np.sqrt(var)


def ei(mu, var, best):
    s = np.sqrt(var + 1e-12)
    z = (mu - best) / s
    return s * (z * norm.cdf(z) + norm.pdf(z))


# ---------------------------------------------------------------------------
# L-ensemble DPP (copied from reference_impl.py)
# ---------------------------------------------------------------------------

def build_dpp_kernel(quality, cov_matrix):
    """L_ij = q_i * K_ij * q_j"""
    q = quality[:, None]
    return (q @ q.T) * cov_matrix


def sample_dpp(L, k):
    """Sample a size-k subset from an L-ensemble DPP."""
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals = np.maximum(eigvals, 0)

    probs = eigvals / (eigvals + 1)
    selected = np.where(np.random.rand(len(probs)) < probs)[0]
    if len(selected) == 0:
        return np.random.choice(L.shape[0], size=k, replace=False)

    V = eigvecs[:, selected].copy()
    n = L.shape[0]
    chosen = []

    for _ in range(min(k, len(selected))):
        probs_i = np.sum(V**2, axis=1)
        probs_i = np.maximum(probs_i, 0)
        probs_i /= probs_i.sum() + 1e-12
        idx = np.random.choice(n, p=probs_i)
        chosen.append(idx)

        j = np.argmax(np.abs(V[idx]))
        Vj = V[:, j: j + 1].copy()
        V -= (V[idx: idx + 1, :] / (Vj[idx] + 1e-12)) * Vj
        V = np.delete(V, j, axis=1)

    while len(chosen) < k:
        remaining = list(set(range(n)) - set(chosen))
        chosen.append(remaining[np.random.randint(len(remaining))])

    return np.array(chosen[:k])


# ---------------------------------------------------------------------------
# Per-sequence candidate generation (cheap MOO proxy = random search)
# ---------------------------------------------------------------------------

def sample_candidates_per_seq(bounds_list, n_cand_per_seq, rng):
    """Return dict {s: (n_cand, d) array}, one random batch per sequence."""
    out = {}
    for s, bounds in enumerate(bounds_list):
        d = bounds.shape[0]
        lo, hi = bounds[:, 0], bounds[:, 1]
        out[s] = lo + (hi - lo) * rng.random((n_cand_per_seq, d))
    return out


# ---------------------------------------------------------------------------
# Partitioned DPP batch selection: exact m_s quota per sequence
# ---------------------------------------------------------------------------

def select_partitioned_batch(cand_by_seq, quality_by_seq, rbf_by_seq, m_per_seq):
    """Run an L-ensemble DPP per sequence; stack the results.

    Args:
        cand_by_seq: {s: (n_cand, d) candidate array}
        quality_by_seq: {s: (n_cand,) positive quality scores}
        rbf_by_seq: {s: (n_cand, n_cand) input-space RBF kernel matrix}
        m_per_seq: list of ints, per-sequence DOE quota

    Returns:
        S_new (N,), X_new (N, d) with N = sum(m_per_seq).
    """
    S_new, X_new = [], []
    for s, m_s in enumerate(m_per_seq):
        if m_s == 0:
            continue
        L = build_dpp_kernel(quality_by_seq[s], rbf_by_seq[s])
        idx = sample_dpp(L, m_s)
        S_new.extend([s] * len(idx))
        X_new.append(cand_by_seq[s][idx])
    return np.array(S_new, dtype=int), np.vstack(X_new)


# ---------------------------------------------------------------------------
# Standalone suggestion: given history, return next batch (no evaluation)
# ---------------------------------------------------------------------------

def suggest_next_batch(
    S_hist: np.ndarray,
    X_hist: np.ndarray,
    Y_hist: np.ndarray,
    bounds_list: list[np.ndarray],
    m_per_seq: list[int],
    rho: float = 0.3,
    acq: str = "ucb",
    n_cand_per_seq: int = 300,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Propose the next DOE batch given historical measurements.

    Args:
        S_hist:         (N,) int sequence labels.
        X_hist:         (N, d) parameter matrix, in the SAME (original) space
                        as bounds_list — no internal normalization.
        Y_hist:         (N,) or (N, n_obj) objective values (maximized).
        bounds_list:    list of S (d, 2) arrays, per-sequence param bounds
                        [lo, hi]. Defines the original-space box to sample in.
        m_per_seq:      list of S ints, per-sequence DOE quota.
        rho:            ICM shared-variance knob in [0, 1). 0 → independent
                        per-sequence GPs; → 1 approaches fully pooled.
        acq:            "ucb" or "ei".
        n_cand_per_seq: random candidates generated per sequence.
        seed:           if given, reseeds np.random globally (DPP uses it) and
                        builds a fresh Generator. NOTE: passing the same seed
                        on successive calls returns the same suggestions —
                        pass None (or a fresh rng) to advance randomness.
        rng:            optional np.random.Generator; ignored if seed is given.

    Returns:
        S_new:  (sum(m_per_seq),) int sequence labels for the suggestions.
        X_new:  (sum(m_per_seq), d) parameter matrix. Same ORIGINAL space as
                X_hist / bounds_list (not normalized).
    """
    assert acq in ("ucb", "ei"), "acq must be 'ucb' or 'ei'"
    if seed is not None:
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
    elif rng is None:
        rng = np.random.default_rng()

    S_hist = np.asarray(S_hist, dtype=int)
    X_hist = np.asarray(X_hist, dtype=float)
    Y_hist = np.asarray(Y_hist, dtype=float)
    if Y_hist.ndim == 1:
        Y_hist = Y_hist[:, None]

    S = len(bounds_list)
    n_obj = Y_hist.shape[1]
    assert n_cand_per_seq >= max(m_per_seq), \
        "n_cand_per_seq must be >= max DOE quota"

    # fit one ICM MTGP per objective on the joint history
    gps = []
    for j in range(n_obj):
        gp = MTGP(S=S, rho=rho)
        gp.fit(S_hist, X_hist, Y_hist[:, j])
        gps.append(gp)

    # per-sequence candidates + acquisition-value quality
    cand_by_seq = sample_candidates_per_seq(bounds_list, n_cand_per_seq, rng)
    quality_by_seq, rbf_by_seq = {}, {}
    for s in range(S):
        X_c = cand_by_seq[s]
        S_c = np.full(len(X_c), s, dtype=int)
        q = np.zeros(len(X_c))
        for j, gp in enumerate(gps):
            mu, var = gp.predict(S_c, X_c)
            if acq == "ucb":
                q += ucb(mu, var)
            else:
                q += ei(mu, var, best=float(np.max(Y_hist[:, j])))
        quality_by_seq[s] = np.maximum(q, 1e-8)
        rbf_by_seq[s] = gps[0]._rbf(X_c, X_c)

    return select_partitioned_batch(
        cand_by_seq, quality_by_seq, rbf_by_seq, m_per_seq
    )


# ---------------------------------------------------------------------------
# Main loop: Sequence-Aware PDBO
# ---------------------------------------------------------------------------

def sa_pdbo(objectives, bounds_list, m_per_seq,
            n_init_per_seq=4, n_iter=8, rho=0.3,
            n_cand_per_seq=300, seed=0):
    """Sequence-aware batch multi-objective Bayesian optimization.

    Args:
        objectives: list of callables f(s:int, x:(d,)) -> float (maximized).
        bounds_list: list of (d, 2) arrays, one per sequence.
        m_per_seq: list of ints, per-sequence DOE quota per BO iteration.
        n_init_per_seq: random-init points per sequence.
        n_iter: BO iterations.
        rho: ICM shared-variance knob in [0, 1).
        n_cand_per_seq: random candidates generated per sequence per iter.

    Returns:
        S_all (N,), X_all (N, d), Y_all (N, n_obj).
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    S = len(bounds_list)
    n_obj = len(objectives)
    assert n_cand_per_seq >= max(m_per_seq), \
        "n_cand_per_seq must be >= max DOE quota"

    # --- random init per sequence --------------------------------------
    S_all, X_all = [], []
    for s, bounds in enumerate(bounds_list):
        d = bounds.shape[0]
        lo, hi = bounds[:, 0], bounds[:, 1]
        X_s = lo + (hi - lo) * rng.random((n_init_per_seq, d))
        X_all.append(X_s)
        S_all.extend([s] * n_init_per_seq)
    X_all = np.vstack(X_all)
    S_all = np.array(S_all, dtype=int)
    Y_all = np.array([[f(int(s), x) for f in objectives]
                      for s, x in zip(S_all, X_all)])

    # --- MAB over acquisitions (UCB, EI) -------------------------------
    acq_names = ["ucb", "ei"]
    acq_rewards = np.ones(len(acq_names))

    for it in range(n_iter):
        # epsilon-greedy MAB picks which acquisition to use this iter
        if rng.random() < 0.1:
            acq_idx = int(rng.integers(len(acq_names)))
        else:
            acq_idx = int(np.argmax(acq_rewards))

        # fit surrogates, score candidates, run partitioned DPP
        S_new, X_new = suggest_next_batch(
            S_all, X_all, Y_all, bounds_list, m_per_seq,
            rho=rho, acq=acq_names[acq_idx],
            n_cand_per_seq=n_cand_per_seq, rng=rng,
        )

        # sanity: exact per-sequence quota
        for s, m_s in enumerate(m_per_seq):
            assert int(np.sum(S_new == s)) == m_s

        # evaluate objectives
        Y_new = np.array([[f(int(s), x) for f in objectives]
                          for s, x in zip(S_new, X_new)])

        # MAB reward: mean of per-sequence best new value across objectives
        reward = float(np.mean(np.max(Y_new, axis=0)))
        acq_rewards[acq_idx] = 0.9 * acq_rewards[acq_idx] + 0.1 * reward

        # append
        S_all = np.concatenate([S_all, S_new])
        X_all = np.vstack([X_all, X_new])
        Y_all = np.vstack([Y_all, Y_new])

        # log per-sequence best-so-far per objective
        bests = []
        for j in range(n_obj):
            per_s = [float(np.max(Y_all[S_all == s, j])) for s in range(S)]
            bests.append(per_s)
        print(
            f"iter {it+1}/{n_iter} | N={len(X_all)} | acq={acq_names[acq_idx]}"
            f" | best_f0_by_seq={[f'{v:.3f}' for v in bests[0]]}"
            f" | best_f1_by_seq={[f'{v:.3f}' for v in bests[1]]}"
        )

    return S_all, X_all, Y_all


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # S=3 sequences, d=2, 2 objectives. Optima are related but shifted across
    # sequences, so the ICM task kernel (rho > 0) can transfer information.

    shifts = {0: 0.0, 1: 0.1, 2: 0.2}

    def f1(s, x):
        # shared optimum at x = 0.3 for all sequences
        return -float(np.sum((x - 0.3) ** 2))

    def f2(s, x):
        # optimum drifts from 0.7 (s=0) → 0.6 (s=1) → 0.5 (s=2)
        target = 0.7 - shifts[s]
        return -float(np.sum((x - target) ** 2))

    # same bounds for all three sequences in this demo; per-sequence bounds
    # would work too — pass a different (d, 2) array per entry.
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    bounds_list = [bounds, bounds, bounds]
    m_per_seq = [3, 3, 3]

    S_all, X_all, Y_all = sa_pdbo(
        objectives=[f1, f2],
        bounds_list=bounds_list,
        m_per_seq=m_per_seq,
        n_init_per_seq=4,
        n_iter=8,
        rho=0.3,
        n_cand_per_seq=300,
        seed=0,
    )
    print(f"Final dataset size: {len(X_all)} "
          f"(expected {4 * 3 + 8 * sum(m_per_seq)})")

    # ---- Standalone suggestion: given tabular history, get next batch ----
    # Typical use case: you already have measurements and just want the next
    # parameter-value pairs to try. No Python objectives involved.
    print("\n--- suggest_next_batch from history ---")
    S_new, X_new = suggest_next_batch(
        S_hist=S_all,
        X_hist=X_all,
        Y_hist=Y_all,                # accepts (N,) or (N, n_obj)
        bounds_list=bounds_list,
        m_per_seq=[2, 2, 2],         # 2 suggestions per sequence
        rho=0.3,
        acq="ucb",
        seed=42,
    )
    for s, x in zip(S_new, X_new):
        print(f"  sequence {s}: x = {np.round(x, 3)}")
