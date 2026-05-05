"""
Smoke test + sequence-DOE demo for sequence_aware_impl.py.

Two halves:

  Part A — SMOKE TEST
    Quick assertions on every public building block: task kernel PSD-ness,
    MTGP fit/predict shapes and variance non-negativity, DPP quota
    exactness, and a tiny end-to-end sa_pdbo run.

  Part B — SEQUENCE-DOE DEMO
    Runs sa_pdbo and prints each BO iteration's DOE batch *per sequence*
    so the DOE structure (the m_s parameter-sweep points for sequence s)
    is explicit. The main sa_pdbo logger only prints per-sequence bests;
    here we wrap it and also show the raw picked points.

Run:
    python smoke_test.py
"""

import numpy as np

from sequence_aware_impl import (
    MTGP,
    build_task_kernel,
    build_dpp_kernel,
    sample_dpp,
    select_partitioned_batch,
    sa_pdbo,
    ucb,
    ei,
)


# ---------------------------------------------------------------------------
# Part A — smoke test
# ---------------------------------------------------------------------------

def test_task_kernel_psd():
    for rho in [0.0, 0.3, 0.5, 0.99]:
        B = build_task_kernel(S=4, rho=rho)
        assert B.shape == (4, 4)
        eigs = np.linalg.eigvalsh(B)
        assert eigs.min() > -1e-12, f"B not PSD at rho={rho}: {eigs}"
        assert np.allclose(np.diag(B), 1.0)
    print("  [ok] build_task_kernel: PSD, diag=1 for rho in {0, 0.3, 0.5, 0.99}")


def test_mtgp_shapes_and_variance():
    rng = np.random.default_rng(0)
    S, d, N, M = 3, 2, 12, 7
    S_tr = rng.integers(0, S, size=N)
    X_tr = rng.random((N, d))
    y = rng.standard_normal(N)

    gp = MTGP(S=S, rho=0.3, length_scale=0.5, noise=1e-4)
    gp.fit(S_tr, X_tr, y)

    S_te = rng.integers(0, S, size=M)
    X_te = rng.random((M, d))
    mu, var = gp.predict(S_te, X_te)
    assert mu.shape == (M,) and var.shape == (M,)
    assert np.all(var >= 0.0), f"variance went negative: {var}"

    # On a point far from data and different sequence, variance should be close
    # to prior variance = 1 when rho is small.
    gp_indep = MTGP(S=S, rho=0.0, length_scale=0.05)
    gp_indep.fit(np.zeros(1, dtype=int), np.array([[0.0, 0.0]]), np.array([1.0]))
    _, var_far = gp_indep.predict(np.array([1]), np.array([[0.9, 0.9]]))
    assert abs(var_far[0] - 1.0) < 1e-6, f"prior var not 1 under rho=0: {var_far}"
    print("  [ok] MTGP: shapes match, var>=0, prior-var = 1 when decoupled")


def test_dpp_quota_exact():
    rng = np.random.default_rng(1)
    np.random.seed(1)
    n_cand = 40
    x = rng.random((n_cand, 2))
    # quality ∝ proximity to centre → high-quality cluster
    q = np.exp(-np.sum((x - 0.5) ** 2, axis=1))
    sq = np.sum(x**2, 1, keepdims=True) - 2 * x @ x.T + np.sum(x**2, 1)
    K = np.exp(-0.5 * sq / 0.2**2)
    L = build_dpp_kernel(q, K)
    idx = sample_dpp(L, k=5)
    assert len(idx) == 5
    assert len(set(int(i) for i in idx)) == 5, "DPP returned duplicates"
    print("  [ok] sample_dpp: returns k distinct items")


def test_partitioned_batch_quotas():
    rng = np.random.default_rng(2)
    np.random.seed(2)
    S, d = 3, 2
    n_cand = 60
    m_per_seq = [2, 4, 3]

    cand_by_seq, quality_by_seq, rbf_by_seq = {}, {}, {}
    for s in range(S):
        Xc = rng.random((n_cand, d))
        sq = np.sum(Xc**2, 1, keepdims=True) - 2 * Xc @ Xc.T + np.sum(Xc**2, 1)
        K = np.exp(-0.5 * sq / 0.25**2)
        q = np.maximum(rng.random(n_cand), 1e-3)
        cand_by_seq[s] = Xc
        quality_by_seq[s] = q
        rbf_by_seq[s] = K

    S_new, X_new = select_partitioned_batch(
        cand_by_seq, quality_by_seq, rbf_by_seq, m_per_seq
    )
    assert X_new.shape == (sum(m_per_seq), d)
    for s, m_s in enumerate(m_per_seq):
        assert int(np.sum(S_new == s)) == m_s, \
            f"quota violated for seq {s}: got {int(np.sum(S_new == s))}, want {m_s}"
    print(f"  [ok] select_partitioned_batch: quotas {m_per_seq} exact")


def test_acq_signatures():
    mu = np.array([0.1, -0.2, 0.5])
    var = np.array([0.3, 0.05, 0.8])
    assert ucb(mu, var).shape == (3,)
    assert ei(mu, var, best=0.0).shape == (3,)
    print("  [ok] ucb/ei: shape preserved")


def test_sa_pdbo_tiny():
    def f1(s, x):
        return -float(np.sum((x - 0.3) ** 2))

    def f2(s, x):
        return -float(np.sum((x - 0.6) ** 2))

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    bounds_list = [bounds, bounds]        # S = 2
    m_per_seq = [2, 3]                     # heterogeneous quotas
    n_init = 3
    n_iter = 2

    S_all, X_all, Y_all = sa_pdbo(
        objectives=[f1, f2],
        bounds_list=bounds_list,
        m_per_seq=m_per_seq,
        n_init_per_seq=n_init,
        n_iter=n_iter,
        rho=0.3,
        n_cand_per_seq=60,
        seed=7,
    )

    expected = n_init * len(bounds_list) + n_iter * sum(m_per_seq)
    assert len(X_all) == expected, f"got {len(X_all)}, want {expected}"
    assert Y_all.shape == (expected, 2)
    assert set(int(s) for s in S_all) == {0, 1}
    # Overall per-sequence counts must equal init + iter * quota.
    for s, m_s in enumerate(m_per_seq):
        expected_s = n_init + n_iter * m_s
        got = int(np.sum(S_all == s))
        assert got == expected_s, f"seq {s}: got {got}, want {expected_s}"
    print(f"  [ok] sa_pdbo tiny run: N={len(X_all)}, "
          f"per-seq counts = {[int(np.sum(S_all == s)) for s in range(2)]}")


# ---------------------------------------------------------------------------
# Part B — sequence-DOE demo (shows each iteration's DOE batch per sequence)
# ---------------------------------------------------------------------------

def doe_demo():
    """Run sa_pdbo and print each iter's DOE points grouped by sequence."""
    shifts = {0: 0.0, 1: 0.1, 2: 0.2}

    def f1(s, x):
        return -float(np.sum((x - 0.3) ** 2))

    def f2(s, x):
        target = 0.7 - shifts[s]
        return -float(np.sum((x - target) ** 2))

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    bounds_list = [bounds] * 3
    m_per_seq = [3, 3, 3]
    n_init = 4
    n_iter = 4
    S = len(bounds_list)

    S_all, X_all, Y_all = sa_pdbo(
        objectives=[f1, f2],
        bounds_list=bounds_list,
        m_per_seq=m_per_seq,
        n_init_per_seq=n_init,
        n_iter=n_iter,
        rho=0.3,
        n_cand_per_seq=250,
        seed=0,
    )

    # Reconstruct per-iter DOE slices from the append order.
    # layout: [ init (n_init*S rows) | iter1 (sum(m)) | iter2 ... ]
    init_rows = n_init * S
    per_iter_rows = sum(m_per_seq)

    print("\n--- sequence-DOE batches per BO iteration ---")
    for it in range(n_iter):
        start = init_rows + it * per_iter_rows
        end = start + per_iter_rows
        S_it = S_all[start:end]
        X_it = X_all[start:end]
        Y_it = Y_all[start:end]
        print(f"\niter {it+1}:")
        for s in range(S):
            mask = (S_it == s)
            X_s = X_it[mask]
            Y_s = Y_it[mask]
            assert len(X_s) == m_per_seq[s]      # DOE quota sanity
            pts = ", ".join(f"({x[0]:.2f},{x[1]:.2f})" for x in X_s)
            f1s = ", ".join(f"{y[0]:+.3f}" for y in Y_s)
            f2s = ", ".join(f"{y[1]:+.3f}" for y in Y_s)
            print(f"  seq {s} (DOE m={m_per_seq[s]}): x=[{pts}]  "
                  f"f1=[{f1s}]  f2=[{f2s}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PART A: smoke test ===")
    test_task_kernel_psd()
    test_mtgp_shapes_and_variance()
    test_dpp_quota_exact()
    test_partitioned_batch_quotas()
    test_acq_signatures()
    test_sa_pdbo_tiny()
    print("\nall smoke tests passed.\n")

    print("=== PART B: sequence-DOE demo ===")
    doe_demo()
