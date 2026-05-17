"""Minimal reference implementation: Coupled Input-Output Dimension Reduction
(arXiv:2406.13425).

Demonstrates alternating eigendecomposition for joint input-output subspace
identification, applied to goal-oriented sensitivity analysis.

Run:
    python reference_impl.py
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import eigh, svd


# ---------------------------------------------------------------------------
# Toy forward model: linear map with nonlinear perturbation ------------------
# ---------------------------------------------------------------------------
def make_forward_model(d_x: int = 10, d_y: int = 8, seed: int = 42):
    """Create a forward model f(x) = A @ x + small nonlinear term."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d_y, d_x))
    U, S, Vt = svd(A, full_matrices=False)
    S_decayed = S * np.exp(-0.3 * np.arange(len(S)))
    A = U @ np.diag(S_decayed) @ Vt

    def f(x: np.ndarray) -> np.ndarray:
        lin = x @ A.T
        nonlin = 0.1 * np.sin(lin)
        return lin + nonlin

    def jacobian(x: np.ndarray) -> np.ndarray:
        lin = x @ A.T
        J = A[np.newaxis, :, :] * (1 + 0.1 * np.cos(lin))[:, :, np.newaxis]
        return J

    return f, jacobian, A


# ---------------------------------------------------------------------------
# Coupled dimension reduction via alternating eigendecomposition -------------
# ---------------------------------------------------------------------------
def coupled_reduction(jacobian_fn, X_samples: np.ndarray,
                      r: int = 2, s: int = 2, max_iter: int = 20,
                      tol: float = 1e-6):
    """
    Alternating eigendecomposition to find coupled (P_r, Q_s).

    C(Q) = E[ J^T Q Q^T J ]   -> top-r eigenvectors give P_r
    D(P) = E[ J P P^T J^T ]   -> top-s eigenvectors give Q_s
    """
    N = len(X_samples)
    Js = jacobian_fn(X_samples)  # (N, d_y, d_x)
    d_y, d_x = Js.shape[1], Js.shape[2]

    rng = np.random.default_rng(0)
    Q = np.linalg.qr(rng.standard_normal((d_y, s)))[0]

    for it in range(max_iter):
        Q_old = Q.copy()

        # Step 1: Fix Q, compute C(Q), extract P
        C = np.zeros((d_x, d_x))
        for j in range(N):
            JtQ = Js[j].T @ Q  # (d_x, s)
            C += JtQ @ JtQ.T
        C /= N
        eigvals_C, eigvecs_C = eigh(C)
        P = eigvecs_C[:, -r:]  # top-r

        # Step 2: Fix P, compute D(P), extract Q
        D = np.zeros((d_y, d_y))
        for j in range(N):
            JP = Js[j] @ P  # (d_y, r)
            D += JP @ JP.T
        D /= N
        eigvals_D, eigvecs_D = eigh(D)
        Q = eigvecs_D[:, -s:]  # top-s

        # Check convergence (subspace angle)
        diff = np.linalg.norm(Q @ Q.T - Q_old @ Q_old.T)
        if diff < tol:
            print(f"  Converged at iteration {it+1} (diff={diff:.2e})")
            break

    return P, Q


# ---------------------------------------------------------------------------
# Goal-oriented sensitivity analysis ----------------------------------------
# ---------------------------------------------------------------------------
def goal_oriented_sensitivity(f, Q_s: np.ndarray, d_x: int,
                              N: int = 5000, seed: int = 123):
    """Compute Sobol first-order indices for g(y) = Q_s^T y."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d_x))
    Y = f(X)
    g = Y @ Q_s  # (N, s)

    total_var = np.var(g, axis=0)  # (s,)
    S_indices = np.zeros((d_x, Q_s.shape[1]))

    n_bins = 20
    for i in range(d_x):
        bins = np.quantile(X[:, i], np.linspace(0, 1, n_bins + 1))
        bin_idx = np.digitize(X[:, i], bins, right=True)
        bin_idx = np.clip(bin_idx, 1, n_bins)
        cond_means = np.zeros((n_bins, Q_s.shape[1]))
        for b in range(1, n_bins + 1):
            mask = bin_idx == b
            if mask.sum() > 0:
                cond_means[b - 1] = g[mask].mean(axis=0)
        var_cond_mean = np.var(cond_means, axis=0)
        S_indices[i] = var_cond_mean / (total_var + 1e-30)

    return S_indices


# ---------------------------------------------------------------------------
# Demo ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    d_x, d_y = 10, 8
    r, s = 2, 2
    N_samples = 1000

    f, jacobian_fn, A = make_forward_model(d_x, d_y)
    rng = np.random.default_rng(99)
    X_samples = rng.standard_normal((N_samples, d_x))

    print("Coupled Input-Output Dimension Reduction")
    print("=" * 50)
    print(f"Input dim: {d_x}, Output dim: {d_y}")
    print(f"Reduced dims: r={r} (input), s={s} (output)\n")

    P, Q = coupled_reduction(jacobian_fn, X_samples, r=r, s=s)

    # Compare full vs reduced reconstruction
    Y_full = f(X_samples)
    X_reduced = X_samples @ P @ P.T
    Y_reduced = f(X_reduced)
    proj_error = np.mean(np.sum((Q.T @ Y_full.T - Q.T @ Y_reduced.T)**2, axis=0))
    print(f"\nProjected reconstruction MSE: {proj_error:.6f}")

    # Independent reduction baseline (SVD on A)
    U, S_vals, Vt = svd(A, full_matrices=False)
    P_ind = Vt[:r].T
    Q_ind = U[:, :s]
    X_ind = X_samples @ P_ind @ P_ind.T
    Y_ind = f(X_ind)
    proj_error_ind = np.mean(np.sum((Q_ind.T @ Y_full.T - Q_ind.T @ Y_ind.T)**2, axis=0))
    print(f"Independent reduction MSE:    {proj_error_ind:.6f}")

    # Goal-oriented sensitivity
    print("\n--- Goal-oriented Sensitivity (coupled Q) ---")
    S_coupled = goal_oriented_sensitivity(f, Q, d_x)
    for i in range(d_x):
        bar = "#" * int(S_coupled[i, 0] * 50)
        print(f"  x_{i}: S={S_coupled[i,0]:.3f} {bar}")

    print("\n--- Standard Sensitivity (identity Q) ---")
    Q_id = np.eye(d_y)[:, :s]
    S_standard = goal_oriented_sensitivity(f, Q_id, d_x)
    for i in range(d_x):
        bar = "#" * int(S_standard[i, 0] * 50)
        print(f"  x_{i}: S={S_standard[i,0]:.3f} {bar}")
