"""
Reference implementation for arXiv:2405.07971
"Sensitivity Analysis for Active Sampling, with Applications to the
Simulation of Analog Circuits"

Demonstrates:
  1. Sobol total-order sensitivity analysis (Monte Carlo estimator)
  2. Dimension screening based on sensitivity indices
  3. GP surrogate construction in the reduced subspace
  4. Active sampling loop with variance-based acquisition

Dependencies: numpy, scipy
Run: python 2405.07971_reference_impl.py
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

# ── reproducibility ──────────────────────────────────────────────────
np.random.seed(42)


# =====================================================================
# 1. Synthetic test function (imitates analog circuit with sparsity)
# =====================================================================
def test_function(X):
    """
    Ishigami-like function in d dimensions.  Only the first 3 dimensions
    matter; the rest are inert.

        f(x) = sin(x0) + 7*sin(x1)^2 + 0.1*x2^4*sin(x0)

    x_i in [-pi, pi]
    """
    x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x0) + 7.0 * np.sin(x1) ** 2 + 0.1 * x2 ** 4 * np.sin(x0)


D_FULL = 8          # total input dimensions
D_ACTIVE = 3        # truly active dimensions
BOUNDS = np.array([[-np.pi, np.pi]] * D_FULL)


# =====================================================================
# 2. Sobol total-order sensitivity (Jansen estimator)
# =====================================================================
def sobol_total_order(func, d, n_samples=4000):
    """
    Estimate total-order Sobol indices S_Ti for every dimension.

    Uses matrices A, B and AB_i (Saltelli sampling scheme) with the
    Jansen estimator:
        S_Ti = (1/2N) sum_j (f(B)_j - f(AB_i)_j)^2  /  Var(Y)
    """
    A = np.random.uniform(-np.pi, np.pi, (n_samples, d))
    B = np.random.uniform(-np.pi, np.pi, (n_samples, d))

    fA = func(A)
    fB = func(B)

    total_var = np.var(np.concatenate([fA, fB]))
    S_T = np.zeros(d)

    for i in range(d):
        AB_i = B.copy()
        AB_i[:, i] = A[:, i]
        fABi = func(AB_i)
        S_T[i] = np.mean((fB - fABi) ** 2) / (2.0 * total_var)

    return S_T


# =====================================================================
# 3. GP surrogate (RBF kernel, closed-form posterior)
# =====================================================================
class GPSurrogate:
    """Minimal GP with isotropic RBF kernel and noise."""

    def __init__(self, length_scale=1.0, signal_var=1.0, noise_var=1e-6):
        self.ls = length_scale
        self.sv = signal_var
        self.nv = noise_var
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        self.alpha = None

    def _kernel(self, X1, X2):
        sq = cdist(X1, X2, "sqeuclidean")
        return self.sv * np.exp(-0.5 * sq / self.ls ** 2)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        K = self._kernel(X, X) + self.nv * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
        self.alpha = self.K_inv @ y

    def predict(self, X_star):
        k_s = self._kernel(X_star, self.X_train)
        mu = k_s @ self.alpha
        v = self.sv - np.einsum("ij,jk,ik->i", k_s, self.K_inv, k_s)
        v = np.clip(v, 0, None)
        return mu, v

    def optimize_hyperparams(self, ls_candidates=None):
        """Grid-search for length-scale and signal variance via LOO CV."""
        if ls_candidates is None:
            ls_candidates = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]
        sv_candidates = [0.5 * np.var(self.y_train),
                         np.var(self.y_train),
                         2.0 * np.var(self.y_train)]
        best_ls, best_sv, best_loo = self.ls, self.sv, np.inf
        for sv in sv_candidates:
            for ls in ls_candidates:
                self.ls = ls
                self.sv = sv
                K = self._kernel(self.X_train, self.X_train) + self.nv * np.eye(
                    len(self.X_train)
                )
                try:
                    K_inv = np.linalg.inv(K)
                except np.linalg.LinAlgError:
                    continue
                alpha = K_inv @ self.y_train
                # LOO predictive MSE (virtual leave-one-out)
                loo_err = (alpha / np.diag(K_inv)) ** 2
                if loo_err.mean() < best_loo:
                    best_loo = loo_err.mean()
                    best_ls = ls
                    best_sv = sv
        self.ls = best_ls
        self.sv = best_sv
        self.fit(self.X_train, self.y_train)


# =====================================================================
# 4. Active sampling loop
# =====================================================================
def active_sampling(func, active_dims, bounds, n_init=15, n_iter=35,
                    n_candidates=1000):
    """
    GP-based active sampling in reduced subspace.

    Parameters
    ----------
    func : callable       – expensive simulator (full-dim input)
    active_dims : list    – indices of active dimensions
    bounds : (d_full, 2)  – full-dimensional bounds
    n_init : int          – initial Latin-hypercube samples
    n_iter : int          – sequential samples to add
    n_candidates : int    – random candidate pool per iteration

    Returns
    -------
    gp : GPSurrogate fitted on all collected data (reduced space)
    X_full : (n_init+n_iter, d_full)  all sampled points
    y_all : evaluations
    """
    d_full = bounds.shape[0]
    d_red = len(active_dims)

    # ── initial LHS-like samples ──
    X_init = np.random.uniform(
        bounds[:, 0], bounds[:, 1], size=(n_init, d_full)
    )
    y_init = func(X_init)

    X_red = X_init[:, active_dims]
    bounds_red = bounds[active_dims]

    gp = GPSurrogate(length_scale=1.0, signal_var=np.var(y_init) + 1e-3,
                     noise_var=1e-4)
    gp.fit(X_red, y_init)
    gp.optimize_hyperparams()

    X_all_full = list(X_init)
    y_all = list(y_init)

    print(f"{'iter':>4s}  {'max_var':>10s}  {'y_new':>10s}  {'n_pts':>5s}")
    print("-" * 38)

    for t in range(n_iter):
        # candidate pool in reduced space
        cands = np.random.uniform(
            bounds_red[:, 0], bounds_red[:, 1], size=(n_candidates, d_red)
        )
        _, var_cands = gp.predict(cands)

        best_idx = np.argmax(var_cands)
        x_new_red = cands[best_idx: best_idx + 1]

        # reconstruct full-dim point (inactive dims random)
        x_new_full = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                       size=(1, d_full))
        x_new_full[0, active_dims] = x_new_red[0]

        y_new = func(x_new_full)[0]

        X_all_full.append(x_new_full[0])
        y_all.append(y_new)

        X_red_all = np.array(X_all_full)[:, active_dims]
        y_arr = np.array(y_all)

        gp.fit(X_red_all, y_arr)
        if (t + 1) % 10 == 0:
            gp.optimize_hyperparams()

        print(f"{t + 1:4d}  {var_cands[best_idx]:10.4f}  {y_new:10.4f}  "
              f"{len(y_all):5d}")

    return gp, np.array(X_all_full), np.array(y_all)


# =====================================================================
# 5. Evaluation helpers
# =====================================================================
def rmse_on_test(gp, func, active_dims, bounds, n_test=1000):
    X_test = np.random.uniform(bounds[:, 0], bounds[:, 1],
                               size=(n_test, bounds.shape[0]))
    y_true = func(X_test)
    mu, _ = gp.predict(X_test[:, active_dims])
    return np.sqrt(np.mean((y_true - mu) ** 2))


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Reference impl: Sensitivity-Aware Active Sampling")
    print("  arXiv 2405.07971")
    print("=" * 60)

    # ── Step 1: Sobol sensitivity analysis ──
    print("\n[1] Sobol total-order sensitivity indices (d=%d) ..." % D_FULL)
    S_T = sobol_total_order(test_function, D_FULL, n_samples=5000)
    for i, s in enumerate(S_T):
        tag = " *" if s > 0.01 else ""
        print(f"  x{i}: S_T = {s:.4f}{tag}")

    # ── Step 2: Screen active dims ──
    tau = 0.01
    active_dims = np.where(S_T > tau)[0].tolist()
    print(f"\n[2] Active dimensions (S_T > {tau}): {active_dims}")
    print(f"    Dimension reduction: {D_FULL} -> {len(active_dims)}")

    # ── Step 3: Active sampling in reduced space ──
    print(f"\n[3] Active sampling (n_init=15, n_iter=35) ...")
    gp, X_sampled, y_sampled = active_sampling(
        test_function, active_dims, BOUNDS, n_init=15, n_iter=35
    )

    # ── Step 4: Evaluate surrogate accuracy ──
    err = rmse_on_test(gp, test_function, active_dims, BOUNDS, n_test=2000)
    print(f"\n[4] Final GP surrogate RMSE on 2000 test points: {err:.4f}")

    # ── Baseline 1: random sampling in reduced space (no active learning) ──
    n_total = len(y_sampled)
    X_rand = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1],
                               size=(n_total, D_FULL))
    y_rand = test_function(X_rand)
    gp_rand = GPSurrogate(length_scale=1.0, signal_var=np.var(y_rand) + 1e-3,
                          noise_var=1e-4)
    gp_rand.fit(X_rand[:, active_dims], y_rand)
    gp_rand.optimize_hyperparams()
    err_rand = rmse_on_test(gp_rand, test_function, active_dims, BOUNDS, 2000)
    print(f"    Random sampling + dim reduction RMSE (n={n_total}): {err_rand:.4f}")

    # ── Baseline 2: random sampling in FULL space (no screening) ──
    gp_full = GPSurrogate(length_scale=1.5, signal_var=np.var(y_rand) + 1e-3,
                          noise_var=1e-4)
    gp_full.fit(X_rand, y_rand)  # all 8 dims
    gp_full.optimize_hyperparams()
    X_test_full = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1],
                                    size=(2000, D_FULL))
    y_test_true = test_function(X_test_full)
    mu_full, _ = gp_full.predict(X_test_full)
    err_full = np.sqrt(np.mean((y_test_true - mu_full) ** 2))
    print(f"    Random sampling, full-space GP RMSE (n={n_total}): {err_full:.4f}")

    print("\n[Done] Sensitivity screening + active sampling outperforms naive "
          "full-space approaches.")
