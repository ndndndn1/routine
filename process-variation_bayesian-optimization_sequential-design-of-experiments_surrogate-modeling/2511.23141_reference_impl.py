#!/usr/bin/env python3
"""
Laser Dicing Bayesian Optimization Reference Implementation (Simplified)
========================================================================
Automated Discovery of Laser Dicing Processes with Bayesian Optimization
for Semiconductor Manufacturing.

arXiv: 2511.23141

This minimal implementation demonstrates:
  1. Constrained multi-objective Bayesian optimization with GP surrogates.
  2. Sequential experimental design loop for a synthetic laser dicing problem.
  3. Pareto front discovery with constraint satisfaction.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# 1.  Synthetic laser dicing simulator
# ---------------------------------------------------------------------------

def laser_dicing_simulator(x, noise_std=0.05):
    """
    Synthetic laser dicing process with 4 parameters and 2 objectives + 1 constraint.

    Parameters
    ----------
    x : ndarray, shape (4,)
        Process parameters (normalized to [0, 1]):
        x[0]: laser power
        x[1]: repetition rate
        x[2]: scan speed
        x[3]: focus position

    Returns
    -------
    obj1 : float
        Dicing width (minimize) -- smaller is better
    obj2 : float
        Surface roughness (minimize) -- smaller is better
    con1 : float
        Dicing depth constraint -- must be >= threshold (0.8)
        Returns (depth - 0.8): feasible if >= 0
    """
    power, rep_rate, speed, focus = x

    # Objective 1: Dicing width (minimize)
    width = (0.3 + 0.5 * power - 0.2 * speed + 0.15 * np.sin(2 * np.pi * focus)
             + 0.1 * rep_rate * power)
    width += np.random.randn() * noise_std

    # Objective 2: Surface roughness (minimize)
    roughness = (0.5 - 0.3 * power + 0.4 * speed ** 2 - 0.2 * focus
                 + 0.15 * rep_rate + 0.1 * np.cos(3 * np.pi * power * speed))
    roughness += np.random.randn() * noise_std

    # Constraint: Dicing depth (must be >= 0.8)
    depth = (0.2 + 0.7 * power + 0.1 * rep_rate - 0.3 * speed
             + 0.15 * focus - 0.05 * speed * focus)
    depth += np.random.randn() * noise_std * 0.5
    con1 = depth - 0.8  # feasible if >= 0

    return float(width), float(roughness), float(con1)


# ---------------------------------------------------------------------------
# 2.  Gaussian Process surrogate with ARD kernel
# ---------------------------------------------------------------------------

class GaussianProcess:
    """GP regression with squared-exponential ARD kernel."""

    def __init__(self, length_scales=None, signal_var=1.0, noise_var=0.01):
        self.length_scales = length_scales
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def _kernel(self, X1, X2):
        """Squared-exponential kernel with ARD."""
        ls = self.length_scales
        X1_s = X1 / ls
        X2_s = X2 / ls
        sq_dist = cdist(X1_s, X2_s, metric='sqeuclidean')
        return self.signal_var * np.exp(-0.5 * sq_dist)

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y).ravel()
        if self.length_scales is None:
            self.length_scales = np.ones(X.shape[1]) * 0.3
        K = self._kernel(self.X_train, self.X_train)
        K += self.noise_var * np.eye(len(K))
        self.K_inv = np.linalg.inv(K)

    def predict(self, X):
        X = np.atleast_2d(X)
        K_s = self._kernel(X, self.X_train)
        K_ss = self.signal_var * np.ones(len(X))  # diagonal of k(x*,x*)

        mu = K_s @ self.K_inv @ self.y_train
        var = K_ss - np.sum(K_s @ self.K_inv * K_s, axis=1)
        var = np.maximum(var, 1e-10)
        return mu, var

    def optimize_hyperparams(self, n_restarts=3):
        """Simple marginal likelihood optimization for length scales."""
        best_nll = np.inf
        best_ls = self.length_scales.copy()
        d = self.X_train.shape[1]

        for _ in range(n_restarts):
            ls0 = np.random.uniform(0.1, 1.0, size=d)

            def neg_log_marginal_likelihood(log_ls):
                ls = np.exp(log_ls)
                X_s = self.X_train / ls
                sq_dist = cdist(X_s, X_s, metric='sqeuclidean')
                K = self.signal_var * np.exp(-0.5 * sq_dist)
                K += self.noise_var * np.eye(len(K))
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
                    nll = (0.5 * self.y_train @ alpha
                           + np.sum(np.log(np.diag(L)))
                           + 0.5 * len(self.y_train) * np.log(2 * np.pi))
                    return nll
                except np.linalg.LinAlgError:
                    return 1e10

            res = minimize(neg_log_marginal_likelihood, np.log(ls0),
                           method='L-BFGS-B',
                           bounds=[(-2, 2)] * d)
            if res.fun < best_nll:
                best_nll = res.fun
                best_ls = np.exp(res.x)

        self.length_scales = best_ls
        self.fit(self.X_train, self.y_train)  # refit with optimized params


# ---------------------------------------------------------------------------
# 3.  Acquisition function: Constrained Expected Improvement (for each obj)
# ---------------------------------------------------------------------------

def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Standard Expected Improvement."""
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best - mu - xi  # minimization: improvement = y_best - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = np.where(sigma > 1e-10, ei, 0.0)
    return ei


def constrained_ehvi_acquisition(x, gp_obj1, gp_obj2, gp_con,
                                 y1_best, y2_best, xi=0.01):
    """
    Simplified constrained multi-objective acquisition.
    Combines EI for each objective with probability of feasibility.
    """
    x = np.atleast_2d(x)

    mu1, var1 = gp_obj1.predict(x)
    mu2, var2 = gp_obj2.predict(x)
    mu_c, var_c = gp_con.predict(x)

    sigma1 = np.sqrt(var1)
    sigma2 = np.sqrt(var2)
    sigma_c = np.sqrt(var_c)

    ei1 = expected_improvement(mu1, sigma1, y1_best, xi)
    ei2 = expected_improvement(mu2, sigma2, y2_best, xi)

    # Probability of constraint satisfaction: P(con >= 0)
    p_feasible = norm.cdf(mu_c / sigma_c)

    # Combined acquisition: product of EI and feasibility
    acq = (ei1 + ei2) * p_feasible
    return acq


# ---------------------------------------------------------------------------
# 4.  Latin Hypercube Sampling for initial design
# ---------------------------------------------------------------------------

def latin_hypercube_sampling(n_samples, d, seed=None):
    """Generate LHS samples in [0, 1]^d."""
    rng = np.random.RandomState(seed)
    samples = np.zeros((n_samples, d))
    for i in range(d):
        perm = rng.permutation(n_samples)
        samples[:, i] = (perm + rng.uniform(size=n_samples)) / n_samples
    return samples


# ---------------------------------------------------------------------------
# 5.  Pareto front utilities
# ---------------------------------------------------------------------------

def is_pareto_efficient(costs):
    """Find Pareto-efficient points (minimization)."""
    n = len(costs)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        is_efficient[i] = False
        dominated = np.all(costs[i] <= costs, axis=1) & np.any(costs[i] < costs, axis=1)
        is_efficient[dominated] = False
        is_efficient[i] = True
    return is_efficient


def hypervolume_2d(pareto_points, ref_point):
    """Compute 2D hypervolume indicator."""
    if len(pareto_points) == 0:
        return 0.0
    sorted_pts = pareto_points[pareto_points[:, 0].argsort()]
    hv = 0.0
    prev_y = ref_point[1]
    for pt in sorted_pts:
        if pt[0] < ref_point[0] and pt[1] < ref_point[1]:
            hv += (ref_point[0] - pt[0]) * (prev_y - pt[1])
            prev_y = pt[1]
    return hv


# ---------------------------------------------------------------------------
# 6.  Main BO loop
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    d = 4  # process parameters
    n_init = 15  # initial LHS samples
    n_iter = 35  # BO iterations
    noise_std = 0.05

    print("=" * 65)
    print("Laser Dicing Bayesian Optimization Reference Implementation")
    print("=" * 65)

    # --- Initial design via LHS ---
    print(f"\n[1] Initial design: {n_init} LHS samples in [0,1]^{d}")
    X = latin_hypercube_sampling(n_init, d, seed=42)
    Y_obj1 = []
    Y_obj2 = []
    Y_con = []

    for x in X:
        o1, o2, c1 = laser_dicing_simulator(x, noise_std=noise_std)
        Y_obj1.append(o1)
        Y_obj2.append(o2)
        Y_con.append(c1)

    Y_obj1 = np.array(Y_obj1)
    Y_obj2 = np.array(Y_obj2)
    Y_con = np.array(Y_con)

    feasible = Y_con >= 0
    print(f"    Initial feasible points: {np.sum(feasible)}/{n_init}")

    # --- Sequential BO loop ---
    print(f"\n[2] Running {n_iter} sequential BO iterations...")

    for iteration in range(n_iter):
        # Build GP surrogates
        gp1 = GaussianProcess(noise_var=noise_std ** 2 + 1e-4)
        gp2 = GaussianProcess(noise_var=noise_std ** 2 + 1e-4)
        gp_c = GaussianProcess(noise_var=(noise_std * 0.5) ** 2 + 1e-4)

        gp1.fit(X, Y_obj1)
        gp2.fit(X, Y_obj2)
        gp_c.fit(X, Y_con)

        # Optimize hyperparameters every 10 iterations
        if iteration % 10 == 0:
            gp1.optimize_hyperparams()
            gp2.optimize_hyperparams()
            gp_c.optimize_hyperparams()

        # Current best feasible objectives
        if np.any(feasible):
            y1_best = np.min(Y_obj1[feasible])
            y2_best = np.min(Y_obj2[feasible])
        else:
            y1_best = np.max(Y_obj1)
            y2_best = np.max(Y_obj2)

        # Optimize acquisition function via random search + local refinement
        n_candidates = 500
        X_cand = np.random.rand(n_candidates, d)
        acq_vals = constrained_ehvi_acquisition(
            X_cand, gp1, gp2, gp_c, y1_best, y2_best
        )
        best_idx = np.argmax(acq_vals)
        x_next = X_cand[best_idx]

        # Local refinement
        def neg_acq(x_flat):
            x_flat = np.clip(x_flat, 0, 1)
            return -constrained_ehvi_acquisition(
                x_flat.reshape(1, -1), gp1, gp2, gp_c, y1_best, y2_best
            )[0]

        res = minimize(neg_acq, x_next, method='L-BFGS-B',
                       bounds=[(0, 1)] * d)
        x_next = np.clip(res.x, 0, 1)

        # Evaluate new point
        o1, o2, c1 = laser_dicing_simulator(x_next, noise_std=noise_std)

        X = np.vstack([X, x_next])
        Y_obj1 = np.append(Y_obj1, o1)
        Y_obj2 = np.append(Y_obj2, o2)
        Y_con = np.append(Y_con, c1)
        feasible = Y_con >= 0

        if (iteration + 1) % 10 == 0:
            n_feas = np.sum(feasible)
            feas_obj = np.column_stack([Y_obj1[feasible], Y_obj2[feasible]])
            pareto_mask = is_pareto_efficient(feas_obj) if n_feas > 0 else np.array([])
            n_pareto = np.sum(pareto_mask) if len(pareto_mask) > 0 else 0
            print(f"    Iter {iteration+1:3d}: {n_feas} feasible, "
                  f"{n_pareto} Pareto-optimal points")

    # --- Final results ---
    print(f"\n[3] Final results after {n_init + n_iter} total evaluations:")
    feasible = Y_con >= 0
    n_feas = np.sum(feasible)
    print(f"    Total feasible points: {n_feas}/{len(Y_obj1)}")

    if n_feas > 0:
        feas_obj = np.column_stack([Y_obj1[feasible], Y_obj2[feasible]])
        pareto_mask = is_pareto_efficient(feas_obj)
        pareto_front = feas_obj[pareto_mask]
        pareto_X = X[feasible][pareto_mask]

        print(f"    Pareto front size: {len(pareto_front)}")
        print(f"\n    Pareto-optimal solutions:")
        print(f"    {'Width':>8s}  {'Roughness':>10s}  "
              f"{'Power':>6s}  {'RepRate':>8s}  {'Speed':>6s}  {'Focus':>6s}")
        print(f"    {'-'*8}  {'-'*10}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*6}")

        sort_idx = pareto_front[:, 0].argsort()
        for i in sort_idx:
            print(f"    {pareto_front[i, 0]:8.4f}  {pareto_front[i, 1]:10.4f}  "
                  f"{pareto_X[i, 0]:6.3f}  {pareto_X[i, 1]:8.3f}  "
                  f"{pareto_X[i, 2]:6.3f}  {pareto_X[i, 3]:6.3f}")

        # Hypervolume
        ref_point = np.array([1.5, 1.5])
        hv = hypervolume_2d(pareto_front, ref_point)
        print(f"\n    Hypervolume (ref=[1.5, 1.5]): {hv:.4f}")

    # --- GP feature importance via ARD length scales ---
    print(f"\n[4] ARD length scales (smaller = more important):")
    param_names = ["Power", "RepRate", "Speed", "Focus"]
    for name, ls in zip(param_names, gp1.length_scales):
        importance = 1.0 / ls
        bar = "#" * int(importance * 5)
        print(f"    {name:>8s}: l={ls:.3f}  importance={importance:.3f}  {bar}")

    print("\n" + "=" * 65)
    print("Done. Multi-objective BO discovered Pareto-optimal laser dicing")
    print("conditions balancing width vs roughness under depth constraint.")
    print("=" * 65)


if __name__ == "__main__":
    main()
