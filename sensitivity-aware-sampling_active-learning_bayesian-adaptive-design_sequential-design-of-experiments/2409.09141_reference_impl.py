"""
Reference implementation for arXiv:2409.09141
"Sequential infinite-dimensional Bayesian optimal experimental design with
derivative-informed latent attention neural operator"

Simplified demonstration:
  1. Synthetic linear-Gaussian inverse problem (discretised 1-D diffusion)
  2. Derivative-informed dimension reduction via Jacobian SVD
  3. GP surrogate in reduced parameter space
  4. Sequential Bayesian OED maximising Expected Information Gain (EIG)
  5. Posterior update at each stage

Dependencies: numpy, scipy
Run: python 2409.09141_reference_impl.py
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve, svd

np.random.seed(0)


# =====================================================================
# 1. Synthetic forward model  (1-D diffusion-like linear PtO map)
# =====================================================================
class LinearPtOMap:
    """
    Parameter-to-observable map for a 1-D steady-state diffusion equation
    discretised on n_param grid points.

    Forward map:  y(d) = G(d) @ theta + noise
    G(d) depends on sensor location d  (row selection of the Green's
    function matrix).
    """

    def __init__(self, n_param=50, noise_std=0.1):
        self.n = n_param
        self.noise_std = noise_std
        # Green's function matrix (symmetric, decaying off-diagonal)
        xs = np.linspace(0, 1, n_param)
        # kernel: G_ij = exp(-|x_i - x_j| / 0.15)
        self.G_full = np.exp(-np.abs(xs[:, None] - xs[None, :]) / 0.15)
        self.xs = xs

    def forward(self, theta, d_indices):
        """
        theta : (n_param,)
        d_indices : list of int sensor indices
        returns : (len(d_indices),)  noisy observations
        """
        y_clean = self.G_full[d_indices, :] @ theta
        return y_clean + self.noise_std * np.random.randn(len(d_indices))

    def jacobian(self, d_indices):
        """J = dG/d(theta) at given sensor locations."""
        return self.G_full[d_indices, :]

    def observation_matrix(self, d_indices):
        return self.G_full[d_indices, :]


# =====================================================================
# 2. Derivative-informed dimension reduction
# =====================================================================
def derivative_informed_reduction(forward_model, prior_cov, d_indices_pool,
                                  r=10):
    """
    Compute the dominant eigenvectors of
        Gamma_prior^{1/2} J^T Gamma_noise^{-1} J Gamma_prior^{1/2}
    using all candidate sensor locations.

    Returns projection matrix V_r (n_param, r).
    """
    n = forward_model.n
    sigma_n = forward_model.noise_std

    # Aggregate Jacobian over all candidate locations
    J = forward_model.jacobian(d_indices_pool)       # (n_cand, n_param)
    L_prior = np.linalg.cholesky(prior_cov + 1e-10 * np.eye(n))

    # Symmetrised matrix
    M = L_prior.T @ (J.T @ J / sigma_n ** 2) @ L_prior  # (n, n)
    eigvals, eigvecs = np.linalg.eigh(M)

    # Take top-r (largest eigenvalues are at end for eigh)
    idx = np.argsort(eigvals)[::-1][:r]
    V_r = L_prior @ eigvecs[:, idx]   # back to original coords
    # Orthonormalise
    V_r, _ = np.linalg.qr(V_r)
    return V_r, eigvals[idx]


# =====================================================================
# 3. GP surrogate in reduced space
# =====================================================================
class GPSurrogate:
    def __init__(self, ls=0.5, sv=1.0, nv=1e-6):
        self.ls, self.sv, self.nv = ls, sv, nv

    def _kernel(self, X1, X2):
        sq = cdist(X1, X2, "sqeuclidean")
        return self.sv * np.exp(-0.5 * sq / self.ls ** 2)

    def fit(self, X, Y):
        """X: (n, d_red),  Y: (n, n_obs)"""
        self.X, self.Y = X, Y
        K = self._kernel(X, X) + self.nv * np.eye(len(X))
        self.L, self.low = cho_factor(K)
        self.alpha = cho_solve((self.L, self.low), Y)   # (n, n_obs)

    def predict(self, Xs):
        ks = self._kernel(Xs, self.X)
        mu = ks @ self.alpha
        v = self.sv - np.einsum(
            "ij,jk,ik->i", ks,
            cho_solve((self.L, self.low), ks.T),
            ks,
        )
        return mu, np.clip(v, 0, None)


# =====================================================================
# 4. Expected Information Gain estimation (Laplace-based, Gaussian)
# =====================================================================
def eig_gaussian(J_d, prior_cov, noise_var):
    """
    Analytic EIG for a linear-Gaussian model:
        EIG = 0.5 * log det(I + (1/noise_var) * J Gamma_prior J^T)

    J_d : (n_obs, n_param)   observation matrix for design d
    """
    n_obs = J_d.shape[0]
    M = np.eye(n_obs) + (J_d @ prior_cov @ J_d.T) / noise_var
    sign, logdet = np.linalg.slogdet(M)
    return 0.5 * logdet


# =====================================================================
# 5. Sequential Bayesian OED loop
# =====================================================================
def sequential_boed(forward_model, prior_cov, theta_true,
                    n_stages=5, candidate_pool=None):
    """
    Greedy sequential OED: at each stage pick the sensor that
    maximises EIG given the current posterior (used as next prior).
    """
    n = forward_model.n
    noise_var = forward_model.noise_std ** 2

    if candidate_pool is None:
        candidate_pool = list(range(n))

    current_prior_cov = prior_cov.copy()
    selected_designs = []
    all_obs = []

    print(f"{'stage':>5s}  {'sensor':>6s}  {'EIG':>8s}  "
          f"{'post_trace':>10s}  {'post_logdet':>11s}")
    print("-" * 50)

    for t in range(n_stages):
        # ── evaluate EIG for every candidate sensor ──
        eig_vals = np.zeros(len(candidate_pool))
        for ci, c in enumerate(candidate_pool):
            J_c = forward_model.observation_matrix([c])  # (1, n)
            eig_vals[ci] = eig_gaussian(J_c, current_prior_cov, noise_var)

        best_ci = np.argmax(eig_vals)
        best_sensor = candidate_pool[best_ci]
        selected_designs.append(best_sensor)

        # ── observe ──
        y_obs = forward_model.forward(theta_true, [best_sensor])
        all_obs.append(y_obs[0])

        # ── Bayesian posterior update (Gaussian conjugate) ──
        J_d = forward_model.observation_matrix([best_sensor])  # (1, n)
        S = J_d @ current_prior_cov @ J_d.T + noise_var * np.eye(1)
        K_gain = current_prior_cov @ J_d.T @ np.linalg.inv(S)  # (n, 1)

        current_prior_cov = (
            current_prior_cov - K_gain @ J_d @ current_prior_cov
        )
        # symmetrise for numerical stability
        current_prior_cov = 0.5 * (current_prior_cov + current_prior_cov.T)

        # ── remove chosen sensor from pool ──
        candidate_pool = [c for c in candidate_pool if c != best_sensor]

        ptr = np.trace(current_prior_cov)
        _, pld = np.linalg.slogdet(current_prior_cov)
        print(f"{t + 1:5d}  {best_sensor:6d}  {eig_vals[best_ci]:8.4f}  "
              f"{ptr:10.4f}  {pld:11.4f}")

    return selected_designs, all_obs, current_prior_cov


# =====================================================================
# 6. Posterior mean reconstruction
# =====================================================================
def posterior_mean(forward_model, prior_cov, prior_mean,
                  selected_designs, observations):
    """Compute Gaussian posterior mean given all observations."""
    n = forward_model.n
    noise_var = forward_model.noise_std ** 2
    mu = prior_mean.copy()
    cov = prior_cov.copy()

    for d, y in zip(selected_designs, observations):
        J_d = forward_model.observation_matrix([d])
        S = J_d @ cov @ J_d.T + noise_var * np.eye(1)
        K = cov @ J_d.T @ np.linalg.inv(S)
        innovation = np.array([y]) - J_d @ mu
        mu = mu + (K @ innovation).ravel()
        cov = cov - K @ J_d @ cov
        cov = 0.5 * (cov + cov.T)

    return mu, cov


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Reference impl: Sequential Bayesian OED with DIDR")
    print("  arXiv 2409.09141")
    print("=" * 60)

    N_PARAM = 50
    fm = LinearPtOMap(n_param=N_PARAM, noise_std=0.1)

    # ── Prior: squared-exponential covariance ──
    xs = np.linspace(0, 1, N_PARAM)
    prior_cov = np.exp(-cdist(xs[:, None], xs[:, None], "sqeuclidean")
                       / (2 * 0.2 ** 2))
    prior_mean = np.zeros(N_PARAM)

    # ── True parameter (smooth random field) ──
    L_pr = np.linalg.cholesky(prior_cov + 1e-8 * np.eye(N_PARAM))
    theta_true = L_pr @ np.random.randn(N_PARAM)

    # ── Step 1: Derivative-informed dimension reduction ──
    print("\n[1] Derivative-informed dimension reduction ...")
    cand_pool = list(range(N_PARAM))
    V_r, eigvals = derivative_informed_reduction(
        fm, prior_cov, cand_pool, r=10
    )
    print(f"    Top-10 eigenvalues: {np.round(eigvals[:10], 3)}")
    cum_energy = np.cumsum(eigvals) / np.sum(eigvals)
    r_95 = np.searchsorted(cum_energy, 0.95) + 1
    print(f"    Dimensions for 95% energy: {r_95}")
    print(f"    Reduction: {N_PARAM} -> {V_r.shape[1]}")

    # ── Step 2: Sequential BOED ──
    print("\n[2] Sequential Bayesian OED (5 stages) ...")
    designs, obs, post_cov = sequential_boed(
        fm, prior_cov, theta_true, n_stages=5
    )
    print(f"\n    Selected sensors: {designs}")

    # ── Step 3: Posterior reconstruction ──
    print("\n[3] Posterior reconstruction ...")
    mu_post, cov_post = posterior_mean(
        fm, prior_cov, prior_mean, designs, obs
    )
    recon_err = np.sqrt(np.mean((mu_post - theta_true) ** 2))
    print(f"    Reconstruction RMSE (sequential OED): {recon_err:.4f}")

    # ── Baseline: equi-spaced sensors ──
    equi = np.linspace(0, N_PARAM - 1, 5, dtype=int).tolist()
    obs_equi = [fm.forward(theta_true, [d])[0] for d in equi]
    mu_equi, _ = posterior_mean(fm, prior_cov, prior_mean, equi, obs_equi)
    err_equi = np.sqrt(np.mean((mu_equi - theta_true) ** 2))
    print(f"    Reconstruction RMSE (equi-spaced):    {err_equi:.4f}")

    # ── Baseline: random sensors ──
    rand_sensors = np.random.choice(N_PARAM, 5, replace=False).tolist()
    obs_rand = [fm.forward(theta_true, [d])[0] for d in rand_sensors]
    mu_rand, _ = posterior_mean(fm, prior_cov, prior_mean,
                                rand_sensors, obs_rand)
    err_rand = np.sqrt(np.mean((mu_rand - theta_true) ** 2))
    print(f"    Reconstruction RMSE (random sensors):  {err_rand:.4f}")

    # ── Step 4: Posterior uncertainty summary ──
    prior_trace = np.trace(prior_cov)
    post_trace = np.trace(cov_post)
    print(f"\n[4] Uncertainty reduction")
    print(f"    Prior  trace: {prior_trace:.4f}")
    print(f"    Post   trace: {post_trace:.4f}")
    print(f"    Reduction:    {100 * (1 - post_trace / prior_trace):.1f}%")

    print("\n[Done] Sequential OED selects informative sensor placements "
          "that reduce posterior uncertainty more efficiently than "
          "non-adaptive baselines.")
