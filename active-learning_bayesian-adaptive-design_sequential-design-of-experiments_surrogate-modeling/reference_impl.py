"""
Reference implementation of IP-SUR sequential design for surrogate modeling
in Bayesian inverse problems (arXiv:2402.16520).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


# ---------------------------------------------------------------------------
# Forward model and observation model
# ---------------------------------------------------------------------------

def forward_model(theta):
    """Toy 1D forward model: noisy sum of sin and polynomial."""
    theta = np.atleast_1d(theta)
    return np.sin(3 * theta) + 0.5 * theta**2 - 0.3 * theta


def log_likelihood(y_obs, gp_mean, gp_std, noise_std):
    """Gaussian log-likelihood of observed data given GP predictive mean/std."""
    total_var = gp_std**2 + noise_std**2
    return -0.5 * np.sum((y_obs - gp_mean) ** 2 / total_var + np.log(2 * np.pi * total_var))


def log_prior(theta, theta_min, theta_max):
    """Uniform log-prior over [theta_min, theta_max]."""
    if np.all(theta >= theta_min) and np.all(theta <= theta_max):
        return 0.0
    return -np.inf


# ---------------------------------------------------------------------------
# Posterior on a grid
# ---------------------------------------------------------------------------

def compute_posterior_grid(grid, y_obs, gp, noise_std, theta_min, theta_max):
    """Compute unnormalised log-posterior on a 1D grid and return normalised weights."""
    log_post = np.zeros(len(grid))
    for i, th in enumerate(grid):
        lp = log_prior(th, theta_min, theta_max)
        if not np.isfinite(lp):
            log_post[i] = -np.inf
            continue
        gp_mean, gp_std = gp.predict(th.reshape(-1, 1), return_std=True)
        gp_std = np.maximum(gp_std, 1e-10)
        ll = log_likelihood(y_obs, gp_mean[0], gp_std[0], noise_std)
        log_post[i] = lp + ll

    # Numerically stable softmax-style normalisation
    log_post -= np.max(log_post)
    weights = np.exp(log_post)
    weights /= weights.sum()
    return weights


# ---------------------------------------------------------------------------
# GP surrogate helpers
# ---------------------------------------------------------------------------

def build_gp():
    """Construct a GP regressor with Matern-like (RBF) kernel."""
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))
    return GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5, normalize_y=True)


def gp_prediction_grid(gp, x_grid):
    """Return GP posterior mean and std on a grid of input points."""
    mean, std = gp.predict(x_grid.reshape(-1, 1), return_std=True)
    return mean, np.maximum(std, 1e-10)


# ---------------------------------------------------------------------------
# IP-SUR acquisition function
# ---------------------------------------------------------------------------

def ip_sur_criterion(x_cand, gp, x_grid, posterior_weights):
    """
    IP-SUR: weighted IMSPE reduction after adding candidate point x_cand.

    Minimise the expected weighted integrated mean square prediction error
    (wIMSPE) of the updated GP when x_cand is added to the design.
    """
    x_cand = np.atleast_1d(x_cand).reshape(-1, 1)

    # Current GP predictive variance on grid
    _, std_current = gp_prediction_grid(gp, x_grid)
    var_current = std_current**2

    # Kernel function (from fitted GP)
    kernel = gp.kernel_
    X_train = gp.X_train_

    # k(x_grid, x_cand): covariance between grid and candidate
    k_grid_cand = kernel(x_grid.reshape(-1, 1), x_cand).ravel()

    # k(x_cand, X_train): covariance between candidate and training set
    k_cand_train = kernel(x_cand, X_train).ravel()

    # K(X_train, X_train) + noise I  (approximated via GP's alpha_ & L_)
    # Efficient variance reduction formula:
    # delta_var(x_grid) = k(x_grid, x_cand)^2 / [k(x_cand,x_cand) + sigma_n^2]
    # where sigma_n^2 comes from the noise term in the GP
    k_cand_cand = kernel(x_cand, x_cand)[0, 0]

    # Posterior variance at x_cand (denominator of the update)
    _, std_cand = gp.predict(x_cand, return_std=True)
    var_cand = float(std_cand[0]) ** 2 + 1e-10

    # Variance reduction at each grid point (rank-1 update approximation)
    delta_var = k_grid_cand**2 / var_cand

    # Updated variance (clipped to be non-negative)
    var_updated = np.maximum(var_current - delta_var, 0.0)

    # Weighted IMSPE after adding x_cand
    wIMSPE = np.sum(posterior_weights * var_updated)
    return wIMSPE


def select_next_point(gp, x_grid, posterior_weights, x_min, x_max, n_restarts=10):
    """Minimise IP-SUR criterion over candidate space via multi-start optimisation."""
    best_val = np.inf
    best_x = None

    # Random starting points
    starts = np.random.uniform(x_min, x_max, size=n_restarts)

    for x0 in starts:
        res = minimize(
            ip_sur_criterion,
            x0=np.array([x0]),
            args=(gp, x_grid, posterior_weights),
            method="L-BFGS-B",
            bounds=[(x_min, x_max)],
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x[0]

    return best_x, best_val


# ---------------------------------------------------------------------------
# Sequential Bayesian inverse problem loop
# ---------------------------------------------------------------------------

def run_ip_sur(
    theta_true,
    y_obs,
    noise_std,
    x_min,
    x_max,
    n_initial,
    n_sequential,
    n_grid=200,
    seed=0,
):
    """
    Run the IP-SUR sequential design loop for a 1D Bayesian inverse problem.

    Returns dict with design points, GP fits, posteriors, and MAP estimates.
    """
    rng = np.random.default_rng(seed)

    # Parameter grid for posterior evaluation
    theta_grid = np.linspace(x_min, x_max, n_grid)

    # Initial space-filling design (Latin-hypercube-like via uniform quantiles)
    X_design = np.linspace(x_min + 0.05, x_max - 0.05, n_initial) + rng.uniform(
        -0.05, 0.05, size=n_initial
    )
    X_design = np.clip(X_design, x_min, x_max)
    Y_design = forward_model(X_design) + rng.normal(0, noise_std, size=n_initial)

    history = {
        "X_design": [],
        "Y_design": [],
        "posterior_weights": [],
        "map_estimates": [],
        "wIMSPE": [],
    }

    gp = build_gp()

    for iteration in range(n_sequential + 1):
        # Fit GP surrogate to current design
        gp.fit(X_design.reshape(-1, 1), Y_design)

        # Compute posterior weights on grid
        weights = compute_posterior_grid(theta_grid, y_obs, gp, noise_std, x_min, x_max)

        # MAP estimate
        map_idx = np.argmax(weights)
        map_est = theta_grid[map_idx]

        # Current wIMSPE (before adding next point)
        _, std_grid = gp_prediction_grid(gp, theta_grid)
        current_wIMSPE = np.sum(weights * std_grid**2)

        history["X_design"].append(X_design.copy())
        history["Y_design"].append(Y_design.copy())
        history["posterior_weights"].append(weights.copy())
        history["map_estimates"].append(map_est)
        history["wIMSPE"].append(current_wIMSPE)

        if iteration < n_sequential:
            # Select next design point via IP-SUR
            x_next, acq_val = select_next_point(
                gp, theta_grid, weights, x_min, x_max
            )
            x_next = float(x_next)
            y_next = float(np.squeeze(forward_model(x_next))) + float(rng.normal(0, noise_std))

            X_design = np.append(X_design, x_next)
            Y_design = np.append(Y_design, y_next)

            print(
                f"  Iter {iteration + 1:2d}: x_next={x_next:.4f}, "
                f"y_next={y_next:.4f}, MAP={map_est:.4f}, wIMSPE={current_wIMSPE:.5f}"
            )

    return history, theta_grid


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def posterior_credible_interval(theta_grid, weights, alpha=0.95):
    """Return equal-tailed credible interval at level alpha from grid-based posterior."""
    cdf = np.cumsum(weights)
    lo = theta_grid[np.searchsorted(cdf, (1 - alpha) / 2)]
    hi = theta_grid[np.searchsorted(cdf, 1 - (1 - alpha) / 2)]
    return lo, hi


def posterior_mean(theta_grid, weights):
    """Compute posterior mean from grid weights."""
    return np.sum(theta_grid * weights)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    # Problem setup
    THETA_TRUE = 1.2          # true (unknown) parameter
    NOISE_STD = 0.1           # observational noise
    THETA_MIN, THETA_MAX = -2.0, 3.0
    N_OBS = 3                  # number of repeated observations at theta_true
    N_INITIAL = 5              # initial space-filling points
    N_SEQUENTIAL = 12          # sequential IP-SUR iterations

    # Generate observations
    rng_obs = np.random.default_rng(7)
    y_obs = forward_model(THETA_TRUE) + rng_obs.normal(0, NOISE_STD, size=N_OBS)
    y_obs_scalar = float(np.mean(y_obs))   # use mean as summary statistic
    effective_noise = NOISE_STD / np.sqrt(N_OBS)

    print("=" * 60)
    print("IP-SUR Sequential Design for Bayesian Inverse Problems")
    print(f"  True parameter: theta* = {THETA_TRUE}")
    print(f"  Observed mean:  y_obs  = {y_obs_scalar:.4f}")
    print(f"  Effective noise std:   = {effective_noise:.4f}")
    print("=" * 60)
    print(f"Initial design: {N_INITIAL} points, then {N_SEQUENTIAL} IP-SUR steps")
    print()

    history, theta_grid = run_ip_sur(
        theta_true=THETA_TRUE,
        y_obs=y_obs_scalar,
        noise_std=effective_noise,
        x_min=THETA_MIN,
        x_max=THETA_MAX,
        n_initial=N_INITIAL,
        n_sequential=N_SEQUENTIAL,
        n_grid=300,
        seed=0,
    )

    # Final posterior summary
    final_weights = history["posterior_weights"][-1]
    post_mean = posterior_mean(theta_grid, final_weights)
    ci_lo, ci_hi = posterior_credible_interval(theta_grid, final_weights, alpha=0.95)
    final_map = history["map_estimates"][-1]

    print()
    print("=" * 60)
    print("Final posterior summary (after IP-SUR)")
    print(f"  True parameter:   {THETA_TRUE:.4f}")
    print(f"  Posterior mean:   {post_mean:.4f}")
    print(f"  MAP estimate:     {final_map:.4f}")
    print(f"  95% CI:           [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Contains true?:   {ci_lo <= THETA_TRUE <= ci_hi}")
    print()
    print("wIMSPE reduction over iterations:")
    wimse_init = history["wIMSPE"][0]
    for i, w in enumerate(history["wIMSPE"]):
        bar_len = int(40 * w / wimse_init)
        bar = "#" * bar_len
        print(f"  Iter {i:2d}: {w:.5f}  |{bar}")
    print("=" * 60)
