"""
Reference implementation of IP-SUR sequential design for surrogate modeling
in Bayesian inverse problems (arXiv:2402.16520).
"""

import warnings
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


# ---------------------------------------------------------------------------
# Forward model and observation model
# ---------------------------------------------------------------------------

def forward_model(theta):
    """1D-input, 2D-output toy forward model: polynomial plus sinusoidal components."""
    theta = np.asarray(theta, dtype=float).ravel()
    f1 = theta + 0.3 * np.sin(2.0 * theta)
    f2 = 0.5 * theta ** 2 - 0.1 * np.cos(4.0 * theta)
    return np.stack([f1, f2], axis=-1)   # shape (n, 2)


def log_likelihood_gaussian(y_obs, gp_means, gp_stds, noise_std):
    """Sum of per-output Gaussian log-likelihoods."""
    ll = 0.0
    for y, mu, sig in zip(y_obs, gp_means, gp_stds):
        total_var = sig ** 2 + noise_std ** 2
        ll += -0.5 * ((y - mu) ** 2 / total_var + np.log(2.0 * np.pi * total_var))
    return float(ll)


def log_prior_uniform(theta, lo, hi):
    """Log of uniform prior on [lo, hi]."""
    return 0.0 if lo <= float(theta) <= hi else -np.inf


# ---------------------------------------------------------------------------
# Grid-based posterior
# ---------------------------------------------------------------------------

def compute_posterior_weights(theta_grid, y_obs, gps, noise_std, lo, hi):
    """Evaluate unnormalised log-posterior on theta_grid and return normalised weights."""
    n = len(theta_grid)
    log_post = np.zeros(n)

    X_grid = theta_grid.reshape(-1, 1)
    # Pre-compute GP predictions for all outputs at once
    gp_preds = []
    for gp in gps:
        mu, sig = gp.predict(X_grid, return_std=True)
        gp_preds.append((mu, np.maximum(sig, 1e-8)))

    for i, th in enumerate(theta_grid):
        lp = log_prior_uniform(th, lo, hi)
        if not np.isfinite(lp):
            log_post[i] = -np.inf
            continue
        gp_means = [gp_preds[k][0][i] for k in range(len(gps))]
        gp_stds  = [gp_preds[k][1][i] for k in range(len(gps))]
        log_post[i] = lp + log_likelihood_gaussian(y_obs, gp_means, gp_stds, noise_std)

    log_post -= np.max(log_post)
    weights = np.exp(log_post)
    weights /= weights.sum() + 1e-300
    return weights


# ---------------------------------------------------------------------------
# GP surrogate
# ---------------------------------------------------------------------------

def build_gp():
    """Construct a GP regressor with Matern-5/2 kernel plus white noise."""
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=0.5, length_scale_bounds=(0.05, 5.0), nu=2.5)
        + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
    )
    return GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=3, normalize_y=True
    )


def fit_gps(X_design, Y_design):
    """Fit one GP per output dimension; return list of fitted GPs."""
    n_out = Y_design.shape[1]
    gps = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in range(n_out):
            gp = build_gp()
            gp.fit(X_design.reshape(-1, 1), Y_design[:, k])
            gps.append(gp)
    return gps


# ---------------------------------------------------------------------------
# IP-SUR acquisition
# ---------------------------------------------------------------------------

def _weighted_var_after_update(x_cand_scalar, gps, theta_grid, weights):
    """
    Compute weighted sum of GP posterior variances after adding virtual obs at x_cand.

    Applies the rank-1 variance reduction formula across all output GPs.
    """
    x_cand = np.array([[float(x_cand_scalar)]])
    X_grid = theta_grid.reshape(-1, 1)

    total_wvar = np.zeros(len(theta_grid))
    for gp in gps:
        _, std_grid = gp.predict(X_grid, return_std=True)
        var_grid = np.maximum(std_grid, 1e-8) ** 2

        # Cross-covariance k(x_grid, x_cand) using fitted kernel
        k_xc = gp.kernel_(X_grid, x_cand).ravel()

        _, std_cand = gp.predict(x_cand, return_std=True)
        var_cand = max(float(std_cand[0]) ** 2, 1e-10)

        var_updated = np.maximum(var_grid - k_xc ** 2 / var_cand, 0.0)
        total_wvar += var_updated

    return float(np.dot(weights, total_wvar))


def ip_sur_acquisition(x_arr, gps, theta_grid, weights):
    """IP-SUR criterion: posterior-weighted IMSPE after adding x as next design point."""
    return _weighted_var_after_update(float(x_arr[0]), gps, theta_grid, weights)


def select_next_point(gps, theta_grid, weights, lo, hi, n_restarts=12, seed=None):
    """Minimise IP-SUR via multi-start L-BFGS-B; return best x and criterion value."""
    rng = np.random.default_rng(seed)
    starts = rng.uniform(lo, hi, size=n_restarts)

    best_val = np.inf
    best_x = float(starts[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x0 in starts:
            res = minimize(
                ip_sur_acquisition,
                x0=np.array([float(x0)]),
                args=(gps, theta_grid, weights),
                method="L-BFGS-B",
                bounds=[(lo, hi)],
                options={"maxiter": 200, "ftol": 1e-12},
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = float(res.x[0])

    return best_x, best_val


# ---------------------------------------------------------------------------
# Sequential loop
# ---------------------------------------------------------------------------

def run_ip_sur(
    y_obs,
    noise_std,
    lo,
    hi,
    n_initial,
    n_sequential,
    n_grid=300,
    seed=0,
):
    """
    Run the IP-SUR sequential design loop for a Bayesian inverse problem.

    Returns history dict with design points, posteriors, MAP estimates, wIMSPE values,
    and the theta grid used for posterior approximation.
    """
    rng = np.random.default_rng(seed)

    theta_grid = np.linspace(lo, hi, n_grid)

    # Initial space-filling design
    X = np.linspace(lo + 0.05, hi - 0.05, n_initial) + rng.uniform(
        -0.02, 0.02, size=n_initial
    )
    X = np.clip(X, lo, hi)
    Y_clean = forward_model(X)
    Y = Y_clean + rng.normal(0, noise_std, size=Y_clean.shape)

    history = {
        "X": [], "Y": [],
        "weights": [], "map": [], "wIMSPE": [],
    }

    for it in range(n_sequential + 1):
        gps = fit_gps(X, Y)

        weights = compute_posterior_weights(
            theta_grid, y_obs, gps, noise_std, lo, hi
        )

        map_idx = int(np.argmax(weights))
        map_est = float(theta_grid[map_idx])

        # Current wIMSPE (sum over outputs)
        X_grid = theta_grid.reshape(-1, 1)
        wIMSPE = 0.0
        for gp in gps:
            _, std_grid = gp.predict(X_grid, return_std=True)
            wIMSPE += float(np.dot(weights, np.maximum(std_grid, 1e-8) ** 2))

        history["X"].append(X.copy())
        history["Y"].append(Y.copy())
        history["weights"].append(weights.copy())
        history["map"].append(map_est)
        history["wIMSPE"].append(wIMSPE)

        if it < n_sequential:
            x_next, _ = select_next_point(
                gps, theta_grid, weights, lo, hi, seed=int(seed) + it
            )
            x_next = float(x_next)
            y_next = forward_model(x_next).ravel() + rng.normal(
                0, noise_std, size=len(gps)
            )
            X = np.append(X, x_next)
            Y = np.vstack([Y, y_next])

            print(
                f"  Step {it + 1:2d}: x_new={x_next:+.4f}, "
                f"MAP={map_est:+.4f}, wIMSPE={wIMSPE:.4e}"
            )

    return history, theta_grid


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------

def posterior_mean(theta_grid, weights):
    """Compute posterior mean from discrete grid weights."""
    return float(np.dot(theta_grid, weights))


def credible_interval(theta_grid, weights, level=0.95):
    """Return equal-tailed credible interval at given level from grid posterior."""
    cdf = np.cumsum(weights)
    tail = (1.0 - level) / 2.0
    lo_idx = int(np.searchsorted(cdf, tail))
    hi_idx = int(np.searchsorted(cdf, 1.0 - tail))
    hi_idx = min(hi_idx, len(theta_grid) - 1)
    return float(theta_grid[lo_idx]), float(theta_grid[hi_idx])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    THETA_TRUE = 1.2
    NOISE_STD  = 0.05
    LO, HI     = -1.5, 2.5
    N_OBS      = 4          # repeated observations at theta_true
    N_INITIAL  = 6
    N_SEQUENTIAL = 10

    rng_obs = np.random.default_rng(7)
    raw_obs = forward_model(THETA_TRUE) + rng_obs.normal(0, NOISE_STD, size=(N_OBS, 2))
    y_obs = raw_obs.mean(axis=0)              # 2D summary statistic
    eff_noise = NOISE_STD / float(np.sqrt(N_OBS))

    print("=" * 64)
    print("  IP-SUR: Sequential GP Design for Bayesian Inverse Problems")
    print("=" * 64)
    print(f"  True parameter  theta* = {THETA_TRUE}")
    print(f"  Observed mean   y_obs  = [{y_obs[0]:.4f}, {y_obs[1]:.4f}]")
    print(f"  Forward model value    = {forward_model(THETA_TRUE).ravel()}")
    print(f"  Effective noise        = {eff_noise:.4f}")
    print(f"  Initial pts: {N_INITIAL},  IP-SUR steps: {N_SEQUENTIAL}")
    print()

    history, theta_grid = run_ip_sur(
        y_obs=y_obs,
        noise_std=eff_noise,
        lo=LO,
        hi=HI,
        n_initial=N_INITIAL,
        n_sequential=N_SEQUENTIAL,
        n_grid=300,
        seed=0,
    )

    final_weights = history["weights"][-1]
    post_mean = posterior_mean(theta_grid, final_weights)
    ci_lo, ci_hi = credible_interval(theta_grid, final_weights, level=0.95)
    final_map = history["map"][-1]

    print()
    print("=" * 64)
    print("  Final posterior (after IP-SUR)")
    print(f"  True parameter:  {THETA_TRUE:.4f}")
    print(f"  Posterior mean:  {post_mean:.4f}")
    print(f"  MAP estimate:    {final_map:.4f}")
    print(f"  95% CI:          [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Contains true?   {ci_lo <= THETA_TRUE <= ci_hi}")
    print()
    print("  wIMSPE trajectory (lower = less GP uncertainty near posterior):")
    w0 = history["wIMSPE"][0] + 1e-30
    for i, w in enumerate(history["wIMSPE"]):
        bar = "#" * max(1, int(40 * w / w0))
        print(f"    Step {i:2d}: {w:.3e}  |{bar}")
    print("=" * 64)
