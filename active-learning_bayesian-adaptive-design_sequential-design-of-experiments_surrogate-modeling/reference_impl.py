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
    """Toy 1D forward model: sin plus quadratic."""
    theta = np.asarray(theta, dtype=float).ravel()
    return np.sin(3.0 * theta) + 0.5 * theta ** 2 - 0.3 * theta


def log_likelihood(y_obs, gp_mean, gp_std, noise_std):
    """Gaussian log-likelihood of scalar observation given GP predictive mean/std."""
    total_var = gp_std ** 2 + noise_std ** 2
    return -0.5 * ((y_obs - gp_mean) ** 2 / total_var + np.log(2.0 * np.pi * total_var))


def log_prior_uniform(theta, lo, hi):
    """Log of uniform prior on [lo, hi]."""
    return 0.0 if lo <= float(theta) <= hi else -np.inf


# ---------------------------------------------------------------------------
# Grid-based posterior
# ---------------------------------------------------------------------------

def compute_posterior_weights(theta_grid, y_obs, gp, noise_std, lo, hi):
    """Evaluate unnormalised log-posterior on theta_grid and return softmax weights."""
    gp_mean, gp_std = gp.predict(theta_grid.reshape(-1, 1), return_std=True)
    gp_std = np.maximum(gp_std, 1e-8)

    log_post = np.array([
        log_prior_uniform(th, lo, hi) + log_likelihood(y_obs, mu, sig, noise_std)
        for th, mu, sig in zip(theta_grid, gp_mean, gp_std)
    ])

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
        kernel=kernel, n_restarts_optimizer=5, normalize_y=True
    )


# ---------------------------------------------------------------------------
# IP-SUR acquisition
# ---------------------------------------------------------------------------

def _posterior_variance_after_update(x_cand_val, gp, x_grid):
    """
    Compute the GP posterior variance on x_grid after a virtual observation at x_cand.

    Uses the rank-1 (Sherman-Morrison) update: var_new(x) = var(x) - k(x,c)^2 / var(c).
    """
    x_cand = np.atleast_1d(x_cand_val).ravel()[:1].reshape(1, 1)

    mean_grid, std_grid = gp.predict(x_grid.reshape(-1, 1), return_std=True)
    var_grid = np.maximum(std_grid, 1e-8) ** 2

    k_grid_cand = gp.kernel_(x_grid.reshape(-1, 1), x_cand).ravel()

    _, std_cand = gp.predict(x_cand, return_std=True)
    var_cand = max(float(std_cand[0]) ** 2, 1e-10)

    var_updated = np.maximum(var_grid - k_grid_cand ** 2 / var_cand, 0.0)
    return var_updated


def ip_sur_acquisition(x_cand_val, gp, x_grid, posterior_weights):
    """
    IP-SUR criterion: posterior-weighted IMSPE after adding candidate x_cand.

    Lower is better; minimise this to select the next design point.
    """
    var_updated = _posterior_variance_after_update(x_cand_val, gp, x_grid)
    return float(np.dot(posterior_weights, var_updated))


def select_next_point(gp, x_grid, posterior_weights, lo, hi, n_restarts=15):
    """Minimise IP-SUR via multi-start L-BFGS-B; return best candidate and value."""
    rng = np.random.default_rng()
    starts = rng.uniform(lo, hi, size=n_restarts)

    best_val = np.inf
    best_x = float(starts[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x0 in starts:
            res = minimize(
                ip_sur_acquisition,
                x0=np.array([float(x0)]),
                args=(gp, x_grid, posterior_weights),
                method="L-BFGS-B",
                bounds=[(lo, hi)],
                options={"maxiter": 200, "ftol": 1e-10},
            )
            if res.success and res.fun < best_val:
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
    Run IP-SUR sequential design loop for a 1D Bayesian inverse problem.

    Returns history dict with design points, posteriors, MAP estimates, wIMSPE values.
    """
    rng = np.random.default_rng(seed)

    theta_grid = np.linspace(lo, hi, n_grid)

    # Initial design: uniform spread with small jitter
    X = np.linspace(lo + 0.1, hi - 0.1, n_initial) + rng.uniform(
        -0.05, 0.05, size=n_initial
    )
    X = np.clip(X, lo, hi)
    Y = forward_model(X) + rng.normal(0, noise_std, size=n_initial)

    history = {
        "X": [],
        "Y": [],
        "weights": [],
        "map": [],
        "wIMSPE": [],
    }

    gp = build_gp()

    for it in range(n_sequential + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X.reshape(-1, 1), Y)

        weights = compute_posterior_weights(theta_grid, y_obs, gp, noise_std, lo, hi)

        map_est = float(theta_grid[np.argmax(weights)])

        _, std_grid = gp.predict(theta_grid.reshape(-1, 1), return_std=True)
        wIMSPE = float(np.dot(weights, np.maximum(std_grid, 1e-8) ** 2))

        history["X"].append(X.copy())
        history["Y"].append(Y.copy())
        history["weights"].append(weights.copy())
        history["map"].append(map_est)
        history["wIMSPE"].append(wIMSPE)

        if it < n_sequential:
            x_next, _ = select_next_point(gp, theta_grid, weights, lo, hi)
            y_next = float(np.squeeze(forward_model(x_next))) + float(
                rng.normal(0, noise_std)
            )
            X = np.append(X, x_next)
            Y = np.append(Y, y_next)

            print(
                f"  Step {it + 1:2d}: x_new={x_next:+.4f}, "
                f"y_new={y_next:+.4f}, MAP={map_est:+.4f}, wIMSPE={wIMSPE:.4e}"
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
    lo = float(theta_grid[np.searchsorted(cdf, tail)])
    hi = float(theta_grid[np.searchsorted(cdf, 1.0 - tail)])
    return lo, hi


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    THETA_TRUE = 1.2
    NOISE_STD = 0.15
    LO, HI = -2.0, 3.0
    N_OBS = 5
    N_INITIAL = 6
    N_SEQUENTIAL = 12

    rng_obs = np.random.default_rng(99)
    obs = forward_model(THETA_TRUE) + rng_obs.normal(0, NOISE_STD, size=N_OBS)
    y_obs = float(np.mean(obs))
    eff_noise = NOISE_STD / float(np.sqrt(N_OBS))

    print("=" * 62)
    print("  IP-SUR: Sequential GP Design for Bayesian Inverse Problems")
    print("=" * 62)
    print(f"  True parameter  theta* = {THETA_TRUE}")
    print(f"  Observed mean   y_obs  = {y_obs:.4f}")
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
    print("=" * 62)
    print("  Final posterior (after IP-SUR)")
    print(f"  True parameter:  {THETA_TRUE:.4f}")
    print(f"  Posterior mean:  {post_mean:.4f}")
    print(f"  MAP estimate:    {final_map:.4f}")
    print(f"  95% CI:          [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Contains true?   {ci_lo <= THETA_TRUE <= ci_hi}")
    print()
    print("  wIMSPE trajectory:")
    w0 = history["wIMSPE"][0]
    for i, w in enumerate(history["wIMSPE"]):
        bar = "#" * int(40 * w / (w0 + 1e-30))
        print(f"    Step {i:2d}: {w:.3e}  |{bar}")
    print("=" * 62)
