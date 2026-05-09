#!/usr/bin/env python3
"""
GUIDe Reference Implementation (Simplified)
============================================
Generative and Uncertainty-Informed Inverse Design for
On-Demand Nonlinear Functional Responses.

arXiv: 2509.05641

This minimal implementation demonstrates:
  1. An ensemble-based forward surrogate with uncertainty quantification.
  2. MCMC-based inverse design sampling with uncertainty-informed likelihood.
  3. One-to-many inverse mapping: multiple designs yield similar responses.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist


# ---------------------------------------------------------------------------
# 1.  Synthetic nonlinear functional response
# ---------------------------------------------------------------------------

def true_response(x, t_grid):
    """
    True forward mapping: design parameters -> nonlinear response curve.

    Parameters
    ----------
    x : ndarray, shape (d,)
        Design parameters (d=3).
    t_grid : ndarray, shape (T,)
        Time/frequency grid points.

    Returns
    -------
    y : ndarray, shape (T,)
        Nonlinear functional response curve.
    """
    d = len(x)
    y = np.zeros_like(t_grid)
    # Superposition of nonlinear basis functions
    y += x[0] * np.sin(2 * np.pi * t_grid * (0.5 + 0.5 * x[1]))
    y += x[1] ** 2 * np.cos(np.pi * t_grid)
    y += 0.3 * x[2] * np.exp(-2 * (t_grid - 0.5 * (1 + x[0])) ** 2)
    # Add interaction terms
    y += 0.2 * x[0] * x[2] * np.sin(4 * np.pi * t_grid)
    return y


# ---------------------------------------------------------------------------
# 2.  Ensemble forward surrogate with uncertainty
# ---------------------------------------------------------------------------

class EnsembleSurrogate:
    """
    Ensemble of randomized linear models as a simplified surrogate.
    Each model uses random Fourier features (RFF) for nonlinearity.
    Provides mean prediction and uncertainty via ensemble disagreement.
    """

    def __init__(self, n_models=20, n_features=50, reg=1e-3, seed=42):
        self.n_models = n_models
        self.n_features = n_features
        self.reg = reg
        self.rng = np.random.RandomState(seed)
        self.models = []  # list of (W_rff, b_rff, weights)
        self.t_dim = None

    def _random_fourier_features(self, X, W, b):
        """Map X to RFF space: sqrt(2/D) * cos(X @ W + b)."""
        Z = np.cos(X @ W + b) * np.sqrt(2.0 / W.shape[1])
        return Z

    def fit(self, X, Y):
        """
        Fit ensemble of models.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Design parameters.
        Y : ndarray, shape (n, T)
            Response curves.
        """
        n, d = X.shape
        self.t_dim = Y.shape[1]
        self.models = []

        for i in range(self.n_models):
            # Bootstrap sample
            idx = self.rng.choice(n, size=n, replace=True)
            X_boot = X[idx]
            Y_boot = Y[idx]

            # Random Fourier features
            W_rff = self.rng.randn(d, self.n_features) * 1.5
            b_rff = self.rng.uniform(0, 2 * np.pi, size=self.n_features)
            Z = self._random_fourier_features(X_boot, W_rff, b_rff)

            # Ridge regression for each output dimension
            ZTZ = Z.T @ Z + self.reg * np.eye(self.n_features)
            weights = np.linalg.solve(ZTZ, Z.T @ Y_boot)

            self.models.append((W_rff, b_rff, weights))

    def predict(self, X):
        """
        Predict response with uncertainty.

        Returns
        -------
        mu : ndarray, shape (n, T)
            Mean prediction.
        sigma : ndarray, shape (n, T)
            Standard deviation (ensemble disagreement).
        """
        X = np.atleast_2d(X)
        preds = []
        for W_rff, b_rff, weights in self.models:
            Z = self._random_fourier_features(X, W_rff, b_rff)
            pred = Z @ weights
            preds.append(pred)

        preds = np.array(preds)  # (n_models, n, T)
        mu = np.mean(preds, axis=0)
        sigma = np.std(preds, axis=0)
        sigma = np.maximum(sigma, 1e-6)
        return mu, sigma


# ---------------------------------------------------------------------------
# 3.  MCMC inverse design with uncertainty-informed likelihood
# ---------------------------------------------------------------------------

def log_likelihood(x, y_target, surrogate, noise_std=0.05):
    """
    Uncertainty-informed log-likelihood.

    L(x | y*) = prod_t N(y*_t | mu_t(x), sigma_t(x)^2 + noise^2)
    """
    mu, sigma = surrogate.predict(x.reshape(1, -1))
    mu = mu[0]
    sigma = sigma[0]

    total_var = sigma ** 2 + noise_std ** 2
    total_std = np.sqrt(total_var)

    ll = np.sum(norm.logpdf(y_target, loc=mu, scale=total_std))
    return ll


def log_prior(x, bounds):
    """Uniform prior within bounds."""
    if np.all(x >= bounds[:, 0]) and np.all(x <= bounds[:, 1]):
        return 0.0  # log(1) = 0 for uniform
    return -np.inf


def mcmc_inverse_design(y_target, surrogate, bounds, n_chains=4,
                        n_samples=400, burn_in=200, proposal_std=0.08,
                        noise_std=0.05):
    """
    Metropolis-Hastings MCMC for inverse design.
    Multiple chains to capture multimodal posterior (one-to-many mapping).

    Returns
    -------
    all_samples : ndarray, shape (n_total, d)
    all_log_likes : ndarray
    accept_rates : list of float
    """
    d = bounds.shape[0]
    all_samples = []
    all_log_likes = []
    accept_rates = []

    for chain_id in range(n_chains):
        # Random initialization within bounds
        x_current = np.random.uniform(bounds[:, 0], bounds[:, 1])
        ll_current = log_likelihood(x_current, y_target, surrogate, noise_std)
        lp_current = log_prior(x_current, bounds)

        samples = []
        log_likes = []
        n_accept = 0
        total = n_samples + burn_in

        for step in range(total):
            # Propose
            x_prop = x_current + np.random.randn(d) * proposal_std
            lp_prop = log_prior(x_prop, bounds)

            if lp_prop == -np.inf:
                # Reject: out of bounds
                if step >= burn_in:
                    samples.append(x_current.copy())
                    log_likes.append(ll_current)
                continue

            ll_prop = log_likelihood(x_prop, y_target, surrogate, noise_std)

            # Acceptance ratio
            log_alpha = (ll_prop + lp_prop) - (ll_current + lp_current)
            if np.log(np.random.rand()) < log_alpha:
                x_current = x_prop
                ll_current = ll_prop
                lp_current = lp_prop
                n_accept += 1

            if step >= burn_in:
                samples.append(x_current.copy())
                log_likes.append(ll_current)

        all_samples.extend(samples)
        all_log_likes.extend(log_likes)
        accept_rates.append(n_accept / total)

    return np.array(all_samples), np.array(all_log_likes), accept_rates


# ---------------------------------------------------------------------------
# 4.  Uncertainty-informed design selection
# ---------------------------------------------------------------------------

def select_best_designs(samples, y_target, surrogate, n_select=10,
                        w_mse=0.7, w_unc=0.3):
    """
    Select designs balancing accuracy (MSE) and low uncertainty.

    score(x) = w_mse * MSE(f(x), y*) + w_unc * mean(sigma(x))
    """
    mu, sigma = surrogate.predict(samples)

    mse = np.mean((mu - y_target[None, :]) ** 2, axis=1)
    mean_unc = np.mean(sigma, axis=1)

    # Normalize
    mse_norm = (mse - mse.min()) / (mse.max() - mse.min() + 1e-10)
    unc_norm = (mean_unc - mean_unc.min()) / (mean_unc.max() - mean_unc.min() + 1e-10)

    score = w_mse * mse_norm + w_unc * unc_norm
    best_idx = np.argsort(score)[:n_select]

    return samples[best_idx], mse[best_idx], mean_unc[best_idx]


# ---------------------------------------------------------------------------
# 5.  Demo
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    d = 3  # design parameter dimension
    T = 30  # response curve discretization
    t_grid = np.linspace(0, 1, T)
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])

    print("=" * 65)
    print("GUIDe Reference Implementation")
    print("Uncertainty-Informed Inverse Design for Functional Responses")
    print("=" * 65)

    # --- Generate training data ---
    n_train = 200
    print(f"\n[1] Generating {n_train} training samples...")
    X_train = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_train, d))
    Y_train = np.array([true_response(x, t_grid) for x in X_train])
    print(f"    Design dim: {d}, Response dim: {T}")
    print(f"    Response range: [{Y_train.min():.3f}, {Y_train.max():.3f}]")

    # --- Train ensemble surrogate ---
    print(f"\n[2] Training ensemble surrogate ({20} models)...")
    surrogate = EnsembleSurrogate(n_models=20, n_features=50, reg=1e-3)
    surrogate.fit(X_train, Y_train)

    # Evaluate surrogate accuracy
    n_test = 50
    X_test = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_test, d))
    Y_test = np.array([true_response(x, t_grid) for x in X_test])
    Y_pred, Y_std = surrogate.predict(X_test)
    rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))
    mean_std = np.mean(Y_std)
    print(f"    Surrogate RMSE: {rmse:.4f}")
    print(f"    Mean predictive std: {mean_std:.4f}")

    # --- Define target response ---
    x_true = np.array([0.5, -0.3, 0.7])
    y_target = true_response(x_true, t_grid)
    print(f"\n[3] Target response generated from x_true = {x_true}")
    print(f"    (Note: inverse problem is one-to-many; multiple x may match)")

    # --- MCMC inverse design ---
    print(f"\n[4] Running MCMC inverse design (4 chains)...")
    samples, log_likes, accept_rates = mcmc_inverse_design(
        y_target, surrogate, bounds,
        n_chains=4, n_samples=400, burn_in=200,
        proposal_std=0.08, noise_std=0.05
    )
    print(f"    Total samples: {len(samples)}")
    print(f"    Acceptance rates: {[f'{r:.2%}' for r in accept_rates]}")

    # --- Select best designs ---
    print(f"\n[5] Selecting top designs (uncertainty-informed)...")
    best_designs, best_mse, best_unc = select_best_designs(
        samples, y_target, surrogate, n_select=8
    )

    print(f"\n    Top 8 designs:")
    print(f"    {'x[0]':>7s}  {'x[1]':>7s}  {'x[2]':>7s}  "
          f"{'MSE':>10s}  {'Mean_UQ':>10s}")
    print(f"    {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*10}")
    for i in range(len(best_designs)):
        x = best_designs[i]
        print(f"    {x[0]:7.3f}  {x[1]:7.3f}  {x[2]:7.3f}  "
              f"{best_mse[i]:10.6f}  {best_unc[i]:10.6f}")

    # --- Validate best designs via true forward model ---
    print(f"\n[6] Validation via true forward model:")
    for i, x in enumerate(best_designs[:5]):
        y_achieved = true_response(x, t_grid)
        true_mse = np.mean((y_achieved - y_target) ** 2)
        dist_to_true = np.linalg.norm(x - x_true)
        print(f"    Design {i+1}: true_MSE={true_mse:.6f}, "
              f"||x - x_true||={dist_to_true:.4f}")

    # --- Diversity analysis ---
    print(f"\n[7] Diversity of inverse solutions:")
    if len(best_designs) > 1:
        pairwise_dists = pdist(best_designs)
        print(f"    Mean pairwise distance: {np.mean(pairwise_dists):.4f}")
        print(f"    Max pairwise distance:  {np.max(pairwise_dists):.4f}")

    # Check if solutions cluster around multiple modes
    print(f"\n    Per-dimension statistics of all MCMC samples:")
    for dim in range(d):
        vals = samples[:, dim]
        print(f"      x[{dim}]: mean={np.mean(vals):.3f}, "
              f"std={np.std(vals):.3f}, "
              f"range=[{np.min(vals):.3f}, {np.max(vals):.3f}]")

    # --- Response curve comparison ---
    print(f"\n[8] Response curve fidelity (best design vs target):")
    x_best = best_designs[0]
    y_best_pred, y_best_std = surrogate.predict(x_best.reshape(1, -1))
    y_best_true = true_response(x_best, t_grid)

    max_err_pred = np.max(np.abs(y_best_pred[0] - y_target))
    max_err_true = np.max(np.abs(y_best_true - y_target))
    print(f"    Max |surrogate(x_best) - y_target|: {max_err_pred:.4f}")
    print(f"    Max |true(x_best) - y_target|:      {max_err_true:.4f}")
    print(f"    Mean surrogate uncertainty at x_best: {np.mean(y_best_std):.4f}")

    print("\n" + "=" * 65)
    print("Done. GUIDe demonstrates uncertainty-informed inverse design")
    print("handling one-to-many ambiguity with MCMC posterior sampling.")
    print("=" * 65)


if __name__ == "__main__":
    main()
