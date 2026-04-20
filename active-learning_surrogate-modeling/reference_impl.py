"""
Reference implementation: Surrogate-Based Bayesian Inference with Active Learning
(arXiv:2603.13646)

Core idea: GP surrogate replaces expensive forward model in Bayesian inference.
Active learning sequentially refines the surrogate in posterior-relevant regions.
Surrogate uncertainty is propagated into the posterior.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


# ---------------------------------------------------------------------------
# Gaussian Process surrogate
# ---------------------------------------------------------------------------

class GPSurrogate:
    def __init__(self, length_scale=0.5, noise=1e-4):
        self.ls = length_scale
        self.noise = noise
        self.X = self.y = None

    def _kernel(self, X1, X2):
        sq = np.sum(X1**2, 1, keepdims=True) - 2 * X1 @ X2.T + np.sum(X2**2, 1)
        return np.exp(-0.5 * sq / self.ls**2)

    def fit(self, X, y):
        self.X, self.y = X.copy(), y.copy()
        K = self._kernel(X, X) + self.noise * np.eye(len(X))
        self.L = cho_factor(K)
        self.alpha = cho_solve(self.L, y)

    def predict(self, Xs, return_cov=False):
        Ks = self._kernel(Xs, self.X)
        mu = Ks @ self.alpha
        v = cho_solve(self.L, Ks.T)
        var = np.maximum(1.0 - np.sum(Ks.T * v, 0), 0.0)
        if return_cov:
            cov = self._kernel(Xs, Xs) - Ks @ v
            return mu, var, cov
        return mu, var


# ---------------------------------------------------------------------------
# Expensive forward model (to be replaced by surrogate)
# ---------------------------------------------------------------------------

def true_forward(theta, n_obs=5):
    """Toy nonlinear forward model: theta -> observables."""
    x = np.linspace(0, 1, n_obs)
    return np.sin(2 * np.pi * theta[0] * x) * np.exp(-theta[1] * x)


# ---------------------------------------------------------------------------
# Bayesian inference with surrogate
# ---------------------------------------------------------------------------

def log_likelihood(y_obs, y_pred, sigma=0.1):
    return -0.5 * np.sum((y_obs - y_pred) ** 2) / sigma**2


def log_prior(theta, bounds):
    if np.all(theta >= bounds[:, 0]) and np.all(theta <= bounds[:, 1]):
        return 0.0
    return -np.inf


def surrogate_posterior_samples(gp_surrogates, y_obs, bounds, n_samples=5000, sigma=0.1):
    """
    MCMC (Metropolis-Hastings) using GP surrogate predictions.
    Includes surrogate uncertainty via sampling from GP predictive.
    """
    d = bounds.shape[0]
    theta = bounds.mean(axis=1)
    log_p = -np.inf
    samples = []
    proposal_scale = 0.05 * (bounds[:, 1] - bounds[:, 0])

    for i in range(n_samples + 1000):  # 1000 burn-in
        theta_prop = theta + proposal_scale * np.random.randn(d)

        lp = log_prior(theta_prop, bounds)
        if lp == -np.inf:
            continue

        # surrogate prediction with uncertainty
        y_pred = np.zeros(len(y_obs))
        y_unc = np.zeros(len(y_obs))
        for j, gp in enumerate(gp_surrogates):
            mu_j, var_j = gp.predict(theta_prop.reshape(1, -1))
            # sample from GP predictive to propagate surrogate uncertainty
            y_pred[j] = mu_j[0] + np.sqrt(var_j[0]) * np.random.randn()
            y_unc[j] = var_j[0]

        ll = log_likelihood(y_obs, y_pred, sigma=np.sqrt(sigma**2 + y_unc.mean()))
        log_p_prop = lp + ll

        if np.log(np.random.rand() + 1e-12) < log_p_prop - log_p:
            theta = theta_prop
            log_p = log_p_prop

        if i >= 1000:
            samples.append(theta.copy())

    return np.array(samples)


# ---------------------------------------------------------------------------
# Active learning: posterior-weighted integrated variance reduction (IVAR)
# ---------------------------------------------------------------------------

def acquisition_ivar(gp_surrogates, candidates, posterior_samples):
    """
    Select the candidate that maximally reduces GP variance weighted by
    the current posterior.
    """
    scores = np.zeros(len(candidates))
    for j, gp in enumerate(gp_surrogates):
        _, var = gp.predict(candidates)
        # weight by posterior density (kernel density estimate)
        for s in posterior_samples[::10]:  # subsample for speed
            dist = np.sum((candidates - s) ** 2, axis=1)
            w = np.exp(-dist / (2 * 0.1**2))
            scores += w * var
    return scores


# ---------------------------------------------------------------------------
# Main active learning loop
# ---------------------------------------------------------------------------

def active_bayesian_inference(y_obs, bounds, n_init=10, n_active=15, sigma=0.1):
    d = bounds.shape[0]
    n_obs = len(y_obs)

    # initial space-filling design (Latin hypercube)
    X_train = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(n_init, d)
    Y_train = np.array([true_forward(x, n_obs) for x in X_train])

    for step in range(n_active):
        # fit one GP per observable dimension
        gps = []
        for j in range(n_obs):
            gp = GPSurrogate(length_scale=0.3)
            gp.fit(X_train, Y_train[:, j])
            gps.append(gp)

        # posterior samples via surrogate
        samples = surrogate_posterior_samples(gps, y_obs, bounds, n_samples=2000, sigma=sigma)

        if len(samples) < 10:
            print(f"  step {step+1}: insufficient samples, adding random point")
            x_new = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(d)
        else:
            # active learning: select next evaluation point
            n_cand = 200
            candidates = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(n_cand, d)
            scores = acquisition_ivar(gps, candidates, samples)
            x_new = candidates[np.argmax(scores)]

        y_new = true_forward(x_new, n_obs)
        X_train = np.vstack([X_train, x_new.reshape(1, -1)])
        Y_train = np.vstack([Y_train, y_new.reshape(1, -1)])

        if (step + 1) % 5 == 0:
            post_mean = samples.mean(axis=0) if len(samples) > 0 else np.full(d, np.nan)
            print(f"  step {step+1}/{n_active}  |train|={len(X_train)}  "
                  f"posterior_mean={post_mean}")

    # final posterior
    gps_final = []
    for j in range(n_obs):
        gp = GPSurrogate(length_scale=0.3)
        gp.fit(X_train, Y_train[:, j])
        gps_final.append(gp)
    final_samples = surrogate_posterior_samples(gps_final, y_obs, bounds, n_samples=5000, sigma=sigma)
    return final_samples, X_train


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    theta_true = np.array([1.5, 0.8])
    bounds = np.array([[0.5, 3.0], [0.1, 2.0]])
    y_obs = true_forward(theta_true) + 0.1 * np.random.randn(5)

    print("Running surrogate-based Bayesian inference with active learning...")
    samples, X_train = active_bayesian_inference(y_obs, bounds, n_init=8, n_active=15)

    print(f"\nTrue theta:      {theta_true}")
    print(f"Posterior mean:  {samples.mean(axis=0)}")
    print(f"Posterior std:   {samples.std(axis=0)}")
    print(f"Total forward model evaluations: {len(X_train)}")
