"""
Reference implementation: Bayesian Inverse Design in CFD via Neural Operator Surrogate
(arXiv:2605.26059)

Core idea: Embed a lightweight neural-operator surrogate into an MCMC loop to
perform Bayesian inverse design of a quasi-1D nozzle geometry from sparse flow
observations, quantifying posterior uncertainty over shape parameters.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Quasi-1D nozzle physics (forward model)
# ---------------------------------------------------------------------------

def nozzle_area(x, ctrl_pts):
    """
    Cubic-B-spline-like nozzle area profile from control points.
    x: (M,) axial positions in [0, 1]
    ctrl_pts: (K,) area values at uniform knots
    Returns: (M,) area profile A(x)
    """
    K = len(ctrl_pts)
    knots = np.linspace(0, 1, K)
    A = np.interp(x, knots, ctrl_pts)
    return np.clip(A, 0.1, 10.0)


def quasi_1d_euler(ctrl_pts, x_grid, gamma=1.4):
    """
    Isentropic quasi-1D nozzle flow.
    Returns pressure ratio p/p0 at each x_grid location.
    """
    A = nozzle_area(x_grid, ctrl_pts)
    A_star = A.min()
    A_ratio = A / A_star

    # subsonic branch: iterative area-Mach relation
    M = np.ones_like(A_ratio) * 0.5
    for _ in range(50):
        f = ((2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M**2))**((gamma + 1) / (2 * (gamma - 1))) - A_ratio
        df = ((2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M**2))**((gamma + 1) / (2 * (gamma - 1)) - 1) \
             * (2 / (gamma + 1)) * (gamma - 1) / 2 * 2 * M * (gamma + 1) / (2 * (gamma - 1))
        df = np.where(np.abs(df) < 1e-12, 1e-12, df)
        M = M - f / df
        M = np.clip(M, 0.01, 0.99)

    p_ratio = (1 + (gamma - 1) / 2 * M**2)**(- gamma / (gamma - 1))
    return p_ratio


# ---------------------------------------------------------------------------
# Simple neural-operator surrogate (MLP approximation)
# ---------------------------------------------------------------------------

class NeuralOperatorSurrogate:
    """Two-layer MLP surrogate: ctrl_pts -> pressure ratio at observation locs."""

    def __init__(self, n_ctrl, n_obs, hidden=64):
        s = np.sqrt(2.0 / n_ctrl)
        self.W1 = np.random.randn(n_ctrl, hidden) * s
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, n_obs) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(n_obs)

    def forward(self, ctrl_pts):
        """ctrl_pts: (n_ctrl,) -> (n_obs,) predicted pressure ratios."""
        h = np.maximum(ctrl_pts @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2

    def forward_batch(self, X):
        """X: (B, n_ctrl) -> (B, n_obs)."""
        h = np.maximum(X @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2

    def fit(self, X, Y, epochs=1000, lr=1e-3):
        for ep in range(epochs):
            pred = self.forward_batch(X)
            err = pred - Y
            loss = 0.5 * np.mean(err**2)

            grad_out = err / len(X)
            h = np.maximum(X @ self.W1 + self.b1, 0)
            self.W2 -= lr * (h.T @ grad_out)
            self.b2 -= lr * grad_out.sum(axis=0)
            grad_h = grad_out @ self.W2.T
            grad_h[h <= 0] = 0
            self.W1 -= lr * (X.T @ grad_h)
            self.b1 -= lr * grad_h.sum(axis=0)

            if (ep + 1) % 500 == 0:
                print(f"  surrogate epoch {ep+1}  loss={loss:.6f}")


# ---------------------------------------------------------------------------
# MCMC (Metropolis-Hastings) for Bayesian inverse design
# ---------------------------------------------------------------------------

def log_likelihood(obs, pred, sigma_obs):
    return -0.5 * np.sum(((obs - pred) / sigma_obs)**2)


def log_prior(ctrl_pts, mu_prior, sigma_prior):
    return -0.5 * np.sum(((ctrl_pts - mu_prior) / sigma_prior)**2)


def mcmc_inverse_design(surrogate, y_obs, sigma_obs,
                        mu_prior, sigma_prior, n_ctrl,
                        n_samples=5000, proposal_std=0.05):
    """
    Metropolis-Hastings MCMC with neural-operator surrogate in the loop.
    """
    theta = mu_prior.copy()
    log_p_current = log_likelihood(y_obs, surrogate.forward(theta), sigma_obs) \
                  + log_prior(theta, mu_prior, sigma_prior)

    samples = np.zeros((n_samples, n_ctrl))
    accept_count = 0

    for i in range(n_samples):
        theta_prop = theta + np.random.randn(n_ctrl) * proposal_std
        theta_prop = np.clip(theta_prop, 0.3, 5.0)

        pred_prop = surrogate.forward(theta_prop)
        log_p_prop = log_likelihood(y_obs, pred_prop, sigma_obs) \
                   + log_prior(theta_prop, mu_prior, sigma_prior)

        if np.log(np.random.rand()) < log_p_prop - log_p_current:
            theta = theta_prop
            log_p_current = log_p_prop
            accept_count += 1

        samples[i] = theta

    accept_rate = accept_count / n_samples
    return samples, accept_rate


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    np.random.seed(42)

    n_ctrl = 7
    n_grid = 50
    x_grid = np.linspace(0, 1, n_grid)

    # observation locations (sparse)
    obs_idx = np.array([5, 15, 25, 35, 45])
    n_obs = len(obs_idx)

    # ground-truth nozzle: converging-diverging
    true_ctrl = np.array([2.0, 1.5, 1.0, 0.6, 1.0, 1.5, 2.0])
    true_pressure = quasi_1d_euler(true_ctrl, x_grid)
    y_obs = true_pressure[obs_idx] + np.random.randn(n_obs) * 0.01
    sigma_obs = 0.01

    # generate training data for surrogate
    print("Generating training data for surrogate...")
    N_train = 500
    X_train = np.random.uniform(0.3, 3.0, (N_train, n_ctrl))
    Y_train = np.zeros((N_train, n_obs))
    for i in range(N_train):
        pr = quasi_1d_euler(X_train[i], x_grid)
        Y_train[i] = pr[obs_idx]

    # train surrogate
    print("Training neural-operator surrogate...")
    surrogate = NeuralOperatorSurrogate(n_ctrl, n_obs, hidden=64)
    surrogate.fit(X_train, Y_train, epochs=2000, lr=5e-3)

    # verify surrogate accuracy
    pred_true = surrogate.forward(true_ctrl)
    actual = true_pressure[obs_idx]
    rel_err = np.mean(np.abs(pred_true - actual) / (np.abs(actual) + 1e-8))
    print(f"Surrogate relative error on true ctrl: {rel_err:.4f}")

    # MCMC inverse design
    print("\nRunning MCMC with surrogate...")
    mu_prior = np.ones(n_ctrl) * 1.5
    sigma_prior = np.ones(n_ctrl) * 1.0

    samples, accept_rate = mcmc_inverse_design(
        surrogate, y_obs, sigma_obs,
        mu_prior, sigma_prior, n_ctrl,
        n_samples=5000, proposal_std=0.08
    )
    print(f"Acceptance rate: {accept_rate:.2%}")

    # posterior summary (burn-in first 1000)
    burn = 1000
    post = samples[burn:]
    post_mean = post.mean(axis=0)
    post_std = post.std(axis=0)

    print(f"\nTrue ctrl pts:      {true_ctrl}")
    print(f"Posterior mean:     {np.round(post_mean, 3)}")
    print(f"Posterior std:      {np.round(post_std, 3)}")
    print(f"Max abs error:      {np.max(np.abs(post_mean - true_ctrl)):.3f}")


if __name__ == "__main__":
    demo()
