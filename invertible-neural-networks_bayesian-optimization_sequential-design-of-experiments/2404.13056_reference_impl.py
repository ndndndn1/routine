"""
Reference implementation: Variational Bayesian Optimal Experimental Design
with Normalizing Flows (arXiv:2404.13056 -- Dong, Jacobsen, Khalloufi,
Akram, Liu, Duraisamy, Huan, 2024)

Core idea: Use a coupling-layer-based normalizing flow as the variational
distribution q(theta|y,d) to estimate a lower bound on the expected
information gain (EIG).  Optimise the design d to maximise EIG.

This demo implements:
  1. Affine coupling layers (numpy-only) forming a conditional normalizing flow
  2. EIG lower bound estimation via Monte Carlo
  3. Grid search over 1D design variable d
  4. Sequential OED: pick designs one at a time, updating the prior

Synthetic problem: y = theta * sin(d * theta) + noise
  - theta ~ N(0, 1)  (parameter to infer)
  - d in [0.5, 5.0]  (design variable = frequency)
  - noise ~ N(0, sigma^2)

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.stats import norm


# -----------------------------------------------------------------------
# 1. Affine coupling layer (numpy-only)
# -----------------------------------------------------------------------

class AffineCouplingLayer:
    """
    A single affine coupling layer for a 1D flow, conditioned on (y, d).

    For 1D theta, we cannot split into halves. Instead, we use a fully
    conditional affine transform:
        theta' = theta * exp(s(y, d)) + t(y, d)

    where s and t are simple parametric functions of (y, d).
    """

    def __init__(self, n_hidden=8, seed=None):
        rng = np.random.RandomState(seed)
        # Weights for s-network: (y, d) -> hidden -> 1
        self.W1_s = rng.randn(2, n_hidden) * 0.5
        self.b1_s = np.zeros(n_hidden)
        self.W2_s = rng.randn(n_hidden, 1) * 0.3
        self.b2_s = np.zeros(1)

        # Weights for t-network: (y, d) -> hidden -> 1
        self.W1_t = rng.randn(2, n_hidden) * 0.5
        self.b1_t = np.zeros(n_hidden)
        self.W2_t = rng.randn(n_hidden, 1) * 0.3
        self.b2_t = np.zeros(1)

    def _net_s(self, y, d):
        """Compute log-scale s(y, d)."""
        inp = np.column_stack([y, d])               # (N, 2)
        h = np.tanh(inp @ self.W1_s + self.b1_s)    # (N, n_hidden)
        out = h @ self.W2_s + self.b2_s              # (N, 1)
        return out.ravel()                           # (N,)

    def _net_t(self, y, d):
        """Compute shift t(y, d)."""
        inp = np.column_stack([y, d])
        h = np.tanh(inp @ self.W1_t + self.b1_t)
        out = h @ self.W2_t + self.b2_t
        return out.ravel()

    def forward(self, theta, y, d):
        """theta -> theta', conditioned on (y, d). Returns (theta', log_det)."""
        s = self._net_s(y, d)
        t = self._net_t(y, d)
        theta_out = theta * np.exp(s) + t
        log_det = s
        return theta_out, log_det

    def inverse(self, theta_out, y, d):
        """theta' -> theta, conditioned on (y, d)."""
        s = self._net_s(y, d)
        t = self._net_t(y, d)
        theta = (theta_out - t) * np.exp(-s)
        return theta

    def get_params(self):
        """Return all trainable parameters as a flat vector."""
        return np.concatenate([
            self.W1_s.ravel(), self.b1_s, self.W2_s.ravel(), self.b2_s,
            self.W1_t.ravel(), self.b1_t, self.W2_t.ravel(), self.b2_t,
        ])

    def set_params(self, flat):
        """Set parameters from a flat vector."""
        idx = 0

        def take(shape):
            nonlocal idx
            size = int(np.prod(shape))
            arr = flat[idx:idx + size].reshape(shape)
            idx += size
            return arr

        self.W1_s = take(self.W1_s.shape)
        self.b1_s = take(self.b1_s.shape)
        self.W2_s = take(self.W2_s.shape)
        self.b2_s = take(self.b2_s.shape)
        self.W1_t = take(self.W1_t.shape)
        self.b1_t = take(self.b1_t.shape)
        self.W2_t = take(self.W2_t.shape)
        self.b2_t = take(self.b2_t.shape)

    def n_params(self):
        return len(self.get_params())


# -----------------------------------------------------------------------
# 2. Normalizing flow (stack of coupling layers)
# -----------------------------------------------------------------------

class ConditionalNormalizingFlow:
    """
    Stack of affine coupling layers forming q_phi(theta | y, d).

    The flow maps z ~ N(0,1) -> theta via:
        theta = C_K(C_{K-1}(...C_1(z; y,d)...; y,d); y,d)

    The log-density is:
        log q(theta|y,d) = log N(z; 0,1) - sum_k log_det_k
    """

    def __init__(self, n_layers=4, n_hidden=8, seed=42):
        self.layers = []
        for k in range(n_layers):
            self.layers.append(AffineCouplingLayer(n_hidden, seed=seed + k))

    def forward(self, z, y, d):
        """z -> theta, return (theta, total_log_det)."""
        total_log_det = np.zeros_like(z)
        theta = z.copy()
        for layer in self.layers:
            theta, ld = layer.forward(theta, y, d)
            total_log_det += ld
        return theta, total_log_det

    def inverse(self, theta, y, d):
        """theta -> z (for density evaluation)."""
        z = theta.copy()
        for layer in reversed(self.layers):
            z = layer.inverse(z, y, d)
        return z

    def log_prob(self, theta, y, d):
        """
        log q_phi(theta | y, d) = log N(z; 0,1) - sum log_det
        where z = T^{-1}(theta; y, d).

        But we need the log_det of the inverse, which is -log_det of forward.
        Easier: compute forward from z, accumulate log_det.
        For evaluation, use inverse + forward to get log_det.
        """
        z = self.inverse(theta, y, d)
        # Re-run forward to get log_det
        _, total_log_det = self.forward(z, y, d)
        log_base = norm.logpdf(z)
        return log_base - total_log_det

    def sample(self, y, d, n_samples=1):
        """Sample theta ~ q(theta|y,d)."""
        z = np.random.randn(n_samples)
        y_rep = np.full(n_samples, y) if np.isscalar(y) else y
        d_rep = np.full(n_samples, d) if np.isscalar(d) else d
        theta, _ = self.forward(z, y_rep, d_rep)
        return theta

    def get_params(self):
        return np.concatenate([l.get_params() for l in self.layers])

    def set_params(self, flat):
        idx = 0
        for layer in self.layers:
            n = layer.n_params()
            layer.set_params(flat[idx:idx + n])
            idx += n

    def n_params(self):
        return sum(l.n_params() for l in self.layers)


# -----------------------------------------------------------------------
# 3. Forward model and EIG estimation
# -----------------------------------------------------------------------

def forward_model(theta, d, noise_std=0.3):
    """y = theta * sin(d * theta) + noise."""
    return theta * np.sin(d * theta) + noise_std * np.random.randn(*theta.shape)


def eig_lower_bound(flow, d, prior_mean=0.0, prior_std=1.0,
                    noise_std=0.3, n_mc=500):
    """
    EIG_LB(d) = E_{p(theta,y|d)}[ log q(theta|y,d) - log p(theta) ]

    Estimated via Monte Carlo with N prior-predictive samples.
    """
    # Sample from prior-predictive
    theta_samples = prior_mean + prior_std * np.random.randn(n_mc)
    y_samples = forward_model(theta_samples, d, noise_std)
    d_arr = np.full(n_mc, d)

    # log q(theta|y,d)
    log_q = flow.log_prob(theta_samples, y_samples, d_arr)

    # log p(theta) -- prior
    log_prior = norm.logpdf(theta_samples, loc=prior_mean, scale=prior_std)

    eig = np.mean(log_q - log_prior)
    return eig


# -----------------------------------------------------------------------
# 4. Train the flow for a given design d
# -----------------------------------------------------------------------

def train_flow(flow, d, prior_mean=0.0, prior_std=1.0, noise_std=0.3,
               n_iter=300, lr=0.005, n_mc=200):
    """
    Train the flow parameters phi to maximise EIG_LB for design d.

    Uses finite-difference gradient estimation (no autograd).
    """
    params = flow.get_params()
    n_p = len(params)
    eps = 1e-4

    best_eig = -np.inf
    best_params = params.copy()

    for it in range(n_iter):
        # Estimate gradient via forward finite differences (stochastic)
        # For efficiency, perturb a random subset of parameters each iteration
        n_perturb = min(20, n_p)
        idx_perturb = np.random.choice(n_p, n_perturb, replace=False)

        # Set seed for consistent MC samples across perturbations
        rng_state = np.random.get_state()

        np.random.set_state(rng_state)
        eig_base = eig_lower_bound(flow, d, prior_mean, prior_std, noise_std, n_mc)

        grad = np.zeros(n_p)
        for idx in idx_perturb:
            params_plus = params.copy()
            params_plus[idx] += eps
            flow.set_params(params_plus)
            np.random.set_state(rng_state)
            eig_plus = eig_lower_bound(flow, d, prior_mean, prior_std, noise_std, n_mc)
            grad[idx] = (eig_plus - eig_base) / eps

        # Gradient ascent
        params = params + lr * grad
        flow.set_params(params)

        if eig_base > best_eig:
            best_eig = eig_base
            best_params = params.copy()

        if (it + 1) % 100 == 0:
            print(f"    iter {it+1}/{n_iter}  EIG_LB = {eig_base:.4f}")

    flow.set_params(best_params)
    return best_eig


# -----------------------------------------------------------------------
# 5. Demo: Sequential OED for 1D problem
# -----------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("Variational Bayesian OED with Normalizing Flows")
    print("(arXiv:2404.13056)")
    print("=" * 65)

    np.random.seed(123)

    noise_std = 0.3
    prior_mean = 0.0
    prior_std = 1.0
    theta_true = 0.8  # ground truth parameter

    # --- Grid search over design variable d ---
    print("\n[1] Estimating EIG over design grid (d in [0.5, 5.0])...")
    d_grid = np.linspace(0.5, 5.0, 10)
    eig_values = []

    for d_val in d_grid:
        flow = ConditionalNormalizingFlow(n_layers=3, n_hidden=6, seed=42)
        eig = train_flow(flow, d_val, prior_mean, prior_std, noise_std,
                         n_iter=150, lr=0.003, n_mc=150)
        eig_values.append(eig)
        print(f"  d = {d_val:.2f}  ->  EIG_LB = {eig:.4f}")

    best_idx = np.argmax(eig_values)
    d_star = d_grid[best_idx]
    print(f"\n  Optimal design: d* = {d_star:.2f}  (EIG_LB = {eig_values[best_idx]:.4f})")

    # --- Sequential OED: 3 rounds ---
    print("\n[2] Sequential OED (3 rounds)...")
    observations = []
    current_prior_mean = prior_mean
    current_prior_std = prior_std

    for t in range(3):
        print(f"\n  --- Round {t+1} ---")
        print(f"  Current prior: N({current_prior_mean:.3f}, {current_prior_std:.3f}^2)")

        # Find best design for current prior
        best_eig = -np.inf
        best_d = d_grid[0]
        best_flow = None

        for d_val in d_grid:
            flow = ConditionalNormalizingFlow(n_layers=3, n_hidden=6, seed=42 + t)
            eig = train_flow(flow, d_val, current_prior_mean, current_prior_std,
                             noise_std, n_iter=100, lr=0.003, n_mc=100)
            if eig > best_eig:
                best_eig = eig
                best_d = d_val
                best_flow = flow

        # Conduct experiment at optimal design
        y_obs = theta_true * np.sin(best_d * theta_true) + noise_std * np.random.randn()
        observations.append((best_d, y_obs))
        print(f"  Selected design: d = {best_d:.2f}")
        print(f"  Observation:     y = {y_obs:.4f}")

        # Sample approximate posterior using the trained flow
        theta_post = best_flow.sample(y_obs, best_d, n_samples=5000)
        post_mean = np.mean(theta_post)
        post_std = np.std(theta_post)
        print(f"  Posterior: mean = {post_mean:.3f}, std = {post_std:.3f}")
        print(f"  True theta = {theta_true:.3f}")

        # Update prior for next round (Bayesian update approximation)
        current_prior_mean = post_mean
        current_prior_std = post_std

    # --- Final summary ---
    print("\n" + "=" * 65)
    print("Sequential OED Summary:")
    print("-" * 65)
    print(f"  True parameter:  theta = {theta_true}")
    print(f"  Final estimate:  mean = {current_prior_mean:.4f}, "
          f"std = {current_prior_std:.4f}")
    print(f"  Designs chosen:  {[f'd={d:.2f}' for d, _ in observations]}")
    print(f"  Observations:    {[f'y={y:.4f}' for _, y in observations]}")
    print("=" * 65)
    print("\nDemo complete. The normalizing flow variational posterior enables")
    print("efficient EIG estimation for sequential Bayesian OED.")


if __name__ == "__main__":
    demo()
