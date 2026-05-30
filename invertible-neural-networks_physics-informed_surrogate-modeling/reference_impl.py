"""
Reference implementation: Bayesian Inverse Problems Meet Flow Matching via
Transformers (arXiv:2503.01375)

Core idea: Conditional Flow Matching + Transformer for amortized posterior
sampling in PDE-based inverse problems. Once trained, the model produces
posterior samples for any new observation set without re-training.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Forward model: 1D Darcy-like elliptic PDE (simplified)
# ---------------------------------------------------------------------------

def darcy_forward(kappa, n_grid=50):
    """
    Solve -d/dx(kappa(x) du/dx) = 1 on [0,1], u(0)=u(1)=0.
    kappa: (n_grid,) permeability field
    Returns: (n_grid,) solution u(x)
    """
    dx = 1.0 / (n_grid + 1)
    # finite difference: tridiagonal system
    kappa_half = 0.5 * (kappa[:-1] + kappa[1:])
    diag = np.zeros(n_grid)
    off_diag = np.zeros(n_grid - 1)

    diag[0] = (kappa_half[0]) / dx**2
    diag[-1] = (kappa_half[-1]) / dx**2
    for i in range(1, n_grid - 1):
        diag[i] = (kappa_half[i-1] + kappa_half[i]) / dx**2
    for i in range(n_grid - 1):
        off_diag[i] = -kappa_half[i] / dx**2

    # Thomas algorithm
    A_diag = diag.copy()
    A_off = off_diag.copy()
    rhs = np.ones(n_grid)

    for i in range(1, n_grid):
        m = A_off[i-1] / A_diag[i-1]
        A_diag[i] -= m * A_off[i-1]
        rhs[i] -= m * rhs[i-1]

    u = np.zeros(n_grid)
    u[-1] = rhs[-1] / A_diag[-1]
    for i in range(n_grid - 2, -1, -1):
        u[i] = (rhs[i] - A_off[i] * u[i+1]) / A_diag[i]

    return u


# ---------------------------------------------------------------------------
# Simplified Transformer-like attention for observation encoding
# ---------------------------------------------------------------------------

class SimpleAttention:
    """Single-head self-attention (numpy)."""

    def __init__(self, d_model, d_k=16):
        self.d_k = d_k
        s = np.sqrt(2.0 / d_model)
        self.Wq = np.random.randn(d_model, d_k) * s
        self.Wk = np.random.randn(d_model, d_k) * s
        self.Wv = np.random.randn(d_model, d_k) * s
        self.Wo = np.random.randn(d_k, d_model) * np.sqrt(2.0 / d_k)

    def forward(self, X):
        """X: (seq_len, d_model) -> (seq_len, d_model)."""
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        scores = Q @ K.T / np.sqrt(self.d_k)
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn /= attn.sum(axis=-1, keepdims=True)
        return (attn @ V) @ self.Wo


# ---------------------------------------------------------------------------
# Conditional Flow Matching with Transformer
# ---------------------------------------------------------------------------

class CFMTransformer:
    """
    Amortized conditional flow matching for Bayesian inverse problems.
    Velocity network conditioned on observation embeddings via attention.
    """

    def __init__(self, d_param, d_obs_feat, hidden=64, lr=1e-3):
        self.d_param = d_param
        self.lr = lr

        # observation encoder (attention)
        self.attn = SimpleAttention(d_obs_feat, d_k=16)

        # velocity MLP: [z_t(d_param), t(1), obs_embed(d_obs_feat)] -> d_param
        d_in = d_param + 1 + d_obs_feat
        s = np.sqrt(2.0 / d_in)
        self.W1 = np.random.randn(d_in, hidden) * s
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, d_param) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(d_param)

    def _encode_obs(self, y_obs, obs_locs):
        """Encode sparse observations. y_obs: (n_obs,), obs_locs: (n_obs,)."""
        feats = np.column_stack([obs_locs, y_obs])
        d = feats.shape[1]
        # pad to d_obs_feat if needed
        if d < self.attn.Wq.shape[0]:
            pad = np.zeros((feats.shape[0], self.attn.Wq.shape[0] - d))
            feats = np.concatenate([feats, pad], axis=1)
        attended = self.attn.forward(feats)
        return attended.mean(axis=0)  # (d_obs_feat,)

    def _velocity(self, z_t, t_val, obs_embed):
        """z_t: (B, d_param), t_val: (B,1), obs_embed: (d_obs_feat,)."""
        B = z_t.shape[0]
        obs_rep = np.tile(obs_embed, (B, 1))
        inp = np.concatenate([z_t, t_val, obs_rep], axis=1)
        h = np.maximum(inp @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2

    def train_step(self, theta_batch, y_obs_batch, obs_locs):
        """
        theta_batch: (B, d_param)
        y_obs_batch: (B, n_obs)
        obs_locs: (n_obs,)
        """
        B = theta_batch.shape[0]
        t = np.random.rand(B, 1)

        # encode observations (use mean across batch for simplicity)
        obs_embed = self._encode_obs(y_obs_batch.mean(axis=0), obs_locs)

        # flow: noise -> theta
        z0 = np.random.randn(B, self.d_param)
        z_t = (1 - t) * z0 + t * theta_batch
        v_target = theta_batch - z0

        v_pred = self._velocity(z_t, t, obs_embed)
        err = v_pred - v_target
        loss = 0.5 * np.mean(err**2)

        # backprop
        grad_out = err / B
        inp = np.concatenate([z_t, t, np.tile(obs_embed, (B, 1))], axis=1)
        h = np.maximum(inp @ self.W1 + self.b1, 0)
        self.W2 -= self.lr * (h.T @ grad_out)
        self.b2 -= self.lr * grad_out.sum(axis=0)
        grad_h = grad_out @ self.W2.T
        grad_h[h <= 0] = 0
        self.W1 -= self.lr * (inp.T @ grad_h)
        self.b1 -= self.lr * grad_h.sum(axis=0)

        return loss

    def sample_posterior(self, y_obs, obs_locs, n_samples=200, n_steps=50):
        """Generate posterior samples given observations."""
        obs_embed = self._encode_obs(y_obs, obs_locs)
        z = np.random.randn(n_samples, self.d_param)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t_val = np.full((n_samples, 1), step * dt)
            v = self._velocity(z, t_val, obs_embed)
            z = z + v * dt

        return z


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    np.random.seed(42)

    n_grid = 30
    d_param = 5
    obs_idx = np.array([5, 10, 15, 20, 25])
    obs_locs = obs_idx / n_grid

    # parameterise kappa as piecewise constant on d_param segments
    def params_to_kappa(params, n_grid):
        kappa = np.zeros(n_grid)
        seg = n_grid // d_param
        for i, p in enumerate(params):
            start = i * seg
            end = min((i + 1) * seg, n_grid)
            kappa[start:end] = np.exp(p)
        return kappa + 0.1

    # generate training data
    print("Generating training data...")
    N = 500
    theta_all = np.random.randn(N, d_param) * 0.5
    y_obs_all = np.zeros((N, len(obs_idx)))
    for i in range(N):
        kappa = params_to_kappa(theta_all[i], n_grid)
        u = darcy_forward(kappa, n_grid)
        y_obs_all[i] = u[obs_idx] + np.random.randn(len(obs_idx)) * 0.01

    # train CFM-Transformer
    d_obs_feat = 2
    model = CFMTransformer(d_param, d_obs_feat, hidden=64, lr=5e-4)
    print("Training CFM-Transformer...")
    batch_size = 64
    for epoch in range(500):
        idx = np.random.choice(N, batch_size)
        loss = model.train_step(theta_all[idx], y_obs_all[idx], obs_locs)
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1}  loss={loss:.5f}")

    # test: infer posterior for a new observation
    true_theta = np.array([0.5, -0.3, 0.8, -0.1, 0.4])
    kappa_true = params_to_kappa(true_theta, n_grid)
    u_true = darcy_forward(kappa_true, n_grid)
    y_test = u_true[obs_idx] + np.random.randn(len(obs_idx)) * 0.01

    print("\nPosterior sampling for test observation...")
    posterior = model.sample_posterior(y_test, obs_locs, n_samples=300, n_steps=80)
    post_mean = posterior.mean(axis=0)
    post_std = posterior.std(axis=0)

    print(f"  True theta:     {true_theta}")
    print(f"  Posterior mean:  {np.round(post_mean, 3)}")
    print(f"  Posterior std:   {np.round(post_std, 3)}")
    rel_err = np.mean(np.abs(post_mean - true_theta) / (np.abs(true_theta) + 1e-8))
    print(f"  Relative error:  {rel_err:.4f}")


if __name__ == "__main__":
    demo()
