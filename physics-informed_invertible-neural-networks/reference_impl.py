"""
Reference implementation: IGNO — Invertible Generative Neural Operator
(arXiv:2511.03241)

Core idea: Encode PDE coefficient fields into a low-dimensional latent space,
use a normalizing flow prior for regularization, and perform gradient-based
inversion in latent space using physics (PDE residual) losses only.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    # -------------------------------------------------------------------
    # Encoder: coefficient field -> latent z
    # -------------------------------------------------------------------

    class Encoder(nn.Module):
        def __init__(self, field_dim, latent_dim, hidden=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(field_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, latent_dim),
            )

        def forward(self, c):
            return self.net(c)

    # -------------------------------------------------------------------
    # Decoder: latent z -> (coefficient field, PDE solution)
    # -------------------------------------------------------------------

    class Decoder(nn.Module):
        def __init__(self, latent_dim, field_dim, sol_dim, hidden=128):
            super().__init__()
            self.coeff_head = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, field_dim),
            )
            self.sol_head = nn.Sequential(
                nn.Linear(latent_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, sol_dim),
            )

        def forward(self, z):
            return self.coeff_head(z), self.sol_head(z)

    # -------------------------------------------------------------------
    # Normalizing flow prior on latent space (RealNVP-style)
    # -------------------------------------------------------------------

    class FlowLayer(nn.Module):
        def __init__(self, dim, hidden=64):
            super().__init__()
            half = dim // 2
            self.s_net = nn.Sequential(nn.Linear(half, hidden), nn.Tanh(), nn.Linear(hidden, dim - half))
            self.t_net = nn.Sequential(nn.Linear(half, hidden), nn.Tanh(), nn.Linear(hidden, dim - half))

        def forward(self, x):
            x1, x2 = x.chunk(2, dim=-1)
            s = self.s_net(x1)
            t = self.t_net(x1)
            y2 = x2 * torch.exp(s) + t
            return torch.cat([x1, y2], -1), s.sum(-1)

        def inverse(self, y):
            y1, y2 = y.chunk(2, dim=-1)
            s = self.s_net(y1)
            t = self.t_net(y1)
            x2 = (y2 - t) * torch.exp(-s)
            return torch.cat([y1, x2], -1)

    class NormalizingFlow(nn.Module):
        def __init__(self, dim, n_layers=4, hidden=64):
            super().__init__()
            self.layers = nn.ModuleList([FlowLayer(dim, hidden) for _ in range(n_layers)])

        def forward(self, x):
            total_ld = 0
            for layer in self.layers:
                x, ld = layer(x)
            total_ld += ld
            return x, total_ld

        def log_prob(self, x):
            z, ld = self.forward(x)
            log_pz = -0.5 * (z ** 2).sum(-1) - 0.5 * z.shape[-1] * np.log(2 * np.pi)
            return log_pz + ld

        def sample(self, n):
            z = torch.randn(n, next(self.parameters()).shape[0])
            for layer in reversed(self.layers):
                z = layer.inverse(z)
            return z

    # -------------------------------------------------------------------
    # Physics loss: 1D Poisson  -(a(x) u'(x))' = f(x)
    # -------------------------------------------------------------------

    def pde_residual(coeff, sol, f_rhs, dx):
        """Finite-difference residual for 1D Poisson."""
        # coeff, sol: (batch, N)
        a = coeff
        u = sol
        # central differences
        a_half_p = 0.5 * (a[:, 1:] + a[:, :-1])
        a_half_m = 0.5 * (a[:, :-1] + a[:, :-1])
        du_p = (u[:, 2:] - u[:, 1:-1]) / dx
        du_m = (u[:, 1:-1] - u[:, :-2]) / dx
        flux_p = a_half_p[:, 1:] * du_p
        flux_m = a_half_p[:, :-1] * du_m
        residual = -(flux_p - flux_m) / dx - f_rhs[:, 1:-1]
        return residual

    # -------------------------------------------------------------------
    # Training loop (physics-informed, no labeled pairs)
    # -------------------------------------------------------------------

    def train_igno(encoder, decoder, flow, n_grid=64, epochs=300, lr=1e-3):
        params = list(encoder.parameters()) + list(decoder.parameters()) + list(flow.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        dx = 1.0 / (n_grid - 1)
        x_grid = torch.linspace(0, 1, n_grid).unsqueeze(0)
        f_rhs = torch.ones(1, n_grid)

        for ep in range(epochs):
            # sample random coefficient fields (log-normal)
            log_a = 0.5 * torch.randn(32, n_grid)
            a_true = torch.exp(log_a)

            z = encoder(a_true)
            a_dec, u_dec = decoder(z)

            # PDE residual loss
            res = pde_residual(a_dec, u_dec, f_rhs.expand(32, -1), dx)
            loss_pde = (res ** 2).mean()

            # reconstruction loss on coefficients
            loss_recon = ((a_dec - a_true) ** 2).mean()

            # flow prior loss
            loss_flow = -flow.log_prob(z).mean()

            # boundary conditions
            loss_bc = (u_dec[:, 0] ** 2 + u_dec[:, -1] ** 2).mean()

            loss = loss_pde + loss_recon + 0.01 * loss_flow + 10.0 * loss_bc
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (ep + 1) % 100 == 0:
                print(f"  epoch {ep+1}/{epochs}  pde={loss_pde.item():.4f}  "
                      f"recon={loss_recon.item():.4f}  flow={loss_flow.item():.2f}")

        return encoder, decoder, flow

    # -------------------------------------------------------------------
    # Inversion: optimize in latent space
    # -------------------------------------------------------------------

    def invert(decoder, flow, y_obs, meas_indices, n_grid=64, steps=200, lr=0.05, lam_flow=0.01):
        """
        Given sparse measurements y_obs at meas_indices, recover the
        coefficient field by optimizing z in latent space.
        """
        latent_dim = next(decoder.parameters()).shape[0]
        # initialize from flow prior
        z = torch.randn(1, latent_dim, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        y_target = torch.tensor(y_obs, dtype=torch.float32).unsqueeze(0)

        for step in range(steps):
            a_dec, u_dec = decoder(z)
            y_pred = u_dec[:, meas_indices]
            loss_data = ((y_pred - y_target) ** 2).sum()
            loss_prior = -flow.log_prob(z).sum()
            loss = loss_data + lam_flow * loss_prior
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            a_rec, u_rec = decoder(z)
        return a_rec.numpy()[0], u_rec.numpy()[0]

    # -------------------------------------------------------------------
    # Demo
    # -------------------------------------------------------------------

    def demo():
        n_grid = 32
        latent_dim = 8

        encoder = Encoder(n_grid, latent_dim)
        decoder = Decoder(latent_dim, n_grid, n_grid)
        flow = NormalizingFlow(latent_dim, n_layers=3, hidden=32)

        print("Training IGNO (physics-informed, no labeled data)...")
        train_igno(encoder, decoder, flow, n_grid=n_grid, epochs=300)

        # synthetic inversion
        a_true = np.exp(0.5 * np.random.randn(n_grid))
        meas_idx = np.array([8, 16, 24])
        # fake observations (in practice from PDE solve)
        y_obs = np.random.randn(len(meas_idx)) * 0.1

        print("Inverting from sparse measurements...")
        a_rec, u_rec = invert(decoder, flow, y_obs, meas_idx, n_grid=n_grid)
        print(f"Recovered coefficient field (first 5): {a_rec[:5]}")

    demo()

else:

    def demo_numpy():
        """NumPy-only: latent-space inversion with Gaussian prior."""
        n_grid = 32
        latent_dim = 8

        # random linear decoder (toy)
        W_a = np.random.randn(n_grid, latent_dim) * 0.1
        W_u = np.random.randn(n_grid, latent_dim) * 0.1

        meas_idx = np.array([8, 16, 24])
        H = W_u[meas_idx]  # measurement matrix

        a_true = np.exp(0.5 * np.random.randn(n_grid))
        z_true = np.linalg.lstsq(W_a, a_true, rcond=None)[0]
        y_obs = H @ z_true + 0.01 * np.random.randn(len(meas_idx))

        # MAP estimate with Gaussian prior N(0, I)
        lam = 0.01
        z_map = np.linalg.solve(H.T @ H + lam * np.eye(latent_dim), H.T @ y_obs)
        a_rec = W_a @ z_map

        print("NumPy-only IGNO demo (linear decoder + Gaussian prior):")
        print(f"  True coeff (first 5):      {a_true[:5]}")
        print(f"  Recovered coeff (first 5): {a_rec[:5]}")

    demo_numpy()
