"""
Reference implementation: INN with Bayesian regularization for inverse problems
(arXiv:2510.26704)

Core idea: Two regularization terms for invertible neural network training
that recover Bayesian point estimators (posterior mean / MAP) upon inversion.
A learned surrogate replaces the expensive forward operator.
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
    # Affine coupling layer (building block of INN)
    # -------------------------------------------------------------------

    class AffineCoupling(nn.Module):
        def __init__(self, dim, hidden=64):
            super().__init__()
            half = dim // 2
            self.net_s = nn.Sequential(nn.Linear(half, hidden), nn.ReLU(), nn.Linear(hidden, dim - half))
            self.net_t = nn.Sequential(nn.Linear(half, hidden), nn.ReLU(), nn.Linear(hidden, dim - half))

        def forward(self, x):
            x1, x2 = x.chunk(2, dim=-1)
            s = self.net_s(x1)
            t = self.net_t(x1)
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=-1)
            return torch.cat([x1, y2], dim=-1), log_det

        def inverse(self, y):
            y1, y2 = y.chunk(2, dim=-1)
            s = self.net_s(y1)
            t = self.net_t(y1)
            x2 = (y2 - t) * torch.exp(-s)
            return torch.cat([y1, x2], dim=-1)

    # -------------------------------------------------------------------
    # Simple INN (stacked coupling layers with permutations)
    # -------------------------------------------------------------------

    class SimpleINN(nn.Module):
        def __init__(self, dim, n_layers=4, hidden=64):
            super().__init__()
            self.layers = nn.ModuleList()
            self.perms = []
            for _ in range(n_layers):
                self.layers.append(AffineCoupling(dim, hidden))
                perm = torch.randperm(dim)
                self.perms.append(perm)

        def forward(self, x):
            total_log_det = 0
            for layer, perm in zip(self.layers, self.perms):
                x, ld = layer(x)
                total_log_det += ld
                x = x[:, perm]
            return x, total_log_det

        def inverse(self, y):
            for layer, perm in zip(reversed(self.layers), reversed(self.perms)):
                inv_perm = torch.argsort(perm)
                y = y[:, inv_perm]
                y = layer.inverse(y)
            return y

    # -------------------------------------------------------------------
    # Surrogate forward model
    # -------------------------------------------------------------------

    class Surrogate(nn.Module):
        def __init__(self, dim_in, dim_out, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim_in, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, dim_out),
            )

        def forward(self, x):
            return self.net(x)

    # -------------------------------------------------------------------
    # Training with R1 (posterior mean) regularization
    # -------------------------------------------------------------------

    def train_inn_r1(inn, surrogate, X_train, Y_train, epochs=200, lr=1e-3, lam=1.0):
        """
        R1 regularization: encourages the inverse to approximate E[x|y].
        Loss = ||G(x) - y||^2 + lam * ||G^{-1}(y) - x||^2 + latent_reg
        The first term trains the forward direction; the second + latent_reg
        push the inverse toward the posterior mean.
        """
        opt = torch.optim.Adam(list(inn.parameters()) + list(surrogate.parameters()), lr=lr)
        X = torch.tensor(X_train, dtype=torch.float32)
        Y = torch.tensor(Y_train, dtype=torch.float32)

        for ep in range(epochs):
            # forward: INN maps x -> (y_pred, z_latent)
            out, log_det = inn(X)
            y_pred = out[:, : Y.shape[1]]
            z_latent = out[:, Y.shape[1]:]

            # forward fidelity
            loss_fwd = torch.mean((y_pred - Y) ** 2)

            # latent regularization (Gaussian prior on z)
            loss_latent = torch.mean(z_latent ** 2)

            # R1: inverse reconstruction toward posterior mean
            y_with_z = torch.cat([Y, torch.randn_like(z_latent)], dim=-1)
            x_recon = inn.inverse(y_with_z)
            loss_r1 = torch.mean((x_recon - X) ** 2)

            loss = loss_fwd + 0.5 * loss_latent + lam * loss_r1
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (ep + 1) % 50 == 0:
                print(f"  epoch {ep+1}/{epochs}  loss={loss.item():.4f}  "
                      f"fwd={loss_fwd.item():.4f}  R1={loss_r1.item():.4f}")

    # -------------------------------------------------------------------
    # Demo: 1D elliptic inverse problem (toy)
    # -------------------------------------------------------------------

    def demo():
        dim_x = 4
        dim_y = 4
        dim_z = dim_x  # latent same size for invertibility
        total_dim = dim_y + dim_z

        # toy forward model: y = A @ x + noise
        A = np.random.randn(dim_y, dim_x) * 0.5
        n_train = 500
        X_train = np.random.randn(n_train, dim_x)
        Y_train = X_train @ A.T + 0.05 * np.random.randn(n_train, dim_y)

        # pad x to match total_dim for INN
        X_padded = np.hstack([X_train, np.zeros((n_train, dim_z))])

        inn = SimpleINN(total_dim, n_layers=4, hidden=64)
        surrogate = Surrogate(dim_x, dim_y)

        print("Training INN with R1 regularization (posterior mean)...")
        train_inn_r1(inn, surrogate, X_padded, Y_train, epochs=200)

        # inversion test
        x_test = np.random.randn(dim_x)
        y_test = x_test @ A.T
        y_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(0)
        z_sample = torch.randn(1, dim_z)
        y_with_z = torch.cat([y_tensor, z_sample], dim=-1)
        with torch.no_grad():
            x_recon = inn.inverse(y_with_z).numpy()[0, :dim_x]
        print(f"True x:        {x_test}")
        print(f"Reconstructed: {x_recon}")
        print(f"Error:         {np.linalg.norm(x_test - x_recon):.4f}")

    demo()

else:
    # -------------------------------------------------------------------
    # NumPy-only fallback: linear Bayesian inversion demo
    # -------------------------------------------------------------------

    def demo_numpy():
        dim_x, dim_y = 4, 4
        A = np.random.randn(dim_y, dim_x) * 0.5
        sigma_noise = 0.05
        Sigma_prior = np.eye(dim_x)

        # posterior mean (closed-form for linear Gaussian)
        x_true = np.random.randn(dim_x)
        y_obs = A @ x_true + sigma_noise * np.random.randn(dim_y)

        Sigma_post_inv = A.T @ A / sigma_noise**2 + np.linalg.inv(Sigma_prior)
        Sigma_post = np.linalg.inv(Sigma_post_inv)
        mu_post = Sigma_post @ (A.T @ y_obs / sigma_noise**2)

        print("Linear Bayesian inversion (posterior mean = R1 target):")
        print(f"  True x:          {x_true}")
        print(f"  Posterior mean:   {mu_post}")
        print(f"  Error:           {np.linalg.norm(x_true - mu_post):.4f}")

    demo_numpy()
