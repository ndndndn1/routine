"""Reference implementation: Tandem / MDN / INN comparison (arXiv:2411.09429).

Compares three canonical approaches to inverse design ambiguity on a toy
many-to-one forward problem y = f(x) where multiple x can map to the same y.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    raise SystemExit("pip install torch")


# ---------------------------------------------------------------------------
# Toy forward problem: 2D → 1D (many-to-one with circular symmetry) --------
# ---------------------------------------------------------------------------
def forward_fn(x: np.ndarray) -> np.ndarray:
    """y = x1^2 + x2^2  →  infinitely many x for each y (circle)."""
    return (x[..., 0] ** 2 + x[..., 1] ** 2)[..., None]


def generate_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, (n, 2))
    Y = forward_fn(X)
    return X.astype(np.float32), Y.astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Tandem Network ---------------------------------------------------------
# ---------------------------------------------------------------------------
class TandemNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inv = nn.Sequential(nn.Linear(1, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 2))
        self.fwd = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, y):
        x_hat = self.inv(y)
        y_hat = self.fwd(x_hat)
        return x_hat, y_hat


def train_tandem(X, Y, epochs=600):
    model = TandemNet()
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    opt = optim.Adam(model.fwd.parameters(), lr=1e-3)
    for _ in range(epochs // 2):
        loss = nn.functional.mse_loss(model.fwd(Xt), Yt)
        opt.zero_grad(); loss.backward(); opt.step()
    for p in model.fwd.parameters():
        p.requires_grad = False
    opt2 = optim.Adam(model.inv.parameters(), lr=1e-3)
    for _ in range(epochs // 2):
        _, y_hat = model(Yt)
        loss = nn.functional.mse_loss(y_hat, Yt)
        opt2.zero_grad(); loss.backward(); opt2.step()
    return model


# ---------------------------------------------------------------------------
# 2. Mixture Density Network (MDN) -----------------------------------------
# ---------------------------------------------------------------------------
class MDN(nn.Module):
    def __init__(self, n_mix=5):
        super().__init__()
        self.n_mix = n_mix
        self.backbone = nn.Sequential(nn.Linear(1, 64), nn.ReLU(),
                                      nn.Linear(64, 64), nn.ReLU())
        self.pi_head = nn.Linear(64, n_mix)
        self.mu_head = nn.Linear(64, n_mix * 2)
        self.logvar_head = nn.Linear(64, n_mix * 2)

    def forward(self, y):
        h = self.backbone(y)
        pi = torch.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h).view(-1, self.n_mix, 2)
        logvar = self.logvar_head(h).view(-1, self.n_mix, 2)
        return pi, mu, logvar

    def nll_loss(self, y, x_true):
        pi, mu, logvar = self(y)
        var = torch.exp(logvar) + 1e-6
        x_exp = x_true.unsqueeze(1).expand_as(mu)
        log_prob = -0.5 * (((x_exp - mu) ** 2) / var + logvar + math.log(2 * math.pi))
        log_prob = log_prob.sum(-1)
        log_mix = torch.log(pi + 1e-9) + log_prob
        return -torch.logsumexp(log_mix, dim=1).mean()

    @torch.no_grad()
    def sample(self, y, n_samples=20):
        pi, mu, logvar = self(y)
        std = torch.exp(0.5 * logvar)
        bs = y.shape[0]
        samples = []
        for _ in range(n_samples):
            k = torch.multinomial(pi, 1).squeeze(-1)
            chosen_mu = mu[torch.arange(bs), k]
            chosen_std = std[torch.arange(bs), k]
            x = chosen_mu + chosen_std * torch.randn_like(chosen_mu)
            samples.append(x)
        return torch.stack(samples, dim=1)


def train_mdn(X, Y, epochs=800):
    model = MDN(n_mix=8)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        loss = model.nll_loss(Yt, Xt)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


# ---------------------------------------------------------------------------
# 3. Invertible Neural Network (simplified RealNVP) -------------------------
# ---------------------------------------------------------------------------
class CouplingLayer(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.s_net = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, dim))
        self.t_net = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, dim))

    def forward(self, x):
        x_masked = x * self.mask
        s = self.s_net(x_masked) * (1 - self.mask)
        t = self.t_net(x_masked) * (1 - self.mask)
        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(-1)
        return y, log_det

    def inverse(self, y):
        y_masked = y * self.mask
        s = self.s_net(y_masked) * (1 - self.mask)
        t = self.t_net(y_masked) * (1 - self.mask)
        x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        return x


class SimpleINN(nn.Module):
    """x (2D) ↔ [y (1D), z (1D)] via coupling layers.
    Forward: x → [y, z]; Inverse: [y, z] → x."""
    def __init__(self, n_layers=6):
        super().__init__()
        masks = [[1, 0], [0, 1]] * (n_layers // 2)
        self.layers = nn.ModuleList([CouplingLayer(2, m) for m in masks])

    def forward(self, x):
        log_det_total = 0
        z = x
        for layer in self.layers:
            z, ld = layer(z)
            log_det_total += ld
        return z, log_det_total

    def inverse(self, yz):
        x = yz
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x


def train_inn(X, Y, epochs=800):
    model = SimpleINN()
    Xt = torch.from_numpy(X)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    for ep in range(epochs):
        z, log_det = model(Xt)
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(-1)
        loss = -(log_pz + log_det).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


# ---------------------------------------------------------------------------
# Demo & comparison ----------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X, Y = generate_data(3000)
    target_y = np.array([[1.0]], dtype=np.float32)
    print("=" * 60)
    print(f"Target y = {target_y.item():.1f}  (true solutions: circle x1²+x2²=1)")
    print("=" * 60)

    # Tandem
    tandem = train_tandem(X, Y)
    with torch.no_grad():
        x_t, _ = tandem(torch.from_numpy(target_y))
    print(f"\n[Tandem] Single prediction: {x_t.numpy().flatten()}")
    print("  → Deterministic: returns ONE point on the circle")

    # MDN
    mdn = train_mdn(X, Y)
    samples = mdn.sample(torch.from_numpy(target_y), n_samples=20)
    s = samples.squeeze(0).numpy()
    radii = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)
    print(f"\n[MDN] 20 samples (radius mean={radii.mean():.3f} std={radii.std():.3f}):")
    print(f"  → Multimodal: covers different points on the circle")

    # INN
    inn = train_inn(X, Y)
    yz_samples = []
    for _ in range(20):
        z = torch.randn(1, 1)
        yz = torch.cat([target_y[0:1, :] * 0.5, z], dim=-1)
        x_inv = inn.inverse(torch.tensor(yz, dtype=torch.float32))
        yz_samples.append(x_inv.detach().numpy().flatten())
    yz_arr = np.array(yz_samples)
    radii_inn = np.sqrt(yz_arr[:, 0] ** 2 + yz_arr[:, 1] ** 2)
    print(f"\n[INN] 20 samples (radius mean={radii_inn.mean():.3f} std={radii_inn.std():.3f}):")
    print(f"  → Latent z sweeps the solution manifold")
