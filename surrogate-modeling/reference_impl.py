"""Reference implementation sketch for
"MCMC Informed Neural Emulators for Uncertainty Quantification in Dynamical Systems"
(Haario et al., 2026, arXiv:2603.10987).

Not the official code. Demonstrates the core workflow relevant to
`variation propagation`:

    1. Train a cheap neural emulator as a surrogate for an ODE system whose
       parameters theta are the source of variation.
    2. Replace the expensive ODE solver inside an MCMC (Metropolis-Hastings)
       loop with the emulator, so we can sample the posterior p(theta | data)
       at a fraction of the original cost.
    3. Propagate the sampled theta forward through the emulator to obtain a
       predictive distribution over observables -- i.e. parameter variation
       propagated into output variation.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# "Physical" model: damped oscillator xdd + 2*zeta*w*xd + w^2 x = 0.
# theta = (w, zeta). Solve analytically for speed in this toy example.
# ---------------------------------------------------------------------------
def simulate_damped_oscillator(theta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """theta: (..., 2) with columns (w, zeta). Returns x(t) shape (..., T)."""
    w = theta[..., 0:1]
    zeta = theta[..., 1:2]
    wd = w * torch.sqrt(torch.clamp(1 - zeta ** 2, min=1e-6))
    x0, v0 = 1.0, 0.0
    A = x0
    B = (v0 + zeta * w * x0) / wd
    env = torch.exp(-zeta * w * t)
    return env * (A * torch.cos(wd * t) + B * torch.sin(wd * t))


# ---------------------------------------------------------------------------
# Neural emulator: MLP mapping theta -> full time-series observation.
# ---------------------------------------------------------------------------
class Emulator(nn.Module):
    def __init__(self, t: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.register_buffer("t", t)
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, t.numel()),
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.net(theta)


def train_emulator(model: Emulator, n: int = 2000, steps: int = 800,
                   batch: int = 64) -> None:
    theta = torch.stack([
        torch.empty(n).uniform_(0.5, 3.0),     # w
        torch.empty(n).uniform_(0.02, 0.3),    # zeta
    ], dim=1)
    with torch.no_grad():
        y = simulate_damped_oscillator(theta, model.t)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        idx = torch.randint(0, n, (batch,))
        pred = model(theta[idx])
        loss = (pred - y[idx]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"[emulator] step={step} mse={loss.item():.5f}")


# ---------------------------------------------------------------------------
# MCMC over theta using the emulator instead of the ODE solver.
# ---------------------------------------------------------------------------
def log_posterior(theta: torch.Tensor, y_obs: torch.Tensor, emulator: Emulator,
                  sigma: float) -> torch.Tensor:
    w, zeta = theta[..., 0], theta[..., 1]
    in_prior = (
        (w > 0.5) & (w < 3.0) & (zeta > 0.02) & (zeta < 0.3)
    )
    if not torch.all(in_prior):
        return torch.tensor(-float("inf"))
    with torch.no_grad():
        y_pred = emulator(theta.unsqueeze(0)).squeeze(0)
    return -0.5 * ((y_pred - y_obs) / sigma).pow(2).sum()


def metropolis(emulator: Emulator, y_obs: torch.Tensor, theta0: torch.Tensor,
               n_samples: int = 4000, prop_sd: float = 0.05,
               sigma: float = 0.05) -> np.ndarray:
    theta = theta0.clone()
    lp = log_posterior(theta, y_obs, emulator, sigma)
    chain = np.zeros((n_samples, 2))
    accepts = 0
    for i in range(n_samples):
        prop = theta + prop_sd * torch.randn_like(theta)
        lp_prop = log_posterior(prop, y_obs, emulator, sigma)
        if torch.log(torch.rand(())) < (lp_prop - lp):
            theta, lp = prop, lp_prop
            accepts += 1
        chain[i] = theta.numpy()
    print(f"[mcmc] acceptance rate = {accepts / n_samples:.3f}")
    return chain


# ---------------------------------------------------------------------------
# Forward-propagate posterior samples into observable variation (UQ output).
# ---------------------------------------------------------------------------
def propagate_variation(chain: np.ndarray, emulator: Emulator) -> np.ndarray:
    with torch.no_grad():
        theta = torch.from_numpy(chain.astype(np.float32))
        y = emulator(theta).numpy()
    return y  # (n_samples, T)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    t = torch.linspace(0, 10, 64)
    emulator = Emulator(t)
    train_emulator(emulator)

    # Synthetic observation with known theta_true + noise.
    theta_true = torch.tensor([1.8, 0.1])
    y_obs = simulate_damped_oscillator(theta_true, t) + 0.05 * torch.randn_like(t)

    chain = metropolis(
        emulator, y_obs,
        theta0=torch.tensor([1.5, 0.15]),
        n_samples=3000,
    )
    burn = 500
    post = chain[burn:]
    print(f"posterior mean w={post[:, 0].mean():.3f} (true {theta_true[0]:.3f})")
    print(f"posterior mean zeta={post[:, 1].mean():.3f} (true {theta_true[1]:.3f})")

    y_ens = propagate_variation(post, emulator)
    print("predictive band width (max std over t):",
          float(y_ens.std(axis=0).max()))
