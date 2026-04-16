"""Minimal reference implementation of AutoTandemML (arXiv:2502.15643).

Reproduces the core loop: a Tandem Neural Network (inverse + forward cascade)
trained with active-learning-based data selection. The TNN avoids the
multimodal average problem in one-to-many inverse design by self-validating
through the forward model.

Official code: https://github.com/lukagrbcic/AutoTandemML

Run:
    python reference_impl.py
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    raise SystemExit("pip install torch  # required for this reference impl")


# ---------------------------------------------------------------------------
# Forward simulator (ground truth) ------------------------------------------
# ---------------------------------------------------------------------------
def forward_simulator(x: np.ndarray) -> np.ndarray:
    """Toy many-to-one simulator: 3D design → 2D response."""
    x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
    y1 = np.sin(x1) * np.cos(x2) + 0.1 * x3
    y2 = np.cos(x1) * np.sin(x3) + 0.1 * x2
    return np.stack([y1, y2], axis=-1)


# ---------------------------------------------------------------------------
# Simple MLP block ----------------------------------------------------------
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Tandem Neural Network -----------------------------------------------------
# ---------------------------------------------------------------------------
class TandemNN(nn.Module):
    def __init__(self, dim_y: int, dim_x: int, hidden: int = 64):
        super().__init__()
        self.inverse_net = MLP([dim_y, hidden, hidden, dim_x])
        self.forward_net = MLP([dim_x, hidden, hidden, dim_y])

    def forward(self, y_target):
        x_pred = self.inverse_net(y_target)
        y_pred = self.forward_net(x_pred)
        return x_pred, y_pred


def pretrain_forward(model: TandemNN, X: np.ndarray, Y: np.ndarray,
                     epochs: int = 300, lr: float = 1e-3):
    """Pre-train the forward sub-network on (x, y) pairs."""
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    opt = optim.Adam(model.forward_net.parameters(), lr=lr)
    for _ in range(epochs):
        loss = nn.functional.mse_loss(model.forward_net(Xt), Yt)
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_tandem(model: TandemNN, Y: np.ndarray,
                 epochs: int = 500, lr: float = 1e-3):
    """Train full tandem: freeze forward, optimize inverse via forward loss."""
    Yt = torch.tensor(Y, dtype=torch.float32)
    for p in model.forward_net.parameters():
        p.requires_grad = False
    opt = optim.Adam(model.inverse_net.parameters(), lr=lr)
    for _ in range(epochs):
        _, y_pred = model(Yt)
        loss = nn.functional.mse_loss(y_pred, Yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in model.forward_net.parameters():
        p.requires_grad = True


# ---------------------------------------------------------------------------
# Active learning: uncertainty-based selection -------------------------------
# ---------------------------------------------------------------------------
def active_select(model: TandemNN, pool_Y: np.ndarray,
                  n_select: int = 10) -> np.ndarray:
    """Select candidates with highest forward-reconstruction error (proxy
    for model uncertainty in the tandem architecture)."""
    Yt = torch.tensor(pool_Y, dtype=torch.float32)
    with torch.no_grad():
        _, y_pred = model(Yt)
    error = ((y_pred.numpy() - pool_Y) ** 2).sum(axis=1)
    idx = np.argsort(error)[-n_select:]
    return idx


# ---------------------------------------------------------------------------
# AutoTandemML main loop -----------------------------------------------------
# ---------------------------------------------------------------------------
def auto_tandem_ml(
    simulator,
    dim_x: int = 3,
    dim_y: int = 2,
    x_bounds: tuple = (-2.0, 2.0),
    n_init: int = 30,
    n_rounds: int = 5,
    n_select: int = 10,
    pool_size: int = 500,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    X_train = rng.uniform(x_bounds[0], x_bounds[1], (n_init, dim_x))
    Y_train = simulator(X_train)

    model = TandemNN(dim_y, dim_x)

    for rd in range(n_rounds):
        pretrain_forward(model, X_train, Y_train, epochs=200)
        train_tandem(model, Y_train, epochs=300)

        pool_X = rng.uniform(x_bounds[0], x_bounds[1], (pool_size, dim_x))
        pool_Y = simulator(pool_X)
        idx = active_select(model, pool_Y, n_select=n_select)

        X_train = np.vstack([X_train, pool_X[idx]])
        Y_train = np.vstack([Y_train, pool_Y[idx]])
        print(f"Round {rd + 1}: train size = {len(X_train)}")

    return model, X_train, Y_train


# ---------------------------------------------------------------------------
# Demo ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model, X, Y = auto_tandem_ml(forward_simulator)

    test_X = np.random.default_rng(99).uniform(-2, 2, (50, 3))
    test_Y = forward_simulator(test_X)
    Yt = torch.tensor(test_Y, dtype=torch.float32)
    with torch.no_grad():
        x_pred, y_recon = model(Yt)
    mse = ((y_recon.numpy() - test_Y) ** 2).mean()
    print(f"\nTest forward-reconstruction MSE: {mse:.6f}")

    y_from_sim = forward_simulator(x_pred.numpy())
    sim_mse = ((y_from_sim - test_Y) ** 2).mean()
    print(f"Simulator validation MSE: {sim_mse:.6f}")
