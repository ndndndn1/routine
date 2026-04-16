"""Minimal reference implementation: Surrogate Sensitivity Regularizer
(arXiv:2503.04181).

Demonstrates how a sensitivity-informed regularizer (SIR) stabilizes a
surrogate model for offline optimization, preventing overestimation in
out-of-distribution regions by penalizing the gradient norm.

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
    raise SystemExit("pip install torch")


# ---------------------------------------------------------------------------
# Toy black-box function (ground truth) -------------------------------------
# ---------------------------------------------------------------------------
def true_function(x: np.ndarray) -> np.ndarray:
    """Multimodal 1D function with narrow peaks."""
    return (np.sin(3 * x) * np.exp(-0.1 * x ** 2) + 0.5 * np.cos(x)).flatten()


# ---------------------------------------------------------------------------
# MLP surrogate with sensitivity regularizer --------------------------------
# ---------------------------------------------------------------------------
class Surrogate(nn.Module):
    def __init__(self, dim_in: int = 1, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_surrogate(X: np.ndarray, y: np.ndarray,
                    lam_sir: float = 0.0,
                    epochs: int = 500, lr: float = 1e-3):
    """Train surrogate with optional Sensitivity-Informed Regularizer.

    L = MSE(f_hat, y) + λ * E[||∇_x f_hat||²]
    """
    model = Surrogate(dim_in=X.shape[1])
    Xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.float32)
    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        Xt_grad = Xt.detach().requires_grad_(True)
        pred = model(Xt_grad)
        loss_fit = nn.functional.mse_loss(pred, yt)

        loss_sir = torch.tensor(0.0)
        if lam_sir > 0:
            grads = torch.autograd.grad(pred.sum(), Xt_grad, create_graph=True)[0]
            loss_sir = (grads ** 2).mean()

        loss = loss_fit + lam_sir * loss_sir
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


# ---------------------------------------------------------------------------
# Surrogate sensitivity measurement -----------------------------------------
# ---------------------------------------------------------------------------
def measure_sensitivity(model: Surrogate, X_eval: np.ndarray,
                        sigma: float = 0.05, n_perturb: int = 50):
    """S(x) = E_δ[|f_hat(x+δ) - f_hat(x)|²]"""
    Xt = torch.tensor(X_eval, dtype=torch.float32)
    with torch.no_grad():
        base = model(Xt).numpy()
    sens = np.zeros(len(X_eval))
    for _ in range(n_perturb):
        delta = np.random.randn(*X_eval.shape) * sigma
        Xp = torch.tensor(X_eval + delta, dtype=torch.float32)
        with torch.no_grad():
            perturbed = model(Xp).numpy()
        sens += (perturbed - base) ** 2
    return sens / n_perturb


# ---------------------------------------------------------------------------
# Offline optimization via gradient ascent on surrogate ----------------------
# ---------------------------------------------------------------------------
def offline_optimize(model: Surrogate, x_init: np.ndarray,
                     n_steps: int = 100, lr: float = 0.05):
    x = torch.tensor(x_init, dtype=torch.float32, requires_grad=True)
    opt = optim.Adam([x], lr=lr)
    trajectory = [x.detach().numpy().copy()]
    for _ in range(n_steps):
        loss = -model(x.unsqueeze(0) if x.dim() == 1 else x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        trajectory.append(x.detach().numpy().copy())
    return x.detach().numpy(), trajectory


# ---------------------------------------------------------------------------
# Demo: compare with and without SIR ----------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X_train = rng.uniform(-3, 3, (60, 1)).astype(np.float32)
    y_train = true_function(X_train[:, 0]).astype(np.float32)

    print("Offline Optimization: Surrogate Sensitivity Regularizer")
    print("=" * 55)

    for lam, label in [(0.0, "No SIR (λ=0)"), (0.1, "With SIR (λ=0.1)")]:
        model = train_surrogate(X_train, y_train, lam_sir=lam)

        x_init = np.array([0.5], dtype=np.float32)
        x_opt, traj = offline_optimize(model, x_init)

        with torch.no_grad():
            f_surr = model(torch.tensor(x_opt.reshape(1, -1))).item()
        f_true = true_function(x_opt)[0]

        X_eval = np.linspace(-4, 4, 100).reshape(-1, 1).astype(np.float32)
        sens = measure_sensitivity(model, X_eval)

        print(f"\n[{label}]")
        print(f"  Optimized x = {x_opt[0]:.3f}")
        print(f"  Surrogate f̂(x*) = {f_surr:.3f}")
        print(f"  True f(x*) = {f_true:.3f}")
        print(f"  Gap = {abs(f_surr - f_true):.3f}")
        print(f"  Mean sensitivity = {sens.mean():.4f}")
        print(f"  Max sensitivity  = {sens.max():.4f}")
