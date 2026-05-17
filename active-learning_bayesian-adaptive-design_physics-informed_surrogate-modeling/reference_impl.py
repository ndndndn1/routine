"""Reference implementation: Multi-Fidelity PINN with Bayesian UQ via MC Dropout (arXiv:2602.01176)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


torch.manual_seed(42)
np.random.seed(42)


class LowFidelityNet(nn.Module):
    """Fully-connected network producing cheap low-fidelity PDE solution approximations."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return low-fidelity prediction for input collocation points x."""
        return self.net(x)


class GatedResidualNet(nn.Module):
    """Gated residual correction network blending linear and nonlinear cross-fidelity terms."""

    def __init__(self, hidden: int = 32, dropout_p: float = 0.1):
        super().__init__()
        self.linear_branch = nn.Linear(2, 1)
        self.nonlinear_branch = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, 1),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, u_lf: torch.Tensor) -> torch.Tensor:
        """Compute gated blend of linear and nonlinear corrections given x and u_lf."""
        inp = torch.cat([x, u_lf], dim=-1)
        g = self.gate_net(inp)
        lin = self.linear_branch(inp)
        nlin = self.nonlinear_branch(inp)
        return g * lin + (1.0 - g) * nlin


class MultiFidelityPINN(nn.Module):
    """Hierarchical MF-PINN: low-fidelity base plus gated residual high-fidelity correction."""

    def __init__(self, hidden: int = 32, dropout_p: float = 0.1):
        super().__init__()
        self.lf_net = LowFidelityNet(hidden=hidden)
        self.hf_correction = GatedResidualNet(hidden=hidden, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (u_lf, u_hf) predictions; u_hf = u_lf + gated correction."""
        u_lf = self.lf_net(x)
        delta = self.hf_correction(x, u_lf.detach())
        u_hf = u_lf + delta
        return u_lf, u_hf

    def enable_dropout(self) -> None:
        """Switch all Dropout layers to train mode for MC Dropout inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


def pde_residual(model: MultiFidelityPINN, x_col: torch.Tensor, f_col: torch.Tensor) -> torch.Tensor:
    """Compute 1-D diffusion residual: d²u/dx² - f(x) using autograd on HF output."""
    x_col = x_col.requires_grad_(True)
    _, u_hf = model(x_col)
    du_dx = torch.autograd.grad(u_hf.sum(), x_col, create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), x_col, create_graph=True)[0]
    return d2u_dx2 - f_col


def build_toy_problem(n_col: int = 200, n_lf: int = 80, n_hf: int = 12):
    """
    Generate toy 1-D PDE data: u_xx = f(x), f(x) = -pi^2 sin(pi x), exact u = sin(pi x).

    Low-fidelity observations are noisy; high-fidelity observations are sparse and clean.
    Returns collocation, lf, and hf tensors as (x, y) pairs plus the forcing values.
    """
    def u_exact(x): return np.sin(np.pi * x)
    def f_forcing(x): return -(np.pi ** 2) * np.sin(np.pi * x)

    x_col = np.linspace(0.0, 1.0, n_col).reshape(-1, 1).astype(np.float32)
    f_col = f_forcing(x_col).astype(np.float32)

    x_lf = np.random.uniform(0.0, 1.0, (n_lf, 1)).astype(np.float32)
    y_lf = (u_exact(x_lf) + 0.15 * np.random.randn(n_lf, 1)).astype(np.float32)

    idx = np.random.choice(n_col, n_hf, replace=False)
    x_hf = x_col[idx]
    y_hf = u_exact(x_hf).astype(np.float32)

    to = lambda a: torch.tensor(a)
    return (to(x_col), to(f_col)), (to(x_lf), to(y_lf)), (to(x_hf), to(y_hf))


def train(
    model: MultiFidelityPINN,
    col_data: tuple,
    lf_data: tuple,
    hf_data: tuple,
    n_epochs: int = 4000,
    lr: float = 1e-3,
    lambda_pde: float = 1.0,
    lambda_lf: float = 0.5,
    lambda_hf: float = 2.0,
) -> list[float]:
    """Train MF-PINN with combined PDE residual, LF supervised, and HF supervised losses."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.3)
    x_col, f_col = col_data
    x_lf, y_lf = lf_data
    x_hf, y_hf = hf_data
    losses = []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        res = pde_residual(model, x_col, f_col)
        loss_pde = lambda_pde * (res ** 2).mean()

        u_lf_pred, _ = model(x_lf)
        loss_lf = lambda_lf * ((u_lf_pred - y_lf) ** 2).mean()

        _, u_hf_pred = model(x_hf)
        loss_hf = lambda_hf * ((u_hf_pred - y_hf) ** 2).mean()

        loss = loss_pde + loss_lf + loss_hf
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            losses.append(loss.item())
            print(f"Epoch {epoch:5d}  loss={loss.item():.5f}  pde={loss_pde.item():.5f}  "
                  f"lf={loss_lf.item():.5f}  hf={loss_hf.item():.5f}")

    return losses


def mc_dropout_predict(
    model: MultiFidelityPINN,
    x: torch.Tensor,
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MC Dropout forward passes and return mean, std of HF predictions."""
    model.eval()
    model.enable_dropout()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            _, u_hf = model(x)
            preds.append(u_hf.numpy())
    preds = np.concatenate(preds, axis=-1)
    return preds.mean(axis=-1), preds.std(axis=-1), preds


def print_ascii_plot(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    truth: np.ndarray,
    width: int = 60,
    height: int = 20,
) -> None:
    """Render ASCII uncertainty band plot of predictions vs ground truth."""
    x_min, x_max = x.min(), x.max()
    y_min = min((mean - 2 * std).min(), truth.min()) - 0.05
    y_max = max((mean + 2 * std).max(), truth.max()) + 0.05

    def to_col(xi): return int((xi - x_min) / (x_max - x_min) * (width - 1))
    def to_row(yi): return int((y_max - yi) / (y_max - y_min) * (height - 1))

    grid = [[" "] * width for _ in range(height)]
    for xi, mi, si in zip(x, mean, std):
        c = to_col(xi)
        for y_val in np.linspace(mi - 2 * si, mi + 2 * si, 6):
            r = to_row(y_val)
            if 0 <= r < height and grid[r][c] == " ":
                grid[r][c] = "."
    for xi, ti in zip(x, truth):
        c = to_col(xi)
        r = to_row(ti)
        if 0 <= r < height:
            grid[r][c] = "*"
    for xi, mi in zip(x, mean):
        c = to_col(xi)
        r = to_row(mi)
        if 0 <= r < height:
            grid[r][c] = "#"

    print("\n  MF-PINN prediction  (* truth  # mean  . 2-sigma band)")
    print("  " + "-" * width)
    for row in grid:
        print("  |" + "".join(row) + "|")
    print("  " + "-" * width)
    print(f"  x: [{x_min:.2f}, {x_max:.2f}]   y: [{y_min:.2f}, {y_max:.2f}]")


if __name__ == "__main__":
    print("=== Multi-Fidelity PINN with MC Dropout UQ (arXiv:2602.01176) ===\n")

    col_data, lf_data, hf_data = build_toy_problem(n_col=200, n_lf=80, n_hf=12)

    model = MultiFidelityPINN(hidden=48, dropout_p=0.1)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    train(model, col_data, lf_data, hf_data, n_epochs=4000)

    x_test = torch.linspace(0.0, 1.0, 120).reshape(-1, 1)
    mean_pred, std_pred, all_preds = mc_dropout_predict(model, x_test, n_samples=300)

    x_np = x_test.numpy().ravel()
    truth_np = np.sin(np.pi * x_np)

    rmse = np.sqrt(((mean_pred.ravel() - truth_np) ** 2).mean())
    mean_unc = std_pred.mean()
    coverage = np.mean(np.abs(mean_pred.ravel() - truth_np) <= 2 * std_pred.ravel())

    print(f"\n--- Evaluation on 120-point test grid ---")
    print(f"  RMSE (HF mean vs exact):       {rmse:.5f}")
    print(f"  Mean MC-Dropout std:           {mean_unc:.5f}")
    print(f"  Empirical 2-sigma coverage:    {coverage:.1%}  (ideal ~95%)")

    print_ascii_plot(x_np, mean_pred.ravel(), std_pred.ravel(), truth_np)

    print("\nDone.")
