"""
Reference implementation of Physics-Informed Uncertainty (PIU) for AI-driven inverse design.
Paper: "Physics-Informed Uncertainty Enables Reliable AI-driven Design" (arXiv:2601.18638)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------

class SurrogateMLP(nn.Module):
    """MLP surrogate mapping design parameters to complex S-parameter responses."""

    def __init__(self, n_design: int, n_freq: int, hidden: int = 64):
        super().__init__()
        # Output: real + imag parts of S11 and S21 at each frequency point
        self.net = nn.Sequential(
            nn.Linear(n_design, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 4 * n_freq),
        )
        self.n_freq = n_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw output of shape (batch, 4*n_freq)."""
        return self.net(x)

    def predict_s_params(self, x: torch.Tensor):
        """Return (S11, S21) as complex tensors, each of shape (batch, n_freq)."""
        out = self.forward(x)
        nf = self.n_freq
        s11 = torch.complex(out[:, :nf],       out[:, nf:2*nf])
        s21 = torch.complex(out[:, 2*nf:3*nf], out[:, 3*nf:])
        return s11, s21


# ---------------------------------------------------------------------------
# Physics-Informed Uncertainty
# ---------------------------------------------------------------------------

def physics_informed_uncertainty(model: SurrogateMLP,
                                  x: torch.Tensor,
                                  eps: float = 1e-6) -> torch.Tensor:
    """
    Compute PIU as the mean energy-conservation violation |S11|^2 + |S21|^2 - 1
    averaged over frequency; positive excess signals a physically impossible prediction.
    """
    model.eval()
    with torch.no_grad():
        s11, s21 = model.predict_s_params(x)
        power_sum = s11.abs() ** 2 + s21.abs() ** 2          # (batch, n_freq)
        violation = torch.clamp(power_sum - 1.0, min=0.0)    # passive-device bound
        piu = violation.mean(dim=1) + eps                     # (batch,)
    return piu


# ---------------------------------------------------------------------------
# Acquisition function (UCB-style)
# ---------------------------------------------------------------------------

def ucb_acquisition(model: SurrogateMLP,
                     x: torch.Tensor,
                     target_response: torch.Tensor,
                     beta: float = 2.0) -> torch.Tensor:
    """
    UCB = -mean_squared_error(predicted_S21, target) + beta * PIU.
    Higher value → more promising candidate (exploitation + exploration).
    """
    model.eval()
    with torch.no_grad():
        _, s21 = model.predict_s_params(x)
        s21_mag = s21.abs()                                    # (batch, n_freq)
        mse = ((s21_mag - target_response.unsqueeze(0)) ** 2).mean(dim=1)
        piu = physics_informed_uncertainty(model, x)
    return -mse + beta * piu


# ---------------------------------------------------------------------------
# Mock expensive simulator
# ---------------------------------------------------------------------------

def mock_simulator(x: np.ndarray, n_freq: int, noise: float = 0.02) -> np.ndarray:
    """
    Cheap stand-in for a full-wave EM solver; returns 4*n_freq real values
    encoding (Re S11, Im S11, Re S21, Im S21) satisfying energy conservation.
    """
    batch = x.shape[0]
    freqs = np.linspace(0, np.pi, n_freq)
    results = np.zeros((batch, 4 * n_freq))
    for i, xi in enumerate(x):
        # Resonance depth and centre controlled by design parameters
        centre = 0.3 + 0.4 * xi[0]
        width  = 0.1 + 0.3 * xi[1]
        t21_mag = np.exp(-((freqs - centre * np.pi) ** 2) / (2 * width ** 2))
        t11_mag = np.sqrt(np.clip(1.0 - t21_mag ** 2, 0, 1))
        phase11 = np.pi * xi[0] * freqs
        phase21 = np.pi * xi[1] * freqs
        results[i, :n_freq]         = t11_mag * np.cos(phase11)
        results[i, n_freq:2*n_freq] = t11_mag * np.sin(phase11)
        results[i, 2*n_freq:3*n_freq] = t21_mag * np.cos(phase21)
        results[i, 3*n_freq:]         = t21_mag * np.sin(phase21)
    results += noise * np.random.randn(*results.shape)
    return results.astype(np.float32)


# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------

def train_surrogate(model: SurrogateMLP,
                    x_data: np.ndarray,
                    y_data: np.ndarray,
                    epochs: int = 200,
                    lr: float = 1e-3,
                    batch_size: int = 32) -> float:
    """Train surrogate on (x_data, y_data) and return final MSE loss."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    tx = torch.tensor(x_data)
    ty = torch.tensor(y_data)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True)
    final_loss = float("inf")
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        final_loss = loss.item()
    return final_loss


# ---------------------------------------------------------------------------
# Candidate grid helper
# ---------------------------------------------------------------------------

def make_candidate_grid(n_design: int, resolution: int = 30) -> torch.Tensor:
    """Build a uniform grid of candidate designs in [0, 1]^n_design (n_design <= 2)."""
    axes = [np.linspace(0, 1, resolution)] * n_design
    grids = np.meshgrid(*axes)
    candidates = np.stack([g.ravel() for g in grids], axis=1).astype(np.float32)
    return torch.tensor(candidates)


# ---------------------------------------------------------------------------
# PIU-driven active-learning optimisation loop
# ---------------------------------------------------------------------------

def run_piu_optimization(n_design: int = 2,
                          n_freq: int = 20,
                          n_init: int = 10,
                          n_iterations: int = 8,
                          n_candidates: int = 900,
                          beta: float = 2.0,
                          target_centre: float = 0.5,
                          verbose: bool = True) -> dict:
    """
    Full PIU-based inverse-design loop.

    1. Seed with random simulations.
    2. Train surrogate on accumulated data.
    3. Evaluate UCB acquisition (using PIU as uncertainty) over candidate grid.
    4. Query simulator at top-K candidates.
    5. Retrain and repeat.
    Returns history dict with per-iteration best design and objective value.
    """
    rng = np.random.default_rng(42)

    # Target: Gaussian bandpass centred at target_centre (normalised frequency)
    freqs_np = np.linspace(0, np.pi, n_freq)
    target_mag = np.exp(-((freqs_np - target_centre * np.pi) ** 2) / (2 * 0.15 ** 2))
    target_t = torch.tensor(target_mag, dtype=torch.float32)

    # --- Seed dataset ---
    x_pool = rng.uniform(0, 1, size=(n_init, n_design)).astype(np.float32)
    y_pool = mock_simulator(x_pool, n_freq)

    model = SurrogateMLP(n_design, n_freq)
    candidates = make_candidate_grid(n_design, resolution=int(n_candidates ** 0.5))

    history = {"iteration": [], "best_design": [], "best_obj": [], "train_loss": []}

    for it in range(n_iterations):
        loss = train_surrogate(model, x_pool, y_pool, epochs=300)

        acq = ucb_acquisition(model, candidates, target_t, beta=beta)
        best_idx = int(acq.argmax().item())
        best_candidate = candidates[best_idx].numpy()

        # Objective: MSE between predicted |S21| magnitude and target
        model.eval()
        with torch.no_grad():
            _, s21 = model.predict_s_params(candidates[best_idx:best_idx+1])
            obj = float(((s21.abs() - target_t) ** 2).mean())

        history["iteration"].append(it)
        history["best_design"].append(best_candidate.copy())
        history["best_obj"].append(obj)
        history["train_loss"].append(loss)

        if verbose:
            piu_val = float(physics_informed_uncertainty(model, candidates[best_idx:best_idx+1]))
            print(f"Iter {it+1:2d} | loss={loss:.4f} | obj={obj:.4f} "
                  f"| PIU={piu_val:.4f} | design={np.round(best_candidate, 3)}")

        # Query simulator at top-5 candidates (batch active sampling)
        top_k = 5
        top_idx = acq.topk(top_k).indices.numpy()
        new_x = candidates[top_idx].numpy()
        new_y = mock_simulator(new_x, n_freq)
        x_pool = np.vstack([x_pool, new_x])
        y_pool = np.vstack([y_pool, new_y])

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    print("=" * 60)
    print("PIU-based Inverse Design Demo (arXiv:2601.18638)")
    print("=" * 60)
    print(f"Task: find 2-D FSS design whose |S21| matches a Gaussian")
    print(f"      bandpass filter at normalised centre frequency 0.5")
    print("=" * 60)

    history = run_piu_optimization(
        n_design=2,
        n_freq=20,
        n_init=10,
        n_iterations=10,
        beta=2.0,
        target_centre=0.5,
        verbose=True,
    )

    best_iter = int(np.argmin(history["best_obj"]))
    print("=" * 60)
    print(f"Best result at iteration {history['iteration'][best_iter]+1}")
    print(f"  Design parameters : {np.round(history['best_design'][best_iter], 4)}")
    print(f"  Surrogate MSE obj : {history['best_obj'][best_iter]:.6f}")
    print("=" * 60)

    # Verify that the final model broadly respects energy conservation
    n_freq = 20
    model_final = SurrogateMLP(2, n_freq)
    x_pool_final = np.random.default_rng(99).uniform(0, 1, (50, 2)).astype(np.float32)
    y_pool_final = mock_simulator(x_pool_final, n_freq)
    train_surrogate(model_final, x_pool_final, y_pool_final, epochs=400)

    test_x = torch.rand(200, 2)
    piu_vals = physics_informed_uncertainty(model_final, test_x)
    print(f"\nSanity check on fresh model (200 random points):")
    print(f"  Mean PIU : {piu_vals.mean():.4f}")
    print(f"  Max  PIU : {piu_vals.max():.4f}")
    print("  (PIU near 0 indicates good energy-conservation compliance)")
