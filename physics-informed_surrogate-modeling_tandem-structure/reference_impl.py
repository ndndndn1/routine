"""
Reference implementation of the tandem physics-informed DNN for reactively loaded
metasurface / antenna array inverse design (arXiv:2603.15430).

Tandem architecture:
  target pattern --> [Inverse DNN] --> reactances --> [Physics Forward] --> predicted pattern
                                                                               |
                                            cosine-similarity loss <-----------+---------> target pattern
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Physics-based forward model (fixed, not trained)
# ---------------------------------------------------------------------------

def build_impedance_matrix(n_elem: int, d: float, wavelength: float) -> np.ndarray:
    """Construct the mutual-impedance matrix for a 1-D dipole array using the induced-EMF method."""
    k = 2 * np.pi / wavelength
    Z = np.zeros((n_elem, n_elem), dtype=complex)
    for i in range(n_elem):
        for j in range(n_elem):
            r = abs(i - j) * d
            if i == j:
                # Self-impedance: resonant half-wave dipole (real part only for simplicity)
                Z[i, j] = 73.0 + 42.5j
            else:
                # Mutual impedance approximation via sinc-like coupling kernel
                Z[i, j] = 30.0 * np.exp(-1j * k * r) / (k * r + 1e-9) * np.sinc(r / wavelength)
    return Z


def element_pattern(theta: np.ndarray) -> np.ndarray:
    """Active element pattern for a half-wave dipole (E-plane, normalised)."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta) + 1e-9
    return np.cos(0.5 * np.pi * cos_t) / sin_t


def physics_forward(
    reactances: torch.Tensor,
    Z: torch.Tensor,
    theta_grid: torch.Tensor,
    d: float,
    wavelength: float,
) -> torch.Tensor:
    """
    Compute normalised far-field power pattern for a batch of reactance distributions.

    Args:
        reactances : (batch, n_elem) real tensor of load reactance values [ohms]
        Z          : (n_elem, n_elem) complex impedance matrix (fixed)
        theta_grid : (n_theta,) observation angles in radians
        d          : element spacing [m]
        wavelength : free-space wavelength [m]

    Returns:
        pattern : (batch, n_theta) normalised power pattern tensor
    """
    batch, n_elem = reactances.shape
    n_theta = theta_grid.shape[0]
    k = 2 * np.pi / wavelength

    # Build load matrix Z_L = diag(j * X) for each sample
    jX = 1j * reactances.detach().cpu().numpy()  # (batch, n_elem)

    # Element pattern values at each observation angle
    ep = element_pattern(theta_grid.cpu().numpy())  # (n_theta,)

    # Array factor via current distribution: (Z + Z_L) I = V_s (one driven element)
    V_s = np.zeros(n_elem, dtype=complex)
    V_s[n_elem // 2] = 1.0  # feed at centre element

    Z_np = Z.numpy()  # (n_elem, n_elem) complex
    patterns = np.zeros((batch, n_theta), dtype=float)

    for b in range(batch):
        Z_load = np.diag(1j * reactances[b].detach().cpu().numpy())
        Z_total = Z_np + Z_load
        try:
            I = np.linalg.solve(Z_total, V_s)  # (n_elem,)
        except np.linalg.LinAlgError:
            I = np.zeros(n_elem, dtype=complex)

        # Array factor: AF(theta) = sum_n I_n * exp(j k n d cos(theta))
        n_idx = np.arange(n_elem)
        th = theta_grid.cpu().numpy()  # (n_theta,)
        # phase matrix: (n_elem, n_theta)
        phase = np.exp(1j * k * np.outer(n_idx * d, np.cos(th)))
        AF = I @ phase  # (n_theta,)

        power = (np.abs(AF) * np.abs(ep)) ** 2
        norm = np.max(power) + 1e-12
        patterns[b] = power / norm

    return torch.tensor(patterns, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Inverse DNN
# ---------------------------------------------------------------------------

class InverseDNN(nn.Module):
    """Maps a target far-field power pattern to load reactance values."""

    def __init__(self, n_theta: int, n_elem: int, hidden_dims=(256, 256, 128)):
        super().__init__()
        layers = []
        in_dim = n_theta
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_elem))
        self.net = nn.Sequential(*layers)

    def forward(self, pattern: torch.Tensor) -> torch.Tensor:
        """Return predicted reactances (batch, n_elem) given pattern (batch, n_theta)."""
        return self.net(pattern)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cosine_similarity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine-similarity loss: 1 - mean(cos_sim) so minimum is 0 at perfect match."""
    pred_n = pred / (pred.norm(dim=1, keepdim=True) + 1e-12)
    targ_n = target / (target.norm(dim=1, keepdim=True) + 1e-12)
    cos_sim = (pred_n * targ_n).sum(dim=1)
    return 1.0 - cos_sim.mean()


# ---------------------------------------------------------------------------
# Dataset generation (synthetic targets from known reactance distributions)
# ---------------------------------------------------------------------------

def generate_dataset(
    n_samples: int,
    n_elem: int,
    Z_np: np.ndarray,
    theta_grid: np.ndarray,
    d: float,
    wavelength: float,
    reactance_range: tuple = (-200.0, 200.0),
) -> tuple:
    """Generate (target_pattern, ground_truth_reactance) pairs via the forward model."""
    X_min, X_max = reactance_range
    reactances = np.random.uniform(X_min, X_max, size=(n_samples, n_elem)).astype(np.float32)

    Z_torch = torch.tensor(Z_np, dtype=torch.complex64)
    th_torch = torch.tensor(theta_grid, dtype=torch.float32)
    X_torch = torch.tensor(reactances)

    patterns = physics_forward(X_torch, Z_torch, th_torch, d, wavelength)
    return patterns, X_torch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: InverseDNN,
    target_patterns: torch.Tensor,
    Z: torch.Tensor,
    theta_grid: torch.Tensor,
    d: float,
    wavelength: float,
    n_epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> list:
    """Tandem training: inverse DNN predicts reactances, forward physics model evaluates them."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    n = len(target_patterns)
    loss_history = []

    for epoch in range(n_epochs):
        idx = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            tgt = target_patterns[batch_idx]

            # Inverse DNN: target pattern -> predicted reactances
            pred_X = model(tgt)

            # Physics forward: predicted reactances -> predicted pattern (no gradient through forward)
            pred_pattern = physics_forward(pred_X, Z, theta_grid, d, wavelength)

            loss = cosine_similarity_loss(pred_pattern, tgt)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:4d}/{n_epochs}  loss={avg_loss:.5f}")

    return loss_history


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(
    model: InverseDNN,
    test_patterns: torch.Tensor,
    Z: torch.Tensor,
    theta_grid: torch.Tensor,
    d: float,
    wavelength: float,
) -> float:
    """Return mean cosine similarity on a test set."""
    model.eval()
    with torch.no_grad():
        pred_X = model(test_patterns)
    pred_patterns = physics_forward(pred_X, Z, theta_grid, d, wavelength)
    pred_n = pred_patterns / (pred_patterns.norm(dim=1, keepdim=True) + 1e-12)
    targ_n = test_patterns / (test_patterns.norm(dim=1, keepdim=True) + 1e-12)
    return (pred_n * targ_n).sum(dim=1).mean().item()


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # Array parameters: 6-element 1-D half-wave spaced dipole array
    N_ELEM = 6
    WAVELENGTH = 1.0          # normalised units
    D = 0.5 * WAVELENGTH      # half-wavelength spacing
    N_THETA = 181             # observation angles: 0 to pi

    theta_grid = np.linspace(0, np.pi, N_THETA).astype(np.float32)
    Z_np = build_impedance_matrix(N_ELEM, D, WAVELENGTH)
    Z_torch = torch.tensor(Z_np, dtype=torch.complex64)
    th_torch = torch.tensor(theta_grid)

    print(f"Array: {N_ELEM} elements, d={D}λ, {N_THETA} angle samples")
    print(f"Impedance matrix condition number: {np.linalg.cond(Z_np):.2f}")

    # Generate training and test datasets
    print("\nGenerating dataset...")
    all_patterns, all_X = generate_dataset(
        n_samples=2000, n_elem=N_ELEM, Z_np=Z_np,
        theta_grid=theta_grid, d=D, wavelength=WAVELENGTH
    )
    train_patterns, test_patterns = all_patterns[:1800], all_patterns[1800:]
    print(f"Train: {len(train_patterns)}  Test: {len(test_patterns)}")

    # Build and train the inverse DNN
    model = InverseDNN(n_theta=N_THETA, n_elem=N_ELEM, hidden_dims=(256, 256, 128))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nInverse DNN parameters: {n_params:,}")

    print("\nTandem training (inverse DNN + fixed physics forward)...")
    loss_history = train(
        model, train_patterns, Z_torch, th_torch,
        d=D, wavelength=WAVELENGTH,
        n_epochs=300, batch_size=64, lr=1e-3
    )

    # Evaluate
    mean_cos_sim = evaluate(model, test_patterns, Z_torch, th_torch, D, WAVELENGTH)
    print(f"\nTest mean cosine similarity: {mean_cos_sim:.4f}  (1.0 = perfect)")
    print(f"Final training loss: {loss_history[-1]:.5f}")

    # Spot-check: reconstruct a single target pattern
    model.eval()
    sample_tgt = test_patterns[0:1]
    with torch.no_grad():
        pred_X = model(sample_tgt)
    pred_pat = physics_forward(pred_X, Z_torch, th_torch, D, WAVELENGTH)

    tgt_np = sample_tgt[0].numpy()
    pred_np = pred_pat[0].numpy()
    cos_s = float(np.dot(tgt_np, pred_np) / (np.linalg.norm(tgt_np) * np.linalg.norm(pred_np) + 1e-12))
    print(f"\nSpot-check sample 0: cosine similarity = {cos_s:.4f}")
    print(f"  Predicted reactances: {pred_X[0].numpy().round(1)} Ω")
    print(f"  Target pattern peak at θ index: {np.argmax(tgt_np)}")
    print(f"  Predicted pattern peak at θ index: {np.argmax(pred_np)}")
