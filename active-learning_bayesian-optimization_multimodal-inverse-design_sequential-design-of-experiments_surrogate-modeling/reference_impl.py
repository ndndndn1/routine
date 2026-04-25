"""GUIDe: Generative and Uncertainty-Informed Inverse Design via forward surrogate + MCMC."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── constants ──────────────────────────────────────────────────────────────────
DESIGN_DIM = 4       # dimensionality of design parameter space
RESPONSE_DIM = 10    # discretised stress-strain curve points
N_MC_SAMPLES = 50    # forward passes for MC Dropout UQ
DROPOUT_P = 0.1      # dropout probability during training and inference


# ── toy physics ────────────────────────────────────────────────────────────────

def toy_forward(x: np.ndarray) -> np.ndarray:
    """Map 4-D design parameters to a 10-point stress-strain curve via a deterministic toy model."""
    x = np.atleast_2d(x)
    strains = np.linspace(0.0, 1.0, RESPONSE_DIM)
    # nonlinear response: sigmoidal rise controlled by x[0..1], plateau by x[2], softening by x[3]
    stiffness = 1.0 + 3.0 * x[:, 0:1]          # (N,1)
    strength  = 0.5 + 1.5 * x[:, 1:2]          # (N,1)
    plateau   = 0.3 + 0.6 * x[:, 2:3]          # (N,1)
    softening = 0.5 + 1.0 * x[:, 3:4]          # (N,1)
    rise = strength * (1.0 - np.exp(-stiffness * strains))
    decay = plateau * np.exp(-softening * np.maximum(strains - 0.4, 0.0))
    return rise + decay                          # (N, RESPONSE_DIM)


def generate_dataset(n: int = 2000, seed: int = 0) -> tuple:
    """Sample random designs uniformly in [0,1]^4 and evaluate toy forward model."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (n, DESIGN_DIM)).astype(np.float32)
    Y = toy_forward(X).astype(np.float32)
    return X, Y


# ── surrogate model ────────────────────────────────────────────────────────────

class ForwardSurrogate(nn.Module):
    """MLP forward surrogate with MC Dropout for epistemic uncertainty quantification."""

    def __init__(self, in_dim: int = DESIGN_DIM, out_dim: int = RESPONSE_DIM,
                 hidden: int = 128, p: float = DROPOUT_P):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single deterministic forward pass (dropout active when model.train())."""
        return self.net(x)

    def predict_with_uncertainty(self, x: torch.Tensor,
                                  n_samples: int = N_MC_SAMPLES) -> tuple:
        """Return (mean, std) over MC Dropout stochastic forward passes."""
        self.train()  # keep dropout active for stochastic sampling
        with torch.no_grad():
            preds = torch.stack([self.net(x) for _ in range(n_samples)], dim=0)
        mean = preds.mean(dim=0)
        std  = preds.std(dim=0) + 1e-6
        return mean, std


# ── training ───────────────────────────────────────────────────────────────────

def train_surrogate(model: ForwardSurrogate, X: np.ndarray, Y: np.ndarray,
                    epochs: int = 300, lr: float = 1e-3, batch: int = 256) -> list:
    """Train the forward surrogate with MSE loss; return per-epoch loss history."""
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=batch, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    losses = []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        sched.step()
        losses.append(epoch_loss / len(X))
    return losses


# ── uncertainty-informed log-likelihood ────────────────────────────────────────

def log_likelihood(design: np.ndarray, target: np.ndarray,
                   model: ForwardSurrogate, tolerance: float = 0.05) -> float:
    """Compute log p(target | design) = sum of Gaussian log-likelihoods with MC Dropout std."""
    x = torch.from_numpy(design.astype(np.float32)).unsqueeze(0)
    mean, std = model.predict_with_uncertainty(x)
    mean_np = mean.squeeze(0).numpy()
    std_np  = std.squeeze(0).numpy()
    # effective std blends model uncertainty with user tolerance
    sigma = np.sqrt(std_np**2 + tolerance**2)
    residuals = (mean_np - target) / sigma
    return -0.5 * np.sum(residuals**2 + np.log(2.0 * np.pi * sigma**2))


def log_prior(design: np.ndarray) -> float:
    """Uniform prior over [0,1]^4; returns 0 inside, -inf outside."""
    if np.all(design >= 0.0) and np.all(design <= 1.0):
        return 0.0
    return -np.inf


# ── Metropolis-Hastings MCMC ───────────────────────────────────────────────────

def metropolis_hastings(target: np.ndarray, model: ForwardSurrogate,
                        n_samples: int = 500, n_burnin: int = 200,
                        step_size: float = 0.05, tolerance: float = 0.05,
                        seed: int = 42) -> np.ndarray:
    """Sample designs from p(design | target) via Metropolis-Hastings MCMC."""
    rng = np.random.default_rng(seed)
    current = rng.uniform(0.0, 1.0, DESIGN_DIM)
    log_p_current = (log_prior(current) +
                     log_likelihood(current, target, model, tolerance))
    samples = []
    accepted = 0
    total_steps = n_burnin + n_samples

    for step in range(total_steps):
        proposal = current + rng.normal(0.0, step_size, DESIGN_DIM)
        lp = log_prior(proposal)
        if lp == -np.inf:
            continue
        lp += log_likelihood(proposal, target, model, tolerance)
        log_accept = lp - log_p_current
        if np.log(rng.uniform()) < log_accept:
            current = proposal
            log_p_current = lp
            accepted += 1
        if step >= n_burnin:
            samples.append(current.copy())

    acceptance_rate = accepted / total_steps
    print(f"  MH acceptance rate: {acceptance_rate:.3f}")
    return np.array(samples)


# ── thinning for diversity ─────────────────────────────────────────────────────

def thin_samples(samples: np.ndarray, n_keep: int = 10) -> np.ndarray:
    """Select evenly-spaced samples from the chain to reduce autocorrelation."""
    indices = np.linspace(0, len(samples) - 1, n_keep, dtype=int)
    return samples[indices]


# ── main demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # 1. generate training data
    print("Generating training data...")
    X_train, Y_train = generate_dataset(n=3000)

    # 2. train surrogate
    print("Training forward surrogate...")
    surrogate = ForwardSurrogate()
    loss_hist = train_surrogate(surrogate, X_train, Y_train, epochs=400)
    print(f"  Final training loss: {loss_hist[-1]:.6f}")

    # 3. define a target stress-strain curve
    target_design = np.array([0.7, 0.4, 0.6, 0.3], dtype=np.float32)
    target_response = toy_forward(target_design[None])[0]
    print(f"\nTarget design (ground truth): {target_design}")
    print(f"Target response: {np.round(target_response, 3)}")

    # 4. run MCMC to sample feasible designs
    print("\nRunning Metropolis-Hastings MCMC...")
    chain = metropolis_hastings(
        target=target_response,
        model=surrogate,
        n_samples=1000,
        n_burnin=300,
        step_size=0.06,
        tolerance=0.04,
        seed=7,
    )
    print(f"  Chain length: {len(chain)}")

    # 5. thin chain and report diverse solutions
    diverse_designs = thin_samples(chain, n_keep=8)
    print("\n── Sampled designs and their predicted responses ──")
    print(f"{'Design':>40s}   {'Pred response (first 5 pts)':>30s}   {'RMSE':>6s}")
    for i, d in enumerate(diverse_designs):
        xt = torch.from_numpy(d.astype(np.float32)).unsqueeze(0)
        mean, std = surrogate.predict_with_uncertainty(xt)
        pred = mean.squeeze(0).numpy()
        rmse = np.sqrt(np.mean((pred - target_response) ** 2))
        d_str = "[" + ", ".join(f"{v:.3f}" for v in d) + "]"
        r_str = "[" + ", ".join(f"{v:.3f}" for v in pred[:5]) + " …]"
        print(f"  {i+1}. {d_str}   {r_str}   {rmse:.4f}")

    # 6. verify surrogate quality on hold-out set
    X_test, Y_test = generate_dataset(n=500, seed=99)
    Xt = torch.from_numpy(X_test)
    surrogate.eval()
    with torch.no_grad():
        Y_pred = surrogate(Xt).numpy()
    test_rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))
    print(f"\nSurrogate hold-out RMSE: {test_rmse:.6f}")
