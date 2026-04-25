"""
MDN inverse design: Gaussian mixture over design params given target properties.
Mirrors the pipeline from arXiv:2409.00564 (Curvo et al., 2024).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DESIGN_DIM = 5
TARGET_DIM = 3
K = 4          # mixture components
HIDDEN = 128
BATCH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Toy physics: non-unique forward map  (design -> target)
# ---------------------------------------------------------------------------

def simulate(x: np.ndarray) -> np.ndarray:
    """Nonlinear map with intentional symmetry so the inverse has multiple solutions."""
    t0 = np.sin(x[:, 0] * np.pi) * np.cos(x[:, 1] * np.pi) + 0.3 * x[:, 2]
    t1 = (x[:, 0] ** 2 + x[:, 1] ** 2) / 2.0 + 0.2 * np.sin(x[:, 3] * np.pi)
    t2 = np.tanh(x[:, 0] + x[:, 4]) * np.cos(x[:, 2] * np.pi)
    return np.stack([t0, t1, t2], axis=1).astype(np.float32)


def make_dataset(n: int = 60_000, seed: int = 0) -> tuple:
    """Sample uniform design params and run toy simulator to get paired data."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, DESIGN_DIM)).astype(np.float32)
    Y = simulate(X)
    return X, Y


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

class Normaliser:
    """Stores mean/std and applies zero-mean unit-variance transforms."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, arr: np.ndarray):
        self.mean = arr.mean(0, keepdims=True).astype(np.float32)
        self.std = arr.std(0, keepdims=True).astype(np.float32) + 1e-8
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std + self.mean


# ---------------------------------------------------------------------------
# Forward surrogate: MLP  design -> target
# ---------------------------------------------------------------------------

class ForwardNet(nn.Module):
    """MLP surrogate mapping normalised design params to normalised target properties."""

    def __init__(self, in_dim: int = DESIGN_DIM, out_dim: int = TARGET_DIM, h: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.SiLU(),
            nn.Linear(h, h),      nn.SiLU(),
            nn.Linear(h, h),      nn.SiLU(),
            nn.Linear(h, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_forward(model: ForwardNet, loader: DataLoader, epochs: int = 40) -> list:
    """Train surrogate with MSE loss; returns per-epoch loss history."""
    opt = optim.Adam(model.parameters(), lr=3e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        sched.step()
        history.append(total / len(loader.dataset))
    return history


# ---------------------------------------------------------------------------
# MDN inverse model: target -> mixture over design params
# ---------------------------------------------------------------------------

class MDNHead(nn.Module):
    """Outputs (log_pi, mu, log_sigma) for K Gaussian components over design_dim."""

    def __init__(self, in_dim: int = TARGET_DIM, out_dim: int = DESIGN_DIM,
                 k: int = K, h: int = HIDDEN):
        super().__init__()
        self.k = k
        self.out_dim = out_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h),  nn.SiLU(),
            nn.Linear(h, h),       nn.SiLU(),
            nn.Linear(h, h),       nn.SiLU(),
        )
        self.pi_head    = nn.Linear(h, k)
        self.mu_head    = nn.Linear(h, k * out_dim)
        self.sigma_head = nn.Linear(h, k * out_dim)

    def forward(self, y: torch.Tensor):
        """Return (log_pi [B,K], mu [B,K,D], log_sigma [B,K,D])."""
        h = self.trunk(y)
        log_pi    = torch.log_softmax(self.pi_head(h), dim=-1)
        mu        = self.mu_head(h).view(-1, self.k, self.out_dim)
        log_sigma = self.sigma_head(h).view(-1, self.k, self.out_dim).clamp(-6, 2)
        return log_pi, mu, log_sigma


def mdn_nll(log_pi: torch.Tensor, mu: torch.Tensor,
            log_sigma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of x under the Gaussian mixture."""
    sigma = log_sigma.exp()
    # x: [B, D]  -> [B, 1, D]
    x_exp = x.unsqueeze(1)
    # log N(x | mu_k, sigma_k)  summed over dimensions -> [B, K]
    log_gauss = -0.5 * (((x_exp - mu) / sigma) ** 2 + 2 * log_sigma
                        + np.log(2 * np.pi)).sum(-1)
    log_mix = torch.logsumexp(log_pi + log_gauss, dim=-1)
    return -log_mix.mean()


def train_mdn(model: MDNHead, loader: DataLoader, epochs: int = 60) -> list:
    """Train MDN inverse model; input is (target, design) pairs."""
    opt = optim.Adam(model.parameters(), lr=3e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for _ in range(epochs):
        total = 0.0
        for yb, xb in loader:   # note: y is input, x is target for MDN
            yb, xb = yb.to(DEVICE), xb.to(DEVICE)
            log_pi, mu, log_sigma = model(yb)
            loss = mdn_nll(log_pi, mu, log_sigma, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        sched.step()
        history.append(total / len(loader.dataset))
    return history


# ---------------------------------------------------------------------------
# Inference: sample designs, validate through forward surrogate
# ---------------------------------------------------------------------------

def sample_designs(mdn: MDNHead, y_norm: torch.Tensor,
                   n_samples: int = 64) -> torch.Tensor:
    """Draw n_samples candidate design vectors from the MDN posterior."""
    mdn.eval()
    with torch.no_grad():
        log_pi, mu, log_sigma = mdn(y_norm.to(DEVICE))
        pi = log_pi.exp()                          # [B, K]
        B = y_norm.shape[0]
        sigma = log_sigma.exp()                    # [B, K, D]
        # repeat batch for n_samples
        pi_rep    = pi.unsqueeze(1).expand(B, n_samples, K)          # [B, S, K]
        mu_rep    = mu.unsqueeze(1).expand(B, n_samples, K, DESIGN_DIM)
        sigma_rep = sigma.unsqueeze(1).expand(B, n_samples, K, DESIGN_DIM)
        # sample component indices
        idx = torch.multinomial(pi_rep.reshape(B * n_samples, K),
                                num_samples=1).view(B, n_samples)    # [B, S]
        idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(B, n_samples, 1, DESIGN_DIM)
        mu_sel    = mu_rep.gather(2, idx_exp).squeeze(2)             # [B, S, D]
        sigma_sel = sigma_rep.gather(2, idx_exp).squeeze(2)
        eps = torch.randn_like(mu_sel)
        samples = mu_sel + sigma_sel * eps                           # [B, S, D]
    return samples


def validate_samples(fwd: ForwardNet, samples_norm: torch.Tensor,
                     y_target_norm: torch.Tensor,
                     x_norm: Normaliser, y_norm_obj: Normaliser) -> dict:
    """Score each candidate design via the forward surrogate; return best per query."""
    fwd.eval()
    B, S, D = samples_norm.shape
    results = {"best_design": [], "best_error": []}
    with torch.no_grad():
        for i in range(B):
            cands = samples_norm[i].to(DEVICE)          # [S, D]
            y_pred = fwd(cands)                          # [S, T]
            target = y_target_norm[i].unsqueeze(0).to(DEVICE)
            err = ((y_pred - target) ** 2).mean(-1)     # [S]
            best_idx = err.argmin()
            results["best_design"].append(cands[best_idx].cpu())
            results["best_error"].append(err[best_idx].item())
    return results


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("Generating simulation dataset...")
    X_raw, Y_raw = make_dataset(n=80_000)
    split = int(0.9 * len(X_raw))
    X_tr, X_te = X_raw[:split], X_raw[split:]
    Y_tr, Y_te = Y_raw[:split], Y_raw[split:]

    xn = Normaliser().fit(X_tr)
    yn = Normaliser().fit(Y_tr)
    X_tr_n, X_te_n = xn.transform(X_tr), xn.transform(X_te)
    Y_tr_n, Y_te_n = yn.transform(Y_tr), yn.transform(Y_te)

    # --- forward surrogate ---
    print("Training forward surrogate...")
    fwd_ds = TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(Y_tr_n))
    fwd_loader = DataLoader(fwd_ds, batch_size=BATCH, shuffle=True)
    fwd_net = ForwardNet().to(DEVICE)
    fwd_hist = train_forward(fwd_net, fwd_loader, epochs=40)
    fwd_net.eval()
    with torch.no_grad():
        te_pred = fwd_net(torch.from_numpy(X_te_n).to(DEVICE)).cpu().numpy()
    fwd_mse = float(np.mean((te_pred - Y_te_n) ** 2))
    print(f"  Forward surrogate test MSE (normalised): {fwd_mse:.5f}")

    # --- MDN inverse model ---
    print("Training MDN inverse model...")
    mdn_ds = TensorDataset(torch.from_numpy(Y_tr_n), torch.from_numpy(X_tr_n))
    mdn_loader = DataLoader(mdn_ds, batch_size=BATCH, shuffle=True)
    mdn_net = MDNHead().to(DEVICE)
    mdn_hist = train_mdn(mdn_net, mdn_loader, epochs=80)
    print(f"  MDN final NLL: {mdn_hist[-1]:.4f}")

    # --- inference demo ---
    print("Running inverse design inference on 8 test targets...")
    n_query = 8
    Y_q = torch.from_numpy(Y_te_n[:n_query])   # normalised targets
    samples = sample_designs(mdn_net, Y_q, n_samples=128)  # [8, 128, 5]

    res = validate_samples(fwd_net, samples, Y_q, xn, yn)

    print("\nQuery | Forward-model reconstruction error (normalised MSE)")
    print("-" * 55)
    for i, err in enumerate(res["best_error"]):
        print(f"  {i:2d}  |  {err:.6f}")

    mean_err = float(np.mean(res["best_error"]))
    print(f"\nMean best-sample error: {mean_err:.6f}")

    # verify a few designs in the original simulator
    print("\nVerifying best designs through original simulator (un-normalised):")
    print("-" * 55)
    for i in range(min(4, n_query)):
        x_hat_n  = res["best_design"][i].numpy()[None]          # [1, 5]
        x_hat    = xn.inverse(x_hat_n)
        y_hat    = simulate(x_hat)
        y_true   = yn.inverse(Y_te_n[i:i+1])
        sim_err  = float(np.mean((y_hat - y_true) ** 2))
        print(f"  query {i}: simulator MSE = {sim_err:.6f}  |  "
              f"target={y_true.ravel().round(3)}  "
              f"achieved={y_hat.ravel().round(3)}")
