"""
Reference implementation of Multi-modal BNN Surrogates with Conjugate Last-Layer Estimation.
arXiv:2509.21711 — Taylor, Mueller, Bessac (2025).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ---------------------------------------------------------------------------
# Encoder branches
# ---------------------------------------------------------------------------

class ModalityEncoder(nn.Module):
    """Single-modality MLP encoder producing a fixed-size embedding."""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw modality features to an embedding vector."""
        return self.net(x)


class MultiModalEncoder(nn.Module):
    """Fuse embeddings from multiple modality branches, handling missing inputs via masks."""

    def __init__(self, input_dims: list[int], hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.encoders = nn.ModuleList(
            [ModalityEncoder(d, hidden_dim, embed_dim) for d in input_dims]
        )
        self.embed_dim = embed_dim
        self.n_modalities = len(input_dims)

    def forward(
        self,
        modality_inputs: list[torch.Tensor],
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate modality embeddings; missing modalities (mask==0) contribute zeros.

        Args:
            modality_inputs: list of (N, d_m) tensors, one per modality.
            masks: (N, M) binary tensor; masks[i, m]=1 if modality m is observed for sample i.
        Returns:
            (N, embed_dim) fused representation.
        """
        N = modality_inputs[0].shape[0]
        fused = torch.zeros(N, self.embed_dim, device=modality_inputs[0].device)
        present_count = masks.sum(dim=1, keepdim=True).clamp(min=1.0)  # (N, 1)

        for m, (enc, x_m) in enumerate(zip(self.encoders, modality_inputs)):
            emb = enc(x_m)                           # (N, embed_dim)
            mask_m = masks[:, m].unsqueeze(1)        # (N, 1)
            fused = fused + mask_m * emb

        return fused / present_count                 # mean-pool over present modalities


# ---------------------------------------------------------------------------
# Conjugate Bayesian last layer (Normal-Inverse-Gamma)
# ---------------------------------------------------------------------------

class ConjugateLastLayer(nn.Module):
    """
    Bayesian linear output layer with Normal-Inverse-Gamma (NIG) prior on (w, sigma^2).

    Prior:  sigma^2 ~ InvGamma(a0, b0),  w | sigma^2 ~ N(m0, sigma^2 * V0^{-1})
    Closed-form posterior after observing (Phi, y):
        a_n = a0 + N/2
        b_n = b0 + 0.5*(y'y + m0'V0 m0 - m_n'V_n^{-1} m_n)
        V_n  = V0 + Phi'Phi
        m_n  = V_n^{-1} (V0 m0 + Phi'y)
    Predictive: Student-t with df=2*a_n.
    """

    def __init__(self, in_features: int, a0: float = 1.0, b0: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.register_buffer("a0", torch.tensor(a0))
        self.register_buffer("b0", torch.tensor(b0))
        # Prior precision matrix V0 = I (stored as diagonal for efficiency)
        self.register_buffer("V0_diag", torch.ones(in_features))
        self.register_buffer("m0", torch.zeros(in_features))

        # Posterior parameters (updated in-place by fit())
        self.register_buffer("m_n", torch.zeros(in_features))
        self.register_buffer("V_n", torch.eye(in_features))
        self.register_buffer("a_n", torch.tensor(a0))
        self.register_buffer("b_n", torch.tensor(b0))
        self._fitted = False

    def fit(self, Phi: torch.Tensor, y: torch.Tensor) -> None:
        """Update NIG posterior analytically given design matrix Phi (N, D) and targets y (N,)."""
        V0 = torch.diag(self.V0_diag)
        V_n = V0 + Phi.T @ Phi
        V_n_inv = torch.linalg.inv(V_n)
        m_n = V_n_inv @ (V0 @ self.m0 + Phi.T @ y)

        N = y.shape[0]
        a_n = self.a0 + N / 2.0
        b_n = (
            self.b0
            + 0.5 * (y @ y + self.m0 @ (V0 @ self.m0) - m_n @ (V_n @ m_n))
        )

        self.m_n = m_n
        self.V_n = V_n
        self.a_n = a_n
        self.b_n = b_n.clamp(min=1e-6)
        self._fitted = True

    def predict(self, Phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return predictive mean and variance for new design rows Phi (N, D).

        Predictive distribution is Student-t(2*a_n) with the returned mean/variance.
        """
        mean = Phi @ self.m_n                              # (N,)
        sigma2_mean = self.b_n / (self.a_n - 1.0)        # posterior mean of sigma^2
        V_n_inv = torch.linalg.inv(self.V_n)
        # Per-sample predictive variance: sigma2_mean * (1 + phi_i' V_n^{-1} phi_i)
        extra = (Phi @ V_n_inv * Phi).sum(dim=1)          # (N,)
        var = sigma2_mean * (1.0 + extra)
        return mean, var.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Full surrogate model
# ---------------------------------------------------------------------------

class MultiModalBNNSurrogate(nn.Module):
    """Multi-modal BNN surrogate: learned encoders + conjugate Bayesian output layer."""

    def __init__(
        self,
        input_dims: list[int],
        hidden_dim: int = 64,
        embed_dim: int = 32,
        a0: float = 1.0,
        b0: float = 1.0,
    ):
        super().__init__()
        self.encoder = MultiModalEncoder(input_dims, hidden_dim, embed_dim)
        # +1 for bias absorbed into feature vector
        self.bayes_layer = ConjugateLastLayer(embed_dim + 1, a0=a0, b0=b0)

    def _design_matrix(self, modality_inputs, masks):
        """Construct augmented design matrix with bias column."""
        phi = self.encoder(modality_inputs, masks)          # (N, embed_dim)
        ones = torch.ones(phi.shape[0], 1, device=phi.device)
        return torch.cat([phi, ones], dim=1)                # (N, embed_dim+1)

    def fit_last_layer(
        self,
        modality_inputs: list[torch.Tensor],
        masks: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """Freeze encoder, compute closed-form NIG posterior for last layer."""
        with torch.no_grad():
            Phi = self._design_matrix(modality_inputs, masks)
        self.bayes_layer.fit(Phi, y)

    def predict(
        self,
        modality_inputs: list[torch.Tensor],
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior predictive mean and variance."""
        with torch.no_grad():
            Phi = self._design_matrix(modality_inputs, masks)
        return self.bayes_layer.predict(Phi)

    def elbo_loss(
        self,
        modality_inputs: list[torch.Tensor],
        masks: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """SVI surrogate loss: NLL under current NIG predictive, used to train the encoder."""
        Phi = self._design_matrix(modality_inputs, masks)
        mean, var = self.bayes_layer.predict(Phi)
        # Gaussian NLL as a tractable proxy for the Student-t NLL
        nll = 0.5 * (torch.log(var) + (y - mean) ** 2 / var).mean()
        return nll


# ---------------------------------------------------------------------------
# Toy dataset
# ---------------------------------------------------------------------------

def make_toy_dataset(N: int = 300, seed: int = 0):
    """Generate a 3-modality toy regression dataset with 30% random missing entries."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(N,)).astype(np.float32)

    # True function (non-linear)
    y_true = np.sin(x) + 0.3 * x ** 2 - 0.5 * np.cos(2 * x)

    # Three sensor modalities: signal + independent noise
    s_a = (x + rng.normal(0, 0.3, N)).reshape(-1, 1).astype(np.float32)
    s_b = (np.stack([np.sin(x), np.cos(x)], axis=1) + rng.normal(0, 0.2, (N, 2))).astype(np.float32)
    s_c = (np.stack([x, x ** 2, x ** 3], axis=1) / 10 + rng.normal(0, 0.1, (N, 3))).astype(np.float32)

    y = (y_true + rng.normal(0, 0.15, N)).astype(np.float32)

    # Random missing mask: each modality independently missing 30% of the time
    mask = rng.binomial(1, 0.7, size=(N, 3)).astype(np.float32)
    # Guarantee at least one modality per sample
    all_missing = mask.sum(axis=1) == 0
    mask[all_missing, rng.integers(0, 3, size=all_missing.sum())] = 1.0

    return s_a, s_b, s_c, mask, y, x


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model: MultiModalBNNSurrogate, data: tuple, n_epochs: int = 300, lr: float = 1e-3):
    """Alternate between encoder SVI updates and closed-form last-layer posterior updates."""
    s_a, s_b, s_c, mask, y, _ = data
    inputs = [torch.tensor(s_a), torch.tensor(s_b), torch.tensor(s_c)]
    mask_t = torch.tensor(mask)
    y_t = torch.tensor(y)

    optimizer = Adam(model.encoder.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        # Temporarily update last layer (no grad) for current encoder state
        with torch.no_grad():
            Phi = model._design_matrix(inputs, mask_t)
        model.bayes_layer.fit(Phi, y_t)

        # Back-prop through encoder using predictive NLL
        loss = model.elbo_loss(inputs, mask_t, y_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:4d}/{n_epochs}  loss={loss.item():.4f}")

    # Final closed-form update with converged encoder
    model.eval()
    model.fit_last_layer(inputs, mask_t, y_t)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("Generating toy 3-modality dataset ...")
    data = make_toy_dataset(N=400)
    s_a, s_b, s_c, mask, y, x_raw = data

    model = MultiModalBNNSurrogate(
        input_dims=[1, 2, 3],
        hidden_dim=64,
        embed_dim=32,
        a0=2.0,
        b0=1.0,
    )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training ...")
    train(model, data, n_epochs=300, lr=1e-3)

    # Evaluation on a dense test grid (all modalities present)
    model.eval()
    x_test = np.linspace(-3, 3, 200).astype(np.float32)
    s_a_test = x_test.reshape(-1, 1)
    s_b_test = np.stack([np.sin(x_test), np.cos(x_test)], axis=1)
    s_c_test = np.stack([x_test, x_test ** 2, x_test ** 3], axis=1) / 10
    mask_test = np.ones((200, 3), dtype=np.float32)

    test_inputs = [
        torch.tensor(s_a_test),
        torch.tensor(s_b_test),
        torch.tensor(s_c_test),
    ]
    mask_t = torch.tensor(mask_test)

    mean_pred, var_pred = model.predict(test_inputs, mask_t)
    std_pred = var_pred.sqrt()

    mean_np = mean_pred.numpy()
    std_np = std_pred.numpy()
    y_true = np.sin(x_test) + 0.3 * x_test ** 2 - 0.5 * np.cos(2 * x_test)

    rmse = np.sqrt(np.mean((mean_np - y_true) ** 2))
    # 95% coverage: fraction of true values inside mean ± 1.96*std
    coverage = np.mean(np.abs(y_true - mean_np) <= 1.96 * std_np)

    print(f"\nTest RMSE          : {rmse:.4f}")
    print(f"95% PI coverage    : {coverage*100:.1f}%  (target ~95%)")

    # Demonstrate missing-modality robustness
    mask_partial = np.tile([1.0, 0.0, 0.0], (200, 1)).astype(np.float32)  # only sensor A
    mean_partial, var_partial = model.predict(test_inputs, torch.tensor(mask_partial))
    rmse_partial = np.sqrt(np.mean((mean_partial.numpy() - y_true) ** 2))
    print(f"Test RMSE (sensor A only): {rmse_partial:.4f}")

    print("\nDone. Surrogate trained with conjugate Bayesian last layer.")
    print("Uncertainty bands widen automatically when modalities are missing.")
