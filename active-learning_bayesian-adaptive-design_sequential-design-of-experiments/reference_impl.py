"""ALINE: Joint amortized Bayesian inference and active data acquisition via transformer + RL."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ---------------------------------------------------------------------------
# Toy problem: estimate mean mu ~ N(0, 1) from noisy observations y = mu + eps
# The agent chooses query locations x in [-1, 1]; obs noise sigma_obs is fixed.
# ---------------------------------------------------------------------------

SIGMA_PRIOR = 1.0
SIGMA_OBS = 0.5
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
MAX_STEPS = 8


def sample_prior(batch: int) -> torch.Tensor:
    """Draw latent means from the prior N(0, sigma_prior^2)."""
    return torch.randn(batch) * SIGMA_PRIOR


def observe(mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return noisy observation y = mu + N(0, sigma_obs^2) at design x."""
    return mu + torch.randn_like(x) * SIGMA_OBS


def analytic_posterior(
    xs: torch.Tensor, ys: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form Gaussian posterior mean and std given (x, y) history."""
    # For y = mu + eps, x does not affect likelihood; posterior is conjugate.
    n = xs.shape[-1]
    prior_var = SIGMA_PRIOR ** 2
    obs_var = SIGMA_OBS ** 2
    post_var = 1.0 / (1.0 / prior_var + n / obs_var)
    post_mean = post_var * ys.sum(-1) / obs_var
    return post_mean, post_var ** 0.5


def info_gain_analytic(n_obs: int) -> float:
    """Analytic KL(posterior || prior) averaged over the generative model."""
    prior_var = SIGMA_PRIOR ** 2
    obs_var = SIGMA_OBS ** 2
    post_var = 1.0 / (1.0 / prior_var + n_obs / obs_var)
    kl = 0.5 * (prior_var / post_var - 1 - np.log(prior_var / post_var))
    return float(kl)


# ---------------------------------------------------------------------------
# Positional encoding and transformer encoder
# ---------------------------------------------------------------------------


class SinusoidalPE(nn.Module):
    """Add sinusoidal positional encoding to a sequence."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add PE to token sequence (B, T, D)."""
        return x + self.pe[: x.shape[1]]


class ALINEModel(nn.Module):
    """Transformer that maps observation history to posterior params and next design."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
    ):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pe = SinusoidalPE(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Posterior head: outputs (mean, log_std) of approximate posterior q(mu|D)
        self.post_head = nn.Linear(d_model, 2)
        # Design head: outputs (mean, log_std) of proposal distribution pi(x|D)
        self.design_head = nn.Linear(d_model, 2)
        # Learnable [CLS] token for aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(
        self, xs: torch.Tensor, ys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning posterior (mu_q, sig_q) and design proposal (mu_x, sig_x).

        xs: (B, T), ys: (B, T) — history of designs and observations.
        Returns mu_q, sig_q, mu_x, sig_x each of shape (B,).
        """
        B, T = xs.shape
        tokens = torch.stack([xs, ys], dim=-1)          # (B, T, 2)
        tokens = self.input_proj(tokens)                  # (B, T, D)
        cls = self.cls_token.expand(B, 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, T+1, D)
        tokens = self.pe(tokens)
        out = self.encoder(tokens)                        # (B, T+1, D)
        agg = out[:, 0]                                   # CLS output

        p = self.post_head(agg)
        mu_q = p[:, 0]
        sig_q = F.softplus(p[:, 1]) + 1e-3

        d = self.design_head(agg)
        mu_x = torch.tanh(d[:, 0])                       # bound to (-1, 1)
        sig_x = F.softplus(d[:, 1]) + 1e-3

        return mu_q, sig_q, mu_x, sig_x


# ---------------------------------------------------------------------------
# Information-gain reward: KL( q(mu|D_new) || q(mu|D_old) )
# ---------------------------------------------------------------------------


def kl_gaussians(
    mu1: torch.Tensor,
    sig1: torch.Tensor,
    mu2: torch.Tensor,
    sig2: torch.Tensor,
) -> torch.Tensor:
    """KL divergence KL(N(mu1,sig1^2) || N(mu2,sig2^2)), per batch element."""
    return (
        torch.log(sig2 / sig1)
        + (sig1 ** 2 + (mu1 - mu2) ** 2) / (2 * sig2 ** 2)
        - 0.5
    )


# ---------------------------------------------------------------------------
# REINFORCE training loop
# ---------------------------------------------------------------------------


def run_episode(
    model: ALINEModel,
    batch: int,
    n_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Roll out one active-learning episode, returning log-probs, rewards, posterior loss.

    Returns log_probs (B, T), rewards (B, T), post_loss scalar.
    """
    mu_true = sample_prior(batch)                        # (B,)
    xs_hist = torch.zeros(batch, 0)
    ys_hist = torch.zeros(batch, 0)

    log_probs = []
    rewards = []

    # Prior as initial posterior approximation
    mu_q_prev = torch.zeros(batch)
    sig_q_prev = torch.ones(batch) * SIGMA_PRIOR

    for _ in range(n_steps):
        # Pad history with a dummy token when empty so the model always sees T>=1
        if xs_hist.shape[1] == 0:
            xs_in = torch.zeros(batch, 1)
            ys_in = torch.zeros(batch, 1)
        else:
            xs_in = xs_hist
            ys_in = ys_hist

        mu_q, sig_q, mu_x, sig_x = model(xs_in, ys_in)

        # Sample next design from proposal
        dist_x = Normal(mu_x, sig_x)
        x_new = dist_x.rsample().detach()
        x_new = x_new.clamp(-1.0, 1.0)
        log_prob = dist_x.log_prob(x_new)

        # Collect observation
        y_new = observe(mu_true, x_new)

        # Append to history
        xs_hist = torch.cat([xs_hist, x_new.unsqueeze(1)], dim=1)
        ys_hist = torch.cat([ys_hist, y_new.unsqueeze(1)], dim=1)

        # Reward: KL( q_new || q_old ) — information gain proxy
        reward = kl_gaussians(mu_q, sig_q, mu_q_prev.detach(), sig_q_prev.detach())
        reward = reward.clamp(min=0.0)

        log_probs.append(log_prob)
        rewards.append(reward)

        mu_q_prev = mu_q.detach()
        sig_q_prev = sig_q.detach()

    log_probs = torch.stack(log_probs, dim=1)            # (B, T)
    rewards = torch.stack(rewards, dim=1)                # (B, T)

    # Posterior supervised loss: NLL of mu_true under q(mu|D_final)
    mu_q_final, sig_q_final, _, _ = model(xs_hist, ys_hist)
    post_loss = -Normal(mu_q_final, sig_q_final).log_prob(mu_true).mean()

    return log_probs, rewards, post_loss


def train(
    n_iter: int = 2000,
    batch: int = 64,
    n_steps: int = MAX_STEPS,
    lr: float = 3e-4,
    gamma: float = 0.99,
    reinforce_weight: float = 1.0,
    post_weight: float = 1.0,
) -> ALINEModel:
    """Train the ALINE model with REINFORCE + posterior NLL."""
    model = ALINEModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for it in range(1, n_iter + 1):
        log_probs, rewards, post_loss = run_episode(model, batch, n_steps)

        # Discounted returns
        B, T = rewards.shape
        returns = torch.zeros_like(rewards)
        G = torch.zeros(B)
        for t in reversed(range(T)):
            G = rewards[:, t] + gamma * G
            returns[:, t] = G

        # Baseline: mean return across batch
        baseline = returns.mean(0, keepdim=True)
        advantage = returns - baseline

        reinforce_loss = -(log_probs * advantage.detach()).mean()
        loss = reinforce_weight * reinforce_loss + post_weight * post_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 200 == 0:
            r_mean = rewards.sum(1).mean().item()
            print(
                f"iter {it:4d} | loss {loss.item():.4f} | "
                f"mean_episode_reward {r_mean:.4f} | post_loss {post_loss.item():.4f}"
            )

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(model: ALINEModel, n_eval: int = 512, n_steps: int = MAX_STEPS) -> None:
    """Compare model posterior RMSE and analytic posterior RMSE after n_steps."""
    model.eval()
    with torch.no_grad():
        mu_true = sample_prior(n_eval)
        xs_hist = torch.zeros(n_eval, 0)
        ys_hist = torch.zeros(n_eval, 0)

        for _ in range(n_steps):
            if xs_hist.shape[1] == 0:
                xs_in = torch.zeros(n_eval, 1)
                ys_in = torch.zeros(n_eval, 1)
            else:
                xs_in = xs_hist
                ys_in = ys_hist

            _, _, mu_x, sig_x = model(xs_in, ys_in)
            x_new = Normal(mu_x, sig_x).sample().clamp(-1.0, 1.0)
            y_new = observe(mu_true, x_new)
            xs_hist = torch.cat([xs_hist, x_new.unsqueeze(1)], dim=1)
            ys_hist = torch.cat([ys_hist, y_new.unsqueeze(1)], dim=1)

        mu_q, sig_q, _, _ = model(xs_hist, ys_hist)
        model_rmse = (mu_q - mu_true).pow(2).mean().sqrt().item()

        mu_post, sig_post = analytic_posterior(xs_hist, ys_hist)
        analytic_rmse = (mu_post - mu_true).pow(2).mean().sqrt().item()

        # Expected analytic info gain after n_steps
        expected_ig = info_gain_analytic(n_steps)

    print(f"\nEvaluation over {n_eval} episodes, {n_steps} steps:")
    print(f"  Model posterior RMSE    : {model_rmse:.4f}")
    print(f"  Analytic posterior RMSE : {analytic_rmse:.4f}")
    print(f"  Expected analytic IG    : {expected_ig:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    print("Training ALINE on 1-D Gaussian mean estimation task...")
    model = train(n_iter=2000, batch=64, n_steps=MAX_STEPS)

    print("\nRunning evaluation...")
    evaluate(model, n_eval=512, n_steps=MAX_STEPS)
