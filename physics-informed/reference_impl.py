"""Reference implementation sketch for
"Pretrained Video Models as Differentiable Physics Simulators for Urban Wind Flows"
(Perini et al., 2026, arXiv:2603.21210).

This is *not* the official code. It is a minimal, self-contained PyTorch sketch
that captures the core ideas relevant to the `variation propagation` topic:

    1. A (toy) differentiable surrogate for a CFD-like flow field that takes
       a geometry/parameter tensor as input and produces a velocity field.
    2. A physics-informed decoder loss approximating a PDE residual
       (here: a 2D incompressibility / divergence-free constraint).
    3. Gradient-based inverse design: optimize geometry parameters so a
       downstream objective (e.g. pedestrian wind comfort) is minimized,
       propagating variations through the surrogate.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Toy differentiable surrogate (stand-in for a pretrained video-diffusion CFD).
# ---------------------------------------------------------------------------
class ToyFlowSurrogate(nn.Module):
    """Maps a (B, C_geom, H, W) geometry/parameter map to a (B, 2, H, W)
    2-D velocity field (u, v). In the real paper this is a latent video
    transformer initialized from a pretrained video diffusion model."""

    def __init__(self, c_geom: int = 1, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_geom, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        return self.net(geom)


# ---------------------------------------------------------------------------
# Physics-informed decoder loss: divergence-free residual (incompressibility).
# ---------------------------------------------------------------------------
def divergence_residual(velocity: torch.Tensor) -> torch.Tensor:
    """Finite-difference divergence of a 2D vector field. velocity: (B, 2, H, W)."""
    u = velocity[:, 0]
    v = velocity[:, 1]
    du_dx = u[:, :, 2:] - u[:, :, :-2]
    dv_dy = v[:, 2:, :] - v[:, :-2, :]
    # Align shapes by cropping to the common interior.
    du_dx = du_dx[:, 1:-1, :]
    dv_dy = dv_dy[:, :, 1:-1]
    return du_dx + dv_dy  # (B, H-2, W-2)


def physics_informed_loss(velocity: torch.Tensor) -> torch.Tensor:
    return divergence_residual(velocity).pow(2).mean()


# ---------------------------------------------------------------------------
# Downstream design objective: e.g. penalize pedestrian-height wind speed.
# ---------------------------------------------------------------------------
def pedestrian_wind_discomfort(velocity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    speed = velocity.pow(2).sum(dim=1).sqrt()          # (B, H, W)
    # Discomfort: mean squared speed on pedestrian mask, clamp above a threshold.
    return (F.relu(speed - 1.0) * mask).mean()


# ---------------------------------------------------------------------------
# Surrogate training (supervised on synthetic "CFD" data).
# ---------------------------------------------------------------------------
def synthetic_cfd(geom: torch.Tensor) -> torch.Tensor:
    """Cheap, differentiable 'ground-truth' to stand in for an expensive CFD."""
    # Rotational field around the geometry's center of mass + sinusoidal perturbation.
    B, _, H, W = geom.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=geom.device),
        torch.linspace(-1, 1, W, device=geom.device),
        indexing="ij",
    )
    u = -ys.unsqueeze(0).expand(B, -1, -1) + 0.1 * torch.sin(3 * xs)
    v = xs.unsqueeze(0).expand(B, -1, -1) + 0.1 * torch.cos(3 * ys)
    # Modulate with geometry so the surrogate has something to learn.
    u = u * (1 - geom[:, 0])
    v = v * (1 - geom[:, 0])
    return torch.stack([u, v], dim=1)


def train_surrogate(model: nn.Module, steps: int = 200, batch: int = 8,
                    grid: int = 32, lam_phys: float = 0.1) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        geom = (torch.rand(batch, 1, grid, grid) > 0.85).float()
        target = synthetic_cfd(geom)
        pred = model(geom)
        loss_data = F.mse_loss(pred, target)
        loss_phys = physics_informed_loss(pred)
        loss = loss_data + lam_phys * loss_phys
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"[train] step={step} data={loss_data.item():.4f} "
                  f"phys={loss_phys.item():.4f}")


# ---------------------------------------------------------------------------
# Inverse design: optimize geometry parameters through the frozen surrogate.
# ---------------------------------------------------------------------------
def inverse_design(model: nn.Module, steps: int = 100, grid: int = 32) -> torch.Tensor:
    for p in model.parameters():
        p.requires_grad_(False)
    # Continuous relaxation of a building-placement mask.
    geom_logits = torch.zeros(1, 1, grid, grid, requires_grad=True)
    mask = torch.ones(1, grid, grid)
    opt = torch.optim.Adam([geom_logits], lr=5e-2)
    for step in range(steps):
        geom = torch.sigmoid(geom_logits)
        velocity = model(geom)
        obj = pedestrian_wind_discomfort(velocity, mask)
        # Regularize to avoid trivial 'build everywhere' solution.
        reg = 1e-2 * geom.mean()
        loss = obj + reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"[inverse] step={step} obj={obj.item():.4f} "
                  f"coverage={geom.mean().item():.3f}")
    return torch.sigmoid(geom_logits).detach()


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ToyFlowSurrogate()
    train_surrogate(model)
    geom = inverse_design(model)
    print("Final geometry mean coverage:", float(geom.mean()))
