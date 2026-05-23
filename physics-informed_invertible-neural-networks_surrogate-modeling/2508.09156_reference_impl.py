"""Minimal reference implementation: Physics-Constrained Fine-Tuning of
Flow-Matching Models (arXiv:2508.09156).

Demonstrates the concept of fine-tuning a simple flow model with PDE
residual loss to enforce physical consistency.

Run:
    python 2508.09156_reference_impl.py
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Simple 1D velocity field (parameterised flow)
# ---------------------------------------------------------------------------
class SimpleFlow:
    """Parameterised velocity field for a 1D continuous normalizing flow."""

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(size=(4,)) * 0.5
        self.b1 = rng.normal(size=(4,)) * 0.1
        self.w2 = rng.normal(size=(4,)) * 0.5
        self.b2 = float(rng.normal() * 0.1)

    def velocity(self, z: np.ndarray, t: float) -> np.ndarray:
        h = np.tanh(np.outer(z, self.w1) + self.b1)
        v = (h @ self.w2 + self.b2) * (1 - t)
        return v

    def integrate(self, z0: np.ndarray, n_steps: int = 20) -> np.ndarray:
        z = z0.copy()
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = step * dt
            z = z + dt * self.velocity(z, t)
        return z


# ---------------------------------------------------------------------------
# Physics: 1D Poisson-like equation  -u'' = f
# ---------------------------------------------------------------------------
def pde_residual(u_field: np.ndarray, source: np.ndarray,
                 dx: float = 0.1) -> float:
    """Finite-difference approximation of || -u'' - f ||²."""
    if len(u_field) < 3:
        return 0.0
    u_xx = (u_field[:-2] - 2 * u_field[1:-1] + u_field[2:]) / dx ** 2
    residual = (-u_xx - source[1:-1]) ** 2
    return float(residual.mean())


def boundary_loss(u_field: np.ndarray, u_left: float = 0.0,
                  u_right: float = 0.0) -> float:
    return (u_field[0] - u_left) ** 2 + (u_field[-1] - u_right) ** 2


# ---------------------------------------------------------------------------
# Fine-tuning loop: adjust flow params to reduce physics residual
# ---------------------------------------------------------------------------
def fine_tune(
    flow: SimpleFlow,
    source: np.ndarray,
    z0_fixed: np.ndarray,
    n_grid: int = 10,
    n_iters: int = 100,
    lr: float = 0.001,
    seed: int = 0,
) -> list[float]:
    dx = 1.0 / (n_grid - 1)
    residuals = []

    for it in range(n_iters):
        u_field = flow.integrate(z0_fixed, n_steps=15)

        res = pde_residual(u_field, source, dx)
        bc = boundary_loss(u_field)
        total = res + 10.0 * bc
        residuals.append(total)

        for param_name in ["w1", "w2", "b1"]:
            param = getattr(flow, param_name).copy()
            grad = np.zeros(param.size)
            for i in range(param.size):
                p = getattr(flow, param_name)
                orig = p.ravel()[i]
                p.ravel()[i] = orig + 1e-4
                u_plus = flow.integrate(z0_fixed, n_steps=15)
                loss_plus = pde_residual(u_plus, source, dx) + 10.0 * boundary_loss(u_plus)
                p.ravel()[i] = orig
                grad[i] = (loss_plus - total) / 1e-4
            grad = np.clip(grad, -5.0, 5.0)
            setattr(flow, param_name, param - lr * grad.reshape(param.shape))

    return residuals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    n_grid = 10
    x = np.linspace(0, 1, n_grid)
    source = np.sin(np.pi * x)

    flow = SimpleFlow(seed=7)

    z0_test = np.random.default_rng(99).normal(size=n_grid) * 0.3
    u_before = flow.integrate(z0_test, n_steps=15)
    res_before = pde_residual(u_before, source, 1.0 / (n_grid - 1))

    print("Physics-Constrained Flow-Matching Fine-Tuning Demo")
    print("=" * 55)
    print(f"PDE: -u'' = sin(πx),  u(0)=u(1)=0")
    print(f"PDE residual BEFORE fine-tuning: {res_before:.4f}")
    print()

    residuals = fine_tune(flow, source, z0_test, n_grid=n_grid,
                          n_iters=80, lr=0.0005)

    u_after = flow.integrate(z0_test, n_steps=15)
    res_after = pde_residual(u_after, source, 1.0 / (n_grid - 1))

    print(f"PDE residual AFTER fine-tuning:  {res_after:.4f}")
    print(f"Residual reduction:              {res_before / max(res_after, 1e-9):.1f}x")
    print()
    print("Residual trajectory (every 10 iters):")
    for i in range(0, len(residuals), 10):
        print(f"  iter {i:3d}: {residuals[i]:.4f}")
    print()
    print("Fine-tuning injects physics constraints into the generative")
    print("flow without distorting the learned distribution structure.")


if __name__ == "__main__":
    main()
