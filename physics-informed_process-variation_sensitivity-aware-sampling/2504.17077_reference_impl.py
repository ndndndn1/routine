"""Minimal reference implementation: AdjointDiffusion — physics-guided
diffusion sampling with adjoint sensitivity for photonic inverse design
(arXiv:2504.17077).

Demonstrates adjoint-guided reverse diffusion to generate designs that
satisfy physics objectives and fabrication constraints.

Run:
    python 2504.17077_reference_impl.py
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic physics model + adjoint gradient
# ---------------------------------------------------------------------------
def physics_fom(design: np.ndarray) -> float:
    """Figure of merit: higher is better."""
    return float(np.sin(design[0] * 2) * np.cos(design[1]) + 0.5)


def adjoint_gradient(design: np.ndarray, delta: float = 1e-4) -> np.ndarray:
    """Numerical adjoint gradient of the FoM."""
    grad = np.zeros_like(design)
    f0 = physics_fom(design)
    for i in range(len(design)):
        d = design.copy()
        d[i] += delta
        grad[i] = (physics_fom(d) - f0) / delta
    return grad


# ---------------------------------------------------------------------------
# Fabrication constraint: minimum feature size proxy
# ---------------------------------------------------------------------------
def fabrication_filter(design: np.ndarray, min_val: float = -1.5,
                       max_val: float = 1.5) -> np.ndarray:
    return np.clip(design, min_val, max_val)


# ---------------------------------------------------------------------------
# Simple diffusion: forward (add noise) + reverse (denoise + physics guide)
# ---------------------------------------------------------------------------
def forward_diffusion(x0: np.ndarray, t: int, T: int) -> np.ndarray:
    """Add noise proportional to timestep."""
    noise_scale = (t / T) * 2.0
    return x0 + np.random.randn(*x0.shape) * noise_scale


def reverse_step(
    x_t: np.ndarray,
    t: int,
    T: int,
    target_fom: float,
    guidance_scale: float = 0.3,
) -> np.ndarray:
    """One reverse diffusion step with adjoint guidance."""
    noise_scale = (t / T) * 0.5
    x_denoised = x_t + np.random.randn(*x_t.shape) * noise_scale * 0.1

    grad = adjoint_gradient(x_denoised)
    x_guided = x_denoised + guidance_scale * grad

    x_filtered = fabrication_filter(x_guided)
    return x_filtered


# ---------------------------------------------------------------------------
# Full reverse diffusion with adjoint guidance
# ---------------------------------------------------------------------------
def adjoint_guided_diffusion(
    dim: int = 2,
    T: int = 20,
    target_fom: float = 1.0,
    n_samples: int = 5,
    guidance_scale: float = 0.3,
    seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    results = []

    for s in range(n_samples):
        np.random.seed(seed + s)
        x = rng.normal(size=dim) * 2.0

        for t in range(T, 0, -1):
            x = reverse_step(x, t, T, target_fom, guidance_scale)

        fom = physics_fom(x)
        in_fab = np.all(np.abs(x) <= 1.5)
        results.append({
            "sample": s,
            "design": x.copy(),
            "fom": fom,
            "fabricable": in_fab,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("AdjointDiffusion: physics-guided reverse diffusion demo")
    print("=" * 58)

    results = adjoint_guided_diffusion(
        dim=2, T=30, n_samples=8, guidance_scale=0.4, seed=7
    )

    for r in results:
        fab = "YES" if r["fabricable"] else "NO"
        print(
            f"  sample {r['sample']} | design=({r['design'][0]:+.3f}, "
            f"{r['design'][1]:+.3f}) | FoM={r['fom']:.4f} | fabricable={fab}"
        )

    foms = [r["fom"] for r in results]
    fab_count = sum(1 for r in results if r["fabricable"])
    print(f"\nMean FoM: {np.mean(foms):.4f} (higher is better)")
    print(f"Fabricable designs: {fab_count}/{len(results)}")
    print("\nAdjoint guidance steers diffusion toward high-FoM,")
    print("fabrication-feasible designs at each reverse step.")


if __name__ == "__main__":
    main()
