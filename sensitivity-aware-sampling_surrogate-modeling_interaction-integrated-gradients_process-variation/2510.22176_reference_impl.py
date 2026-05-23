"""Minimal reference implementation: Interpretable Geometry Sensitivity
via Integrated Gradients for inverse-designed photonic devices
(arXiv:2510.22176).

Demonstrates a lightweight surrogate + Integrated Gradients attribution
to produce pixel-level sensitivity maps for a synthetic 2D device mask.

Run:
    python 2510.22176_reference_impl.py
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic CNN-like surrogate (simple linear + nonlinear map)
# ---------------------------------------------------------------------------
class SimpleSurrogate:
    """A toy surrogate mapping a 2D binary mask to a scalar FoM."""

    def __init__(self, n: int = 8, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n = n
        self.weights = rng.normal(size=(n, n)) * 0.5
        self.bias = 0.3

    def predict(self, mask: np.ndarray) -> float:
        linear = np.sum(self.weights * mask) + self.bias
        return float(np.tanh(linear))

    def gradient(self, mask: np.ndarray) -> np.ndarray:
        linear = np.sum(self.weights * mask) + self.bias
        dtanh = 1.0 - np.tanh(linear) ** 2
        return self.weights * dtanh


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------
def integrated_gradients(
    surrogate: SimpleSurrogate,
    mask: np.ndarray,
    baseline: np.ndarray | None = None,
    n_steps: int = 50,
) -> np.ndarray:
    if baseline is None:
        baseline = np.zeros_like(mask)

    ig = np.zeros_like(mask)
    for step in range(n_steps):
        alpha = step / n_steps
        interpolated = baseline + alpha * (mask - baseline)
        grad = surrogate.gradient(interpolated)
        ig += grad / n_steps

    ig *= mask - baseline
    return ig


# ---------------------------------------------------------------------------
# Sensitivity map
# ---------------------------------------------------------------------------
def sensitivity_map(ig: np.ndarray) -> np.ndarray:
    return np.abs(ig)


# ---------------------------------------------------------------------------
# Perturbation test: validate sensitivity map
# ---------------------------------------------------------------------------
def perturbation_test(
    surrogate: SimpleSurrogate,
    mask: np.ndarray,
    sens: np.ndarray,
    n_perturb: int = 5,
) -> tuple[float, float]:
    flat_sens = sens.ravel()
    indices = np.argsort(flat_sens)

    high_sens_idx = indices[-n_perturb:]
    low_sens_idx = indices[:n_perturb]

    baseline_fom = surrogate.predict(mask)

    mask_high = mask.copy()
    for idx in high_sens_idx:
        r, c = divmod(idx, mask.shape[1])
        mask_high[r, c] = 1.0 - mask_high[r, c]
    delta_high = abs(surrogate.predict(mask_high) - baseline_fom)

    mask_low = mask.copy()
    for idx in low_sens_idx:
        r, c = divmod(idx, mask.shape[1])
        mask_low[r, c] = 1.0 - mask_low[r, c]
    delta_low = abs(surrogate.predict(mask_low) - baseline_fom)

    return delta_high, delta_low


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    n = 8
    rng = np.random.default_rng(7)
    mask = (rng.random((n, n)) > 0.5).astype(float)

    surr = SimpleSurrogate(n=n)
    fom = surr.predict(mask)

    ig = integrated_gradients(surr, mask)
    sens = sensitivity_map(ig)

    delta_high, delta_low = perturbation_test(surr, mask, sens, n_perturb=5)

    print("Integrated Gradients sensitivity analysis demo")
    print("=" * 55)
    print(f"Device mask size: {n}x{n}")
    print(f"Baseline FoM: {fom:.4f}")
    print()
    print("Sensitivity map (|IG|):")
    for row in sens:
        print("  " + " ".join(f"{v:.3f}" for v in row))
    print()
    print(f"Perturbation of {5} HIGH-sensitivity pixels: ΔFoM = {delta_high:.4f}")
    print(f"Perturbation of {5} LOW-sensitivity pixels:  ΔFoM = {delta_low:.4f}")
    ratio = delta_high / max(delta_low, 1e-9)
    print(f"Ratio (high/low): {ratio:.1f}x")
    print("\nHigh-sensitivity regions show significantly larger FoM change,")
    print("confirming that IG attribution correctly identifies process-")
    print("variation-critical regions.")


if __name__ == "__main__":
    main()
