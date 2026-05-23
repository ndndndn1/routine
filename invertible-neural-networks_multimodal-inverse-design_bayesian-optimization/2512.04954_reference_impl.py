"""Minimal reference implementation: Amortized Inference of Multi-Modal
Posteriors using Likelihood-Weighted Normalizing Flows (arXiv:2512.04954).

Demonstrates how a GMM base distribution avoids spurious probability
bridges between disconnected posterior modes, compared to a standard
Gaussian base.

Run:
    python 2512.04954_reference_impl.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Gaussian Mixture Model base distribution
# ---------------------------------------------------------------------------
class GMMBase:
    def __init__(self, means: list[float], stds: list[float], weights: list[float]):
        self.means = np.array(means)
        self.stds = np.array(stds)
        w = np.array(weights)
        self.weights = w / w.sum()
        self.k = len(means)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        comp = rng.choice(self.k, size=n, p=self.weights)
        return rng.normal(loc=self.means[comp], scale=self.stds[comp])

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        log_ps = np.array([
            np.log(w + 1e-30) + norm.logpdf(z, loc=m, scale=s)
            for w, m, s in zip(self.weights, self.means, self.stds)
        ])
        return np.log(np.exp(log_ps).sum(axis=0) + 1e-30)


class GaussianBase:
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=n)

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        return norm.logpdf(z)


# ---------------------------------------------------------------------------
# Simple affine flow layer (1D)
# ---------------------------------------------------------------------------
class AffineFlow:
    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        self.scale = scale
        self.shift = shift

    def forward(self, z: np.ndarray) -> np.ndarray:
        return self.scale * z + self.shift

    def log_det_jacobian(self) -> float:
        return np.log(abs(self.scale))


# ---------------------------------------------------------------------------
# Target multimodal posterior (synthetic)
# ---------------------------------------------------------------------------
def target_log_posterior(theta: np.ndarray) -> np.ndarray:
    mode1 = norm.logpdf(theta, loc=-2.0, scale=0.4)
    mode2 = norm.logpdf(theta, loc=2.0, scale=0.4)
    return np.log(0.5 * np.exp(mode1) + 0.5 * np.exp(mode2) + 1e-30)


# ---------------------------------------------------------------------------
# Likelihood-weighted evaluation
# ---------------------------------------------------------------------------
def evaluate_approximation(
    base, flow: AffineFlow, n_samples: int = 5000, seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)

    z = base.sample(n_samples, rng)
    theta = flow.forward(z)

    log_q = base.log_prob(z) - flow.log_det_jacobian()
    log_p = target_log_posterior(theta)

    log_w = log_p - log_q
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum() + 1e-12

    ess = 1.0 / np.sum(w ** 2)
    ess_ratio = ess / n_samples

    in_bridge = (theta > -1.0) & (theta < 1.0)
    bridge_mass = np.exp(log_q[in_bridge]).mean() if in_bridge.sum() > 0 else 0.0

    return {
        "ess_ratio": ess_ratio,
        "bridge_mass": bridge_mass,
        "theta_mean": np.sum(w * theta),
        "theta_std": np.sqrt(np.sum(w * (theta - np.sum(w * theta)) ** 2)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    flow = AffineFlow(scale=2.5, shift=0.0)

    gauss_base = GaussianBase()
    gmm_base = GMMBase(means=[-0.8, 0.8], stds=[0.3, 0.3], weights=[0.5, 0.5])

    res_gauss = evaluate_approximation(gauss_base, flow)
    res_gmm = evaluate_approximation(gmm_base, flow)

    print("Likelihood-Weighted NF: Gaussian vs GMM base")
    print("=" * 55)
    print("Target: bimodal posterior with modes at θ = -2 and θ = +2")
    print()
    print("Gaussian base distribution:")
    print(f"  ESS ratio:        {res_gauss['ess_ratio']:.4f}")
    print(f"  Bridge mass:      {res_gauss['bridge_mass']:.6f}")
    print(f"  Posterior std:     {res_gauss['theta_std']:.4f}")
    print()
    print("GMM base distribution (2 components):")
    print(f"  ESS ratio:        {res_gmm['ess_ratio']:.4f}")
    print(f"  Bridge mass:      {res_gmm['bridge_mass']:.6f}")
    print(f"  Posterior std:     {res_gmm['theta_std']:.4f}")
    print()

    if res_gmm["bridge_mass"] < res_gauss["bridge_mass"]:
        print("GMM base produces LESS spurious bridge mass → better topology")
    else:
        print("Both bases show comparable bridge mass in this simplified demo")

    print("\nGMM initialisation matching mode cardinality preserves")
    print("disconnected support and avoids probability bridges.")


if __name__ == "__main__":
    main()
