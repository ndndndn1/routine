"""Minimal reference implementation: Surrogate-based Approximate Bayesian
Computation for inverse uncertainty quantification in manufacturing
(arXiv:2512.13354).

Demonstrates ABC with a simple surrogate to infer process parameters
from noisy thickness measurements, quantifying process variation.

Run:
    python 2512.13354_reference_impl.py
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Simple surrogate (stands in for XGBoost)
# ---------------------------------------------------------------------------
def surrogate_forward(params: np.ndarray) -> np.ndarray:
    """Maps process params (temperature, flow_rate) -> coating thickness."""
    temp, flow = params[:, 0], params[:, 1]
    thickness = 0.5 * temp + 0.3 * flow + 0.1 * temp * flow - 0.05 * temp ** 2
    return thickness


# ---------------------------------------------------------------------------
# ABC sampler
# ---------------------------------------------------------------------------
def abc_sample(
    y_obs: float,
    n_samples: int = 100_000,
    epsilon: float = 0.3,
    prior_bounds: tuple[tuple[float, float], ...] = ((-2, 2), (-2, 2)),
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = len(prior_bounds)
    lows = np.array([b[0] for b in prior_bounds])
    highs = np.array([b[1] for b in prior_bounds])

    theta_prior = rng.uniform(lows, highs, size=(n_samples, dim))
    y_sim = surrogate_forward(theta_prior)
    distances = np.abs(y_sim - y_obs)
    accepted = theta_prior[distances < epsilon]
    return accepted


# ---------------------------------------------------------------------------
# Weighted ABC with adaptive epsilon
# ---------------------------------------------------------------------------
def weighted_abc(
    y_obs: float,
    n_rounds: int = 3,
    n_per_round: int = 50_000,
    initial_eps: float = 1.0,
    shrink: float = 0.5,
    prior_bounds: tuple[tuple[float, float], ...] = ((-2, 2), (-2, 2)),
    seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    eps = initial_eps
    results = []

    for r in range(n_rounds):
        dim = len(prior_bounds)
        lows = np.array([b[0] for b in prior_bounds])
        highs = np.array([b[1] for b in prior_bounds])

        theta = rng.uniform(lows, highs, size=(n_per_round, dim))
        y_sim = surrogate_forward(theta)
        dist = np.abs(y_sim - y_obs)

        accepted = theta[dist < eps]
        acc_dist = dist[dist < eps]

        if len(accepted) > 0:
            weights = np.exp(-0.5 * (acc_dist / eps) ** 2)
            weights /= weights.sum()
            post_mean = (weights[:, None] * accepted).sum(axis=0)
            post_std = np.sqrt(
                (weights[:, None] * (accepted - post_mean) ** 2).sum(axis=0)
            )
        else:
            post_mean = np.full(dim, np.nan)
            post_std = np.full(dim, np.nan)

        results.append({
            "round": r,
            "epsilon": eps,
            "n_accepted": len(accepted),
            "posterior_mean": post_mean,
            "posterior_std": post_std,
        })

        eps *= shrink

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    true_params = np.array([0.8, 0.5])
    y_obs = surrogate_forward(true_params.reshape(1, -1))[0]
    y_obs += 0.05  # measurement noise

    print("Surrogate-based ABC for process variation quantification")
    print("=" * 60)
    print(f"True params:  temp={true_params[0]:.2f}, flow={true_params[1]:.2f}")
    print(f"Observed thickness: {y_obs:.4f}")
    print()

    results = weighted_abc(y_obs, n_rounds=4, initial_eps=0.8, shrink=0.5)

    for r in results:
        print(
            f"  Round {r['round']} | ε={r['epsilon']:.3f} | "
            f"accepted={r['n_accepted']:5d} | "
            f"mean=({r['posterior_mean'][0]:+.3f}, {r['posterior_mean'][1]:+.3f}) | "
            f"std=({r['posterior_std'][0]:.3f}, {r['posterior_std'][1]:.3f})"
        )

    final = results[-1]
    print(f"\nFinal posterior credible interval:")
    for i, name in enumerate(["temperature", "flow_rate"]):
        lo = final["posterior_mean"][i] - 2 * final["posterior_std"][i]
        hi = final["posterior_mean"][i] + 2 * final["posterior_std"][i]
        print(f"  {name}: [{lo:+.3f}, {hi:+.3f}] (true: {true_params[i]:.3f})")


if __name__ == "__main__":
    main()
