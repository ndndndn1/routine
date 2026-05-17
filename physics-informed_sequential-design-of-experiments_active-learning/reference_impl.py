"""
Reference implementation: PIED – Physics-Informed Experimental Design
Based on arXiv:2503.07070 (Hemachandra et al., ICLR 2025)

Official code: https://github.com/apivich-h/pied

Core idea: optimize sensor placement for PDE inverse problems using a
PINN's physics residual as the differentiable objective, with meta-learning
for fast adaptation across parameter samples.

Minimal NumPy/SciPy demo on a 1D heat equation inverse problem.
"""

import numpy as np
from typing import Tuple, List


def heat_equation_solution(x: np.ndarray, t: float, kappa: float, n_terms: int = 20) -> np.ndarray:
    """Analytical solution of 1D heat equation u_t = kappa * u_xx.

    IC: u(x,0) = sin(pi*x), BC: u(0,t) = u(1,t) = 0.
    """
    u = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        bn = 2.0 * (1 - (-1) ** n) / (n * np.pi) if n % 2 == 1 else 0.0
        if n == 1:
            bn = 1.0
        u += bn * np.sin(n * np.pi * x) * np.exp(-kappa * (n * np.pi) ** 2 * t)
    return u


def pinn_residual(
    x: np.ndarray,
    t: float,
    kappa_hat: float,
    u_pred: np.ndarray,
    dx: float = 0.01,
    dt: float = 0.001,
) -> np.ndarray:
    """Approximate PDE residual: |u_t - kappa * u_xx|."""
    u_plus = heat_equation_solution(x, t + dt, kappa_hat)
    u_minus = heat_equation_solution(x, t - dt, kappa_hat)
    u_t = (u_plus - u_minus) / (2 * dt)

    u_xp = heat_equation_solution(x + dx, t, kappa_hat)
    u_xm = heat_equation_solution(x - dx, t, kappa_hat)
    u_xx = (u_xp - 2 * u_pred + u_xm) / dx**2

    return np.abs(u_t - kappa_hat * u_xx)


def eig_from_observations(
    x_design: np.ndarray,
    t_obs: float,
    kappa_samples: np.ndarray,
    kappa_true: float,
    obs_noise: float = 0.02,
) -> float:
    """Approximate EIG for a set of sensor locations."""
    y_true = heat_equation_solution(x_design, t_obs, kappa_true)
    n_mc = len(kappa_samples)
    log_likes = np.zeros(n_mc)
    for i, k in enumerate(kappa_samples):
        y_pred = heat_equation_solution(x_design, t_obs, k)
        log_likes[i] = -0.5 * np.sum((y_true - y_pred) ** 2) / obs_noise**2
    log_likes -= np.max(log_likes)
    weights = np.exp(log_likes)
    weights /= weights.sum()
    entropy_post = -np.sum(weights * np.log(weights + 1e-15))
    entropy_prior = np.log(n_mc)
    return entropy_prior - entropy_post


def optimize_design_gradient_free(
    n_sensors: int,
    t_obs: float,
    kappa_samples: np.ndarray,
    kappa_true: float,
    n_candidates: int = 100,
    n_top: int = 20,
    n_refine: int = 5,
    obs_noise: float = 0.02,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Optimize sensor placement via physics-informed criterion."""
    rng = np.random.default_rng(seed)

    best_design = None
    best_score = -np.inf

    for _ in range(n_refine):
        candidates = rng.uniform(0.05, 0.95, (n_candidates, n_sensors))
        scores = np.zeros(n_candidates)
        for i, design in enumerate(candidates):
            design_sorted = np.sort(design)
            eig = eig_from_observations(design_sorted, t_obs, kappa_samples, kappa_true, obs_noise)
            residual = np.mean([
                np.mean(pinn_residual(
                    design_sorted, t_obs, k,
                    heat_equation_solution(design_sorted, t_obs, k)
                ))
                for k in kappa_samples[:10]
            ])
            scores[i] = eig - 0.1 * residual

        top_idx = np.argsort(-scores)[:n_top]
        if scores[top_idx[0]] > best_score:
            best_score = scores[top_idx[0]]
            best_design = np.sort(candidates[top_idx[0]])

        center = candidates[top_idx].mean(axis=0)
        spread = candidates[top_idx].std(axis=0) + 1e-3
        candidates_new = rng.normal(center, spread, (n_candidates, n_sensors))
        candidates = np.clip(candidates_new, 0.05, 0.95)

    return best_design, best_score


def run_pied_demo(
    n_sensors: int = 4,
    n_steps: int = 3,
    kappa_true: float = 0.1,
    obs_noise: float = 0.02,
    seed: int = 42,
) -> dict:
    """Sequential physics-informed experimental design for heat equation."""
    rng = np.random.default_rng(seed)
    kappa_prior = rng.uniform(0.01, 0.5, 100)
    t_obs = 0.1

    all_designs = []
    all_obs = []
    kappa_samples = kappa_prior.copy()

    for step in range(n_steps):
        design, score = optimize_design_gradient_free(
            n_sensors, t_obs * (step + 1),
            kappa_samples, kappa_true,
            n_candidates=80, n_refine=3,
            obs_noise=obs_noise, seed=seed + step,
        )
        all_designs.append(design)

        y_obs = heat_equation_solution(design, t_obs * (step + 1), kappa_true)
        y_obs += rng.normal(0, obs_noise, len(design))
        all_obs.append(y_obs)

        log_likes = np.zeros(len(kappa_samples))
        for i, k in enumerate(kappa_samples):
            y_pred = heat_equation_solution(design, t_obs * (step + 1), k)
            log_likes[i] = -0.5 * np.sum((y_obs - y_pred) ** 2) / obs_noise**2
        log_likes -= np.max(log_likes)
        weights = np.exp(log_likes)
        weights /= weights.sum()
        idx = rng.choice(len(kappa_samples), size=len(kappa_samples), p=weights)
        kappa_samples = kappa_prior[idx % len(kappa_prior)]
        kappa_samples += rng.normal(0, 0.01, len(kappa_samples))
        kappa_samples = np.clip(kappa_samples, 0.01, 0.5)

    kappa_est = np.mean(kappa_samples)
    return {
        "kappa_true": kappa_true,
        "kappa_estimate": kappa_est,
        "designs": all_designs,
        "n_steps": n_steps,
    }


if __name__ == "__main__":
    result = run_pied_demo(n_sensors=4, n_steps=3, seed=0)
    print(f"True kappa: {result['kappa_true']:.3f}")
    print(f"Estimated kappa: {result['kappa_estimate']:.3f}")
    for i, d in enumerate(result["designs"]):
        print(f"Step {i+1} sensors: {np.round(d, 3)}")
