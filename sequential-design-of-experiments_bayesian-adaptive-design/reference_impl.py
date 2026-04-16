"""Minimal reference implementation: Gradient-Free Sequential BOED via
Interacting Particle Systems (arXiv:2504.13320).

Demonstrates EKI (Ensemble Kalman Inversion) for design optimization
and ensemble-based posterior sampling in a linear Gaussian setting where
the analytic EIG is available for validation.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Linear Gaussian forward model ---------------------------------------------
# ---------------------------------------------------------------------------
class LinearGaussianModel:
    """y = A(ξ) θ + ε,  ε ~ N(0, Σ_noise).
    A(ξ) is a design-dependent observation matrix."""

    def __init__(self, dim_theta: int = 2, noise_std: float = 0.1):
        self.d = dim_theta
        self.Sigma_noise = noise_std ** 2 * np.eye(1)
        self.Sigma_prior = np.eye(dim_theta)
        self.mu_prior = np.zeros(dim_theta)

    def design_matrix(self, xi: np.ndarray) -> np.ndarray:
        return xi.reshape(1, -1)

    def observe(self, theta: np.ndarray, xi: np.ndarray) -> np.ndarray:
        A = self.design_matrix(xi)
        return A @ theta + np.random.randn(1) * np.sqrt(self.Sigma_noise[0, 0])

    def analytic_eig(self, xi: np.ndarray) -> float:
        A = self.design_matrix(xi)
        S = A @ self.Sigma_prior @ A.T + self.Sigma_noise
        return 0.5 * np.log(np.linalg.det(S) / np.linalg.det(self.Sigma_noise))

    def posterior(self, xi_list, y_list):
        Sigma_inv = np.linalg.inv(self.Sigma_prior)
        mu = Sigma_inv @ self.mu_prior
        noise_inv = np.linalg.inv(self.Sigma_noise)
        for xi, y in zip(xi_list, y_list):
            A = self.design_matrix(xi)
            Sigma_inv = Sigma_inv + A.T @ noise_inv @ A
            mu = mu + A.T @ noise_inv @ y
        Sigma_post = np.linalg.inv(Sigma_inv)
        mu_post = Sigma_post @ mu
        return mu_post.flatten(), Sigma_post


# ---------------------------------------------------------------------------
# ALDI-like ensemble posterior sampler (simplified) --------------------------
# ---------------------------------------------------------------------------
def ensemble_posterior_sample(model: LinearGaussianModel,
                              xi_list, y_list,
                              n_particles: int = 50,
                              n_steps: int = 100,
                              step_size: float = 0.05,
                              rng=None):
    """Simplified affine-invariant ensemble sampler for Gaussian posterior."""
    rng = rng or np.random.default_rng()
    d = model.d
    particles = rng.multivariate_normal(model.mu_prior, model.Sigma_prior,
                                         size=n_particles)

    Sigma_prior_inv = np.linalg.inv(model.Sigma_prior)
    noise_inv = np.linalg.inv(model.Sigma_noise)

    for _ in range(n_steps):
        C = np.cov(particles.T) + 1e-6 * np.eye(d)
        mean_p = particles.mean(axis=0)
        for j in range(n_particles):
            grad = -Sigma_prior_inv @ (particles[j] - model.mu_prior)
            for xi, y in zip(xi_list, y_list):
                A = model.design_matrix(xi)
                resid = y - A @ particles[j]
                grad += A.T @ noise_inv @ resid
            interaction = -np.linalg.solve(C, particles[j] - mean_p) / n_particles
            particles[j] += step_size * (grad + interaction) + \
                np.sqrt(2 * step_size) * rng.multivariate_normal(np.zeros(d), C * 0.01)

    return particles


# ---------------------------------------------------------------------------
# EKI for design optimization ------------------------------------------------
# ---------------------------------------------------------------------------
def eki_optimize_design(model: LinearGaussianModel,
                        xi_list, y_list,
                        n_ensemble: int = 30,
                        n_iter: int = 40,
                        bounds=(-2, 2),
                        rng=None):
    """Use Ensemble Kalman Inversion to find design ξ maximizing EIG."""
    rng = rng or np.random.default_rng()
    d = model.d

    xi_ensemble = rng.uniform(bounds[0], bounds[1], (n_ensemble, d))

    mu_post, Sigma_post = model.posterior(xi_list, y_list)

    for it in range(n_iter):
        eigs = np.array([_approx_eig(model, xi, Sigma_post)
                         for xi in xi_ensemble])

        target = eigs.max() * np.ones(n_ensemble)
        C_xi_g = np.cov(xi_ensemble.T, eigs)[:-1, -1:]
        C_gg = np.var(eigs) + 1e-6

        xi_ensemble = xi_ensemble + (C_xi_g @ ((target - eigs) / C_gg).reshape(1, -1)).T
        xi_ensemble = np.clip(xi_ensemble, bounds[0], bounds[1])

    eigs_final = np.array([_approx_eig(model, xi, Sigma_post)
                           for xi in xi_ensemble])
    return xi_ensemble[np.argmax(eigs_final)]


def _approx_eig(model, xi, Sigma_post):
    A = model.design_matrix(xi)
    S = A @ Sigma_post @ A.T + model.Sigma_noise
    return 0.5 * np.log(np.linalg.det(S) / np.linalg.det(model.Sigma_noise))


# ---------------------------------------------------------------------------
# Sequential BOED loop -------------------------------------------------------
# ---------------------------------------------------------------------------
def sequential_boed(
    model: LinearGaussianModel,
    theta_true: np.ndarray,
    n_rounds: int = 8,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    xi_list, y_list = [], []

    for t in range(n_rounds):
        xi_next = eki_optimize_design(model, xi_list, y_list, rng=rng)
        y_next = model.observe(theta_true, xi_next)

        xi_list.append(xi_next)
        y_list.append(y_next)

        mu_post, Sigma_post = model.posterior(xi_list, y_list)
        post_var = np.trace(Sigma_post)
        eig = model.analytic_eig(xi_next)

        print(f"  Round {t+1}: ξ={xi_next.round(3)}, "
              f"EIG={eig:.3f}, post_var={post_var:.4f}, "
              f"θ_est={mu_post.round(3)}")

    return xi_list, y_list


# ---------------------------------------------------------------------------
# Demo -----------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = LinearGaussianModel(dim_theta=2, noise_std=0.2)
    theta_true = np.array([1.5, -0.8])

    print("Sequential BOED (EKI + ensemble posterior)")
    print(f"True θ = {theta_true}")
    print("-" * 55)

    xi_list, y_list = sequential_boed(model, theta_true, n_rounds=8)

    mu_final, Sigma_final = model.posterior(xi_list, y_list)
    print(f"\nFinal estimate: θ = {mu_final.round(3)}")
    print(f"Posterior std:  {np.sqrt(np.diag(Sigma_final)).round(3)}")
    print(f"Error:          {np.abs(mu_final - theta_true).round(3)}")
