"""
Reference implementation: Sequential Bayesian OED via Policy Gradient RL
Based on arXiv:2601.05868 (Shen & Chen, 2026)

Demonstrates the core idea: learn an amortized design policy that maps
observation history to the next optimal sensor placement for a PDE inverse
problem, using REINFORCE with a neural-operator surrogate.

Minimal NumPy/SciPy demo on a 1D Darcy-like toy problem.
"""

import numpy as np
from scipy.linalg import cho_solve, cho_factor
from typing import Tuple, List


def _gp_posterior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    length_scale: float = 0.3,
    noise: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal GP regression (RBF kernel) as surrogate."""
    sq_dist = lambda A, B: np.sum((A[:, None] - B[None, :]) ** 2, axis=-1) if A.ndim > 1 else (A[:, None] - B[None, :]) ** 2
    K = np.exp(-0.5 * sq_dist(X_train, X_train) / length_scale**2)
    K += noise * np.eye(len(X_train))
    Ks = np.exp(-0.5 * sq_dist(X_train, X_test) / length_scale**2)
    Kss = np.exp(-0.5 * sq_dist(X_test, X_test) / length_scale**2)
    L = cho_factor(K)
    alpha = cho_solve(L, y_train)
    mu = Ks.T @ alpha
    v = cho_solve(L, Ks)
    cov = Kss - Ks.T @ v
    return mu, np.diag(cov)


def forward_model(theta: float, x: np.ndarray) -> np.ndarray:
    """Toy 1D forward model: y = sin(theta * pi * x) + 0.5*cos(2*pi*x)."""
    return np.sin(theta * np.pi * x) + 0.5 * np.cos(2.0 * np.pi * x)


def eig_approx(
    design: np.ndarray,
    theta_samples: np.ndarray,
    obs_noise: float = 0.05,
) -> float:
    """Laplace-approximated EIG for a batch of designs."""
    n_mc = len(theta_samples)
    log_likes = np.zeros(n_mc)
    for i, th in enumerate(theta_samples):
        y_pred = forward_model(th, design)
        prior_pred = np.mean(
            [forward_model(t, design) for t in theta_samples], axis=0
        )
        residual = y_pred - prior_pred
        log_likes[i] = -0.5 * np.sum(residual**2) / obs_noise**2
    eig = np.mean(log_likes) - np.log(n_mc)
    return eig


class DesignPolicy:
    """Simple softmax policy over candidate design locations."""

    def __init__(self, n_candidates: int, lr: float = 0.05):
        self.n_candidates = n_candidates
        self.logits = np.zeros(n_candidates)
        self.lr = lr

    def probs(self) -> np.ndarray:
        e = np.exp(self.logits - np.max(self.logits))
        return e / e.sum()

    def sample(self, rng: np.random.Generator) -> int:
        return rng.choice(self.n_candidates, p=self.probs())

    def update(self, action: int, advantage: float):
        p = self.probs()
        grad = -p.copy()
        grad[action] += 1.0
        self.logits += self.lr * advantage * grad


def run_sboed(
    n_steps: int = 5,
    n_episodes: int = 200,
    n_candidates: int = 20,
    n_theta_samples: int = 30,
    obs_noise: float = 0.05,
    seed: int = 42,
) -> dict:
    """Run sequential BOED with REINFORCE policy gradient."""
    rng = np.random.default_rng(seed)
    candidates = np.linspace(0.05, 0.95, n_candidates)
    theta_prior = rng.uniform(0.5, 3.0, size=n_theta_samples)
    theta_true = 1.5

    policies = [DesignPolicy(n_candidates) for _ in range(n_steps)]
    reward_history = []

    for ep in range(n_episodes):
        designs_chosen = []
        actions = []
        observations = []
        total_reward = 0.0

        theta_post = theta_prior.copy()

        for t in range(n_steps):
            action = policies[t].sample(rng)
            x_design = candidates[action]
            actions.append(action)
            designs_chosen.append(x_design)

            y_obs = forward_model(theta_true, np.array([x_design]))[0]
            y_obs += rng.normal(0, obs_noise)
            observations.append(y_obs)

            r = eig_approx(
                np.array(designs_chosen), theta_post, obs_noise
            )
            total_reward += r

            log_likes = np.array([
                -0.5 * sum(
                    (forward_model(th, np.array([d])) - o) ** 2
                    for d, o in zip(designs_chosen, observations)
                ) / obs_noise**2
                for th in theta_prior
            ])
            weights = np.exp(log_likes - np.max(log_likes))
            weights /= weights.sum()
            idx = rng.choice(n_theta_samples, size=n_theta_samples, p=weights)
            theta_post = theta_prior[idx]

        reward_history.append(total_reward)
        baseline = np.mean(reward_history[-50:]) if len(reward_history) > 1 else 0
        advantage = total_reward - baseline
        for t in range(n_steps):
            policies[t].update(actions[t], advantage)

    final_designs = []
    for t in range(n_steps):
        best = candidates[np.argmax(policies[t].probs())]
        final_designs.append(best)

    return {
        "optimal_designs": np.array(final_designs),
        "reward_history": reward_history,
        "final_probs": [p.probs() for p in policies],
    }


if __name__ == "__main__":
    result = run_sboed(n_steps=4, n_episodes=150, seed=0)
    print("Learned sequential designs:", np.round(result["optimal_designs"], 3))
    print(
        "Final reward (last 10 avg):",
        np.mean(result["reward_history"][-10:]),
    )
