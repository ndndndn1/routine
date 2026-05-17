"""
Reference implementation: Multimodal Conditional Mixture Model with Physics Priors
Based on arXiv:2602.10451 (Han et al., 2026)

Core idea: MDN with component-wise physics regularization so each mixture
component satisfies governing equations. Demonstrates on a 1D bifurcation
toy problem where the inverse mapping is one-to-many.

Requires only NumPy (no deep learning framework needed for the demo).
"""

import numpy as np
from typing import Tuple


def forward_model(x: np.ndarray) -> np.ndarray:
    """Toy bifurcating forward model: y = x^3 - 3x (Duffing-like).

    For a given y, there can be 1 or 3 real roots x (multimodal inverse).
    """
    return x**3 - 3.0 * x


def physics_residual(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Residual of the governing equation: x^3 - 3x - y = 0."""
    return (x**3 - 3.0 * x - y) ** 2


class SimpleMDN:
    """Minimal Mixture Density Network with physics-informed regularization.

    Uses a single hidden layer and K Gaussian components.
    """

    def __init__(self, K: int = 3, hidden: int = 32, lr: float = 0.01, lam_phys: float = 0.1):
        self.K = K
        self.hidden = hidden
        self.lr = lr
        self.lam_phys = lam_phys
        rng = np.random.default_rng(0)
        scale = 0.3
        self.W1 = rng.normal(0, scale, (1, hidden))
        self.b1 = np.zeros(hidden)
        self.W_pi = rng.normal(0, scale, (hidden, K))
        self.b_pi = np.zeros(K)
        self.W_mu = rng.normal(0, scale, (hidden, K))
        self.b_mu = np.zeros(K)
        self.W_sigma = rng.normal(0, scale, (hidden, K))
        self.b_sigma = np.zeros(K)

    def _forward(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h = np.tanh(y.reshape(-1, 1) @ self.W1 + self.b1)
        logits = h @ self.W_pi + self.b_pi
        logits -= logits.max(axis=1, keepdims=True)
        pi = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        mu = h @ self.W_mu + self.b_mu
        log_sigma = h @ self.W_sigma + self.b_sigma
        sigma = np.exp(np.clip(log_sigma, -3, 2))
        return h, pi, mu, sigma

    def nll(self, y: np.ndarray, x_true: np.ndarray) -> float:
        _, pi, mu, sigma = self._forward(y)
        x_t = x_true.reshape(-1, 1)
        gauss = np.exp(-0.5 * ((x_t - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        mixture = np.sum(pi * gauss, axis=1)
        return -np.mean(np.log(mixture + 1e-10))

    def physics_loss(self, y: np.ndarray) -> float:
        _, pi, mu, _ = self._forward(y)
        residuals = physics_residual(mu, y.reshape(-1, 1))
        return np.mean(np.sum(pi * residuals, axis=1))

    def fit(self, y_data: np.ndarray, x_data: np.ndarray, epochs: int = 500, batch: int = 64):
        rng = np.random.default_rng(1)
        n = len(y_data)
        losses = []
        for ep in range(epochs):
            idx = rng.choice(n, min(batch, n), replace=False)
            yb, xb = y_data[idx], x_data[idx]
            loss = self.nll(yb, xb) + self.lam_phys * self.physics_loss(yb)
            losses.append(loss)
            self._numerical_grad_step(yb, xb)
        return losses

    def _numerical_grad_step(self, y: np.ndarray, x: np.ndarray, eps: float = 1e-4):
        params = [self.W1, self.b1, self.W_pi, self.b_pi,
                  self.W_mu, self.b_mu, self.W_sigma, self.b_sigma]
        for p in params:
            grad = np.zeros_like(p)
            it = np.nditer(p, flags=["multi_index"])
            while not it.finished:
                ix = it.multi_index
                old = p[ix]
                p[ix] = old + eps
                l1 = self.nll(y, x) + self.lam_phys * self.physics_loss(y)
                p[ix] = old - eps
                l2 = self.nll(y, x) + self.lam_phys * self.physics_loss(y)
                grad[ix] = (l1 - l2) / (2 * eps)
                p[ix] = old
                it.iternext()
            p -= self.lr * grad

    def predict(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, pi, mu, sigma = self._forward(y)
        return pi, mu, sigma


def generate_training_data(n: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.5, 2.5, n)
    y = forward_model(x) + rng.normal(0, 0.05, n)
    return y, x


if __name__ == "__main__":
    y_data, x_data = generate_training_data(300, seed=0)
    mdn = SimpleMDN(K=3, hidden=16, lr=0.005, lam_phys=0.5)
    losses = mdn.fit(y_data, x_data, epochs=100, batch=64)
    print(f"Final loss: {losses[-1]:.4f}")

    y_test = np.array([0.0, 2.0, -2.0])
    pi, mu, sigma = mdn.predict(y_test)
    for i, yt in enumerate(y_test):
        print(f"\ny = {yt:.1f} -> modes:")
        order = np.argsort(-pi[i])
        for k in order:
            res = np.sqrt(physics_residual(mu[i, k], yt))
            print(f"  x={mu[i,k]:+.3f}  π={pi[i,k]:.3f}  σ={sigma[i,k]:.3f}  |residual|={res:.4f}")
