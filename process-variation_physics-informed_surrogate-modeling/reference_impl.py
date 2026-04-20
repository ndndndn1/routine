"""
Reference implementation: Physics-Informed Mixture Model for Additive Manufacturing
(arXiv:2510.26586)

Core idea: A Gaussian Mixture Model augmented with physics-based constraints
from laser-metal thermodynamics to identify defect regimes (Lack of Fusion,
Keyhole, Balling) in AM processes. A neural surrogate predicts melt-pool
characteristics for real-time inference.
"""

import numpy as np
from scipy.stats import multivariate_normal


# ---------------------------------------------------------------------------
# Physics constraints for AM process regimes
# ---------------------------------------------------------------------------

def energy_density(power, speed, hatch, layer_t):
    """Volumetric energy density [J/mm^3]."""
    return power / (speed * hatch * layer_t + 1e-12)


def regime_prior(params):
    """
    Assign soft prior probabilities to regimes based on physics.
    params: (N, 3) array of [power (W), speed (mm/s), layer_thickness (um)]
    Returns: (N, 3) array of [p_LoF, p_OK, p_Keyhole]
    """
    ed = energy_density(params[:, 0], params[:, 1], 0.1, params[:, 2] * 1e-3)
    p = np.zeros((len(params), 3))
    # Lack of Fusion: low energy density
    p[:, 0] = np.exp(-0.01 * ed)
    # Keyhole: high energy density
    p[:, 2] = 1 - np.exp(-0.005 * (ed - 80))
    p[:, 2] = np.clip(p[:, 2], 0, 1)
    # OK: moderate
    p[:, 1] = 1 - p[:, 0] - p[:, 2]
    p[:, 1] = np.clip(p[:, 1], 0.01, 1)
    p /= p.sum(axis=1, keepdims=True)
    return p


# ---------------------------------------------------------------------------
# Physics-Informed GMM
# ---------------------------------------------------------------------------

class PhysicsInformedGMM:
    def __init__(self, n_components=3, max_iter=50, alpha=0.5):
        self.K = n_components
        self.max_iter = max_iter
        self.alpha = alpha  # weight of physics prior vs data
        self.means = None
        self.covs = None
        self.weights = None

    def fit(self, X, params=None):
        """
        X: (N, d) feature matrix (e.g., melt-pool width, depth, temp)
        params: (N, 3) process parameters for physics prior
        """
        N, d = X.shape

        # initialize
        idx = np.random.choice(N, self.K, replace=False)
        self.means = X[idx].copy()
        self.covs = [np.eye(d) * np.var(X, axis=0) for _ in range(self.K)]
        self.weights = np.ones(self.K) / self.K

        # physics prior
        if params is not None:
            prior = regime_prior(params)
        else:
            prior = np.ones((N, self.K)) / self.K

        for it in range(self.max_iter):
            # E-step
            resp = np.zeros((N, self.K))
            for k in range(self.K):
                try:
                    resp[:, k] = self.weights[k] * multivariate_normal.pdf(
                        X, mean=self.means[k], cov=self.covs[k] + 1e-4 * np.eye(d)
                    )
                except np.linalg.LinAlgError:
                    resp[:, k] = 1e-10

            # blend with physics prior
            resp = (1 - self.alpha) * resp + self.alpha * prior * resp.sum(axis=1, keepdims=True)
            resp /= resp.sum(axis=1, keepdims=True) + 1e-12

            # M-step
            Nk = resp.sum(axis=0)
            self.weights = Nk / N
            for k in range(self.K):
                self.means[k] = (resp[:, k:k+1].T @ X) / (Nk[k] + 1e-12)
                diff = X - self.means[k]
                self.covs[k] = (diff * resp[:, k:k+1]).T @ diff / (Nk[k] + 1e-12)
                self.covs[k] += 1e-4 * np.eye(d)

        return self

    def predict(self, X):
        N, d = X.shape
        resp = np.zeros((N, self.K))
        for k in range(self.K):
            resp[:, k] = self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covs[k]
            )
        return np.argmax(resp, axis=1)


# ---------------------------------------------------------------------------
# Neural surrogate (simple numpy MLP for melt-pool prediction)
# ---------------------------------------------------------------------------

class SimpleMLP:
    """Minimal 2-layer MLP surrogate (numpy only)."""

    def __init__(self, d_in, d_out, hidden=32):
        scale = np.sqrt(2.0 / d_in)
        self.W1 = np.random.randn(d_in, hidden) * scale
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, d_out) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(d_out)

    def forward(self, X):
        h = np.maximum(X @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2

    def fit(self, X, Y, epochs=500, lr=1e-3):
        for ep in range(epochs):
            h = np.maximum(X @ self.W1 + self.b1, 0)
            pred = h @ self.W2 + self.b2
            err = pred - Y
            loss = 0.5 * np.mean(err ** 2)

            # backprop
            grad_out = err / len(X)
            grad_W2 = h.T @ grad_out
            grad_b2 = grad_out.sum(axis=0)
            grad_h = grad_out @ self.W2.T
            grad_h[h <= 0] = 0
            grad_W1 = X.T @ grad_h
            grad_b1 = grad_h.sum(axis=0)

            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1

            if (ep + 1) % 200 == 0:
                print(f"  surrogate epoch {ep+1}  loss={loss:.5f}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    np.random.seed(42)
    N = 300

    # process parameters: [power (W), speed (mm/s), layer_thickness (um)]
    power = np.random.uniform(100, 400, N)
    speed = np.random.uniform(200, 1500, N)
    layer_t = np.random.uniform(20, 80, N)
    params = np.column_stack([power, speed, layer_t])

    # synthetic melt-pool features from physics + noise
    ed = energy_density(power, speed, 0.1, layer_t * 1e-3)
    width = 0.05 * ed + 20 + np.random.randn(N) * 5
    depth = 0.03 * ed + 10 + np.random.randn(N) * 3
    temp = 2 * ed + 1500 + np.random.randn(N) * 50
    features = np.column_stack([width, depth, temp])

    # normalize features
    mu_f, std_f = features.mean(0), features.std(0) + 1e-8
    features_n = (features - mu_f) / std_f

    # train surrogate: params -> features
    mu_p, std_p = params.mean(0), params.std(0) + 1e-8
    params_n = (params - mu_p) / std_p
    surrogate = SimpleMLP(3, 3, hidden=32)
    print("Training melt-pool surrogate...")
    surrogate.fit(params_n, features_n, epochs=500, lr=5e-3)

    # physics-informed GMM
    print("Fitting Physics-Informed GMM...")
    gmm = PhysicsInformedGMM(n_components=3, max_iter=30, alpha=0.3)
    gmm.fit(features_n, params)
    labels = gmm.predict(features_n)

    regime_names = ["Lack of Fusion", "Acceptable", "Keyhole"]
    for k in range(3):
        count = (labels == k).sum()
        print(f"  Regime {k} ({regime_names[k]}): {count} samples")

    # surrogate inference on new params
    new_params = np.array([[250, 800, 40]])
    new_params_n = (new_params - mu_p) / std_p
    pred_feat = surrogate.forward(new_params_n) * std_f + mu_f
    pred_label = gmm.predict((pred_feat - mu_f) / std_f)
    print(f"\nNew params (P=250W, v=800mm/s, t=40um):")
    print(f"  Predicted melt-pool: width={pred_feat[0,0]:.1f} depth={pred_feat[0,1]:.1f} temp={pred_feat[0,2]:.0f}")
    print(f"  Predicted regime: {regime_names[pred_label[0]]}")


if __name__ == "__main__":
    demo()
