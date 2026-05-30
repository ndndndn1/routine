"""
Reference implementation: MDN for Multimodal Scientific Inverse Problems
(arXiv:2602.00960)

Core idea: Mixture Density Networks provide superior data efficiency and
interpretability over flow/diffusion models for multimodal conditional
distributions in scientific inverse problems, through explicit parametric
density estimation with direct global probability mass allocation.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Multimodal physics toy problems
# ---------------------------------------------------------------------------

def bifurcation_forward(x, alpha=1.0):
    """
    Pitchfork bifurcation: y = alpha*x - x^3.
    Inverse is one-to-many for y in (-2alpha/3sqrt(3), 2alpha/3sqrt(3)).
    """
    return alpha * x - x**3


def multistable_forward(x):
    """
    Double-well potential: y = x^4 - 2*x^2.
    Multiple x values map to same y.
    """
    return x**4 - 2 * x**2


# ---------------------------------------------------------------------------
# Mixture Density Network (explicit parametric density estimator)
# ---------------------------------------------------------------------------

class ExplicitMDN:
    """
    MDN as explicit parametric density estimator.
    Outputs mixture weights, means, and variances for each component.
    """

    def __init__(self, d_in=1, d_out=1, K=5, hidden=64, lr=1e-3):
        self.K = K
        self.d_out = d_out
        self.lr = lr

        s = np.sqrt(2.0 / d_in)
        self.W1 = np.random.randn(d_in, hidden) * s
        self.b1 = np.zeros(hidden)

        # second hidden layer
        self.W1b = np.random.randn(hidden, hidden) * np.sqrt(2.0 / hidden)
        self.b1b = np.zeros(hidden)

        n_out = K + K * d_out + K * d_out
        self.W2 = np.random.randn(hidden, n_out) * 0.05
        self.b2 = np.zeros(n_out)

    def _forward(self, y_cond):
        """y_cond: (B, d_in) -> pi, mu, sigma."""
        h = np.tanh(y_cond @ self.W1 + self.b1)
        h = np.tanh(h @ self.W1b + self.b1b)
        out = h @ self.W2 + self.b2

        K, d = self.K, self.d_out
        pi_logits = out[:, :K]
        mu = out[:, K:K+K*d].reshape(-1, K, d)
        log_sigma = out[:, K+K*d:].reshape(-1, K, d)

        pi_logits -= pi_logits.max(axis=1, keepdims=True)
        pi = np.exp(pi_logits)
        pi /= pi.sum(axis=1, keepdims=True) + 1e-8

        sigma = np.exp(np.clip(log_sigma, -4, 2))
        return h, pi, mu, sigma

    def nll(self, y_cond, x_target):
        """Negative log-likelihood."""
        _, pi, mu, sigma = self._forward(y_cond)
        x_exp = x_target[:, np.newaxis, :]

        exponent = -0.5 * np.sum(((x_exp - mu) / sigma)**2, axis=2)
        log_norm = np.sum(np.log(sigma), axis=2) + 0.5 * self.d_out * np.log(2 * np.pi)
        log_comp = exponent - log_norm

        log_mix = log_comp + np.log(pi + 1e-10)
        log_max = np.max(log_mix, axis=1, keepdims=True)
        log_p = log_max.ravel() + np.log(np.sum(np.exp(log_mix - log_max), axis=1))
        return -np.mean(log_p)

    def fit(self, y_cond, x_target, epochs=2000, batch_size=128):
        N = len(y_cond)
        eps = 5e-5

        for ep in range(epochs):
            idx = np.random.choice(N, min(batch_size, N))
            yb = y_cond[idx]
            xb = x_target[idx]

            # parameter updates via numerical gradient
            for pname in ['W1', 'b1', 'W1b', 'b1b', 'W2', 'b2']:
                param = getattr(self, pname)
                n_sample = min(30, param.size)
                indices = np.random.choice(param.size, n_sample, replace=False)
                grad = np.zeros_like(param)
                for fi in indices:
                    mi = np.unravel_index(fi, param.shape)
                    old = param[mi]
                    param[mi] = old + eps
                    lp = self.nll(yb, xb)
                    param[mi] = old - eps
                    lm = self.nll(yb, xb)
                    param[mi] = old
                    grad[mi] = (lp - lm) / (2 * eps)
                setattr(self, pname, param - self.lr * grad)

            if (ep + 1) % 500 == 0:
                loss = self.nll(y_cond[:200], x_target[:200])
                print(f"  epoch {ep+1}  NLL={loss:.4f}")

    def predict_modes(self, y_cond):
        """Return mixture components for a given condition."""
        if y_cond.ndim == 1:
            y_cond = y_cond.reshape(1, -1)
        _, pi, mu, sigma = self._forward(y_cond)
        return pi[0], mu[0], sigma[0]

    def sample(self, y_cond, n_samples=100):
        """Sample from p(x | y)."""
        pi, mu, sigma = self.predict_modes(y_cond)
        samples = []
        for _ in range(n_samples):
            k = np.random.choice(self.K, p=pi)
            s = mu[k] + sigma[k] * np.random.randn(self.d_out)
            samples.append(s)
        return np.array(samples)


# ---------------------------------------------------------------------------
# Comparison: implicit model baseline (simple normalizing flow)
# ---------------------------------------------------------------------------

class SimpleFlow:
    """
    Minimal conditional normalizing flow (affine coupling) for comparison.
    """

    def __init__(self, d_cond=1, d_out=1, hidden=64, lr=1e-3):
        self.d_out = d_out
        self.lr = lr
        d_in = d_cond + d_out
        s = np.sqrt(2.0 / d_in)
        self.W1 = np.random.randn(d_in, hidden) * s
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, 2 * d_out) * 0.1
        self.b2 = np.zeros(2 * d_out)

    def _params(self, y_cond, z):
        inp = np.concatenate([y_cond, z], axis=1)
        h = np.tanh(inp @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        s = out[:, :self.d_out]
        t = out[:, self.d_out:]
        return s, t

    def sample(self, y_cond, n_samples=100):
        if y_cond.ndim == 1:
            y_cond = y_cond.reshape(1, -1)
        y_rep = np.tile(y_cond, (n_samples, 1))
        z = np.random.randn(n_samples, self.d_out)
        s, t = self._params(y_rep, z)
        return z * np.exp(s) + t


# ---------------------------------------------------------------------------
# Demo: bifurcation inverse problem
# ---------------------------------------------------------------------------

def demo():
    np.random.seed(42)

    # generate data from bifurcation system
    N = 2000
    x_data = np.random.uniform(-2, 2, (N, 1))
    y_data = bifurcation_forward(x_data)

    print("=== MDN vs Flow on bifurcation inverse problem ===")
    print(f"Data: {N} samples from y = x - x^3\n")

    # train MDN
    mdn = ExplicitMDN(d_in=1, d_out=1, K=5, hidden=48, lr=3e-3)
    print("Training MDN...")
    mdn.fit(y_data, x_data, epochs=2000, batch_size=128)

    # evaluate on a test target with 3 inverse solutions
    y_test = np.array([[0.0]])  # x = {-1, 0, 1} for alpha=1
    pi, mu, sigma = mdn.predict_modes(y_test)
    print(f"\nTarget y = 0.0 (true solutions: x = -1, 0, +1)")
    print(f"MDN modes:")
    for k in range(len(pi)):
        if pi[k] > 0.02:
            print(f"  Mode {k}: mu={mu[k,0]:.3f}, sigma={sigma[k,0]:.3f}, "
                  f"weight={pi[k]:.3f}")

    # sample and check mode coverage
    mdn_samples = mdn.sample(y_test, 500)
    print(f"\nMDN sample stats: mean={mdn_samples.mean():.3f}, "
          f"std={mdn_samples.std():.3f}")
    # check if all 3 modes are covered
    n_neg = np.sum(mdn_samples < -0.5)
    n_zero = np.sum(np.abs(mdn_samples) < 0.5)
    n_pos = np.sum(mdn_samples > 0.5)
    print(f"  Samples near x=-1: {n_neg}, near x=0: {n_zero}, near x=+1: {n_pos}")

    # data efficiency test
    print("\n--- Data efficiency comparison ---")
    for n_train in [100, 500, 2000]:
        idx = np.random.choice(N, n_train)
        mdn_small = ExplicitMDN(d_in=1, d_out=1, K=5, hidden=48, lr=3e-3)
        mdn_small.fit(y_data[idx], x_data[idx], epochs=1500, batch_size=64)
        nll = mdn_small.nll(y_data[:200], x_data[:200])
        print(f"  N={n_train:5d}  NLL={nll:.4f}")


if __name__ == "__main__":
    demo()
