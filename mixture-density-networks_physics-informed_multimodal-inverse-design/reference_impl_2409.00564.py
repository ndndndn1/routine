"""
Reference implementation: MDN for Stellarator Inverse Design
(arXiv:2409.00564)

Core idea: A Mixture Density Network maps target confinement properties to
stellarator design parameters, handling the ill-posed one-to-many inverse
mapping via explicit multimodal probability distributions.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Simplified stellarator forward model (near-axis)
# ---------------------------------------------------------------------------

def stellarator_forward(params):
    """
    Simplified near-axis stellarator physics.
    params: (K,) design parameters [elongation, rotational_transform, ...]
    Returns: (2,) confinement properties [iota, elongation_eff]
    """
    eta = params[0]
    iota_bar = params[1]
    B20 = params[2] if len(params) > 2 else 0.0

    iota_eff = iota_bar + 0.1 * eta**2 + 0.05 * B20
    elong_eff = 1.0 + 0.5 * eta + 0.02 * iota_bar**2

    return np.array([iota_eff, elong_eff])


def generate_data(N=2000, d_param=3, noise=0.01):
    """Generate (design -> properties) dataset."""
    params = np.random.randn(N, d_param) * 0.8
    props = np.array([stellarator_forward(p) for p in params])
    props += np.random.randn(N, 2) * noise
    return params, props


# ---------------------------------------------------------------------------
# Mixture Density Network
# ---------------------------------------------------------------------------

class MDN:
    """
    MDN: maps target properties -> mixture of Gaussians over design params.
    p(design | target) = sum_k pi_k * N(design; mu_k, sigma_k^2 I)
    """

    def __init__(self, d_input, d_output, n_components=5, hidden=64, lr=1e-3):
        self.K = n_components
        self.d_out = d_output
        self.lr = lr

        d_in = d_input
        s = np.sqrt(2.0 / d_in)
        self.W1 = np.random.randn(d_in, hidden) * s
        self.b1 = np.zeros(hidden)

        # output heads: pi (K), mu (K*d_out), log_sigma (K*d_out)
        n_out = n_components + n_components * d_output + n_components * d_output
        self.W2 = np.random.randn(hidden, n_out) * np.sqrt(2.0 / hidden) * 0.1
        self.b2 = np.zeros(n_out)

    def _forward(self, x):
        """x: (B, d_input) -> pi, mu, sigma."""
        h = np.tanh(x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2

        K, d = self.K, self.d_out
        pi_logits = out[:, :K]
        mu = out[:, K:K + K*d].reshape(-1, K, d)
        log_sigma = out[:, K + K*d:].reshape(-1, K, d)

        # softmax for pi
        pi_logits -= pi_logits.max(axis=1, keepdims=True)
        pi = np.exp(pi_logits)
        pi /= pi.sum(axis=1, keepdims=True)

        sigma = np.exp(np.clip(log_sigma, -5, 2))

        return h, pi, mu, sigma

    def nll(self, x, y):
        """Negative log-likelihood. x: (B, d_in), y: (B, d_out)."""
        _, pi, mu, sigma = self._forward(x)
        B = x.shape[0]

        # p(y | x) = sum_k pi_k * N(y; mu_k, sigma_k^2)
        y_exp = y[:, np.newaxis, :]  # (B, 1, d)
        exponent = -0.5 * np.sum(((y_exp - mu) / sigma)**2, axis=2)
        norm = np.sum(np.log(sigma + 1e-8), axis=2) + 0.5 * self.d_out * np.log(2 * np.pi)
        log_comp = exponent - norm  # (B, K)

        log_mix = log_comp + np.log(pi + 1e-8)
        log_p = np.max(log_mix, axis=1) + np.log(
            np.sum(np.exp(log_mix - np.max(log_mix, axis=1, keepdims=True)), axis=1)
        )
        return -np.mean(log_p)

    def fit(self, x, y, epochs=1000, batch_size=128):
        N = len(x)
        for ep in range(epochs):
            idx = np.random.choice(N, min(batch_size, N))
            xb, yb = x[idx], y[idx]

            loss = self.nll(xb, yb)

            # numerical gradient (for simplicity in numpy-only impl)
            eps = 1e-4
            for param_name in ['W1', 'b1', 'W2', 'b2']:
                param = getattr(self, param_name)
                grad = np.zeros_like(param)
                it = np.nditer(param, flags=['multi_index'])
                # sample subset for gradient estimation
                n_sample = min(20, param.size)
                indices = np.random.choice(param.size, n_sample, replace=False)
                for flat_idx in indices:
                    mi = np.unravel_index(flat_idx, param.shape)
                    old = param[mi]
                    param[mi] = old + eps
                    l_plus = self.nll(xb, yb)
                    param[mi] = old - eps
                    l_minus = self.nll(xb, yb)
                    param[mi] = old
                    grad[mi] = (l_plus - l_minus) / (2 * eps)
                setattr(self, param_name, param - self.lr * grad)

            if (ep + 1) % 200 == 0:
                full_loss = self.nll(x[:200], y[:200])
                print(f"  epoch {ep+1}  NLL={full_loss:.4f}")

    def predict(self, x_target, n_samples=10):
        """
        Sample designs for given target properties.
        x_target: (1, d_in) or (d_in,)
        Returns: list of (K,) dicts with pi, mu, sigma per component.
        """
        if x_target.ndim == 1:
            x_target = x_target.reshape(1, -1)
        _, pi, mu, sigma = self._forward(x_target)
        pi, mu, sigma = pi[0], mu[0], sigma[0]

        # sample from mixture
        samples = []
        for _ in range(n_samples):
            k = np.random.choice(self.K, p=pi)
            sample = mu[k] + sigma[k] * np.random.randn(self.d_out)
            samples.append(sample)

        return np.array(samples), pi, mu, sigma


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    np.random.seed(42)

    # generate data: params -> properties
    params, props = generate_data(N=1500, d_param=3, noise=0.01)

    # train MDN: properties -> params (inverse)
    mdn = MDN(d_input=2, d_output=3, n_components=4, hidden=48, lr=5e-3)
    print("Training MDN for stellarator inverse design...")
    mdn.fit(props, params, epochs=1000, batch_size=64)

    # inverse design: given target properties, find design params
    target_props = np.array([0.5, 1.3])
    print(f"\nTarget confinement properties: iota={target_props[0]:.2f}, "
          f"elongation={target_props[1]:.2f}")

    samples, pi, mu, sigma = mdn.predict(target_props, n_samples=50)

    print(f"\nMixture weights: {np.round(pi, 3)}")
    for k in range(len(pi)):
        if pi[k] > 0.05:
            print(f"  Mode {k}: mu={np.round(mu[k], 3)}, sigma={np.round(sigma[k], 3)}, "
                  f"weight={pi[k]:.3f}")

    # verify round-trip
    print("\nRound-trip verification (top 5 samples):")
    for i in range(min(5, len(samples))):
        fwd = stellarator_forward(samples[i])
        err = np.linalg.norm(fwd - target_props)
        print(f"  Sample {i}: params={np.round(samples[i], 3)}, "
              f"fwd={np.round(fwd, 3)}, error={err:.4f}")


if __name__ == "__main__":
    demo()
