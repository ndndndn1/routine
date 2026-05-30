"""
Reference implementation: Diagonal Flow Matching for Inverse Design with Abstention
(arXiv:2603.15925)

Core idea: A flow-based generative model that pairs design coordinates with noise
and labels with zero (zero-anchoring), making the learning provably invariant to
coordinate permutations.  Two intrinsic uncertainty metrics enable abstention on
unreliable predictions.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Toy forward model: many-to-one (multimodal inverse)
# ---------------------------------------------------------------------------

def forward_model(x):
    """
    2D -> 1D forward model with multiple inverse branches.
    y = x0^2 + x1^2  (ring-shaped level sets -> infinite inverse solutions)
    """
    return x[..., 0]**2 + x[..., 1]**2


# ---------------------------------------------------------------------------
# Diagonal Flow Matching (simplified continuous normalising flow)
# ---------------------------------------------------------------------------

class DiagonalFlowMatching:
    """
    Minimal Diag-CFM: learns to map (noise, label=0) -> (design, label) along
    a diagonal path, enforcing permutation invariance in design coordinates.

    Uses a simple MLP velocity field v(z_t, t, y) where y is the target label.
    """

    def __init__(self, d_design, hidden=64, lr=1e-3):
        self.d = d_design
        self.lr = lr
        # velocity network: input = [z_design(d), z_label(1), t(1), y_cond(1)]
        d_in = d_design + 3
        d_out = d_design + 1  # velocity for design + label dims
        s = np.sqrt(2.0 / d_in)
        self.W1 = np.random.randn(d_in, hidden) * s
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, d_out) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(d_out)

    def _velocity(self, z, t, y_cond):
        """Compute velocity field. z: (B, d+1), t: (B,1), y_cond: (B,1)."""
        inp = np.concatenate([z, t, y_cond], axis=1)
        h = np.maximum(inp @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2

    def train_step(self, x_data, y_data):
        """
        One training step.
        x_data: (B, d) design params, y_data: (B,) target labels.
        """
        B = len(x_data)
        d = self.d

        # sample t ~ U(0,1)
        t = np.random.rand(B, 1)

        # zero-anchoring: x0_design = noise, x0_label = 0
        noise_design = np.random.randn(B, d)
        x0 = np.concatenate([noise_design, np.zeros((B, 1))], axis=1)

        # target: x1_design = x_data, x1_label = y_data
        x1 = np.concatenate([x_data, y_data.reshape(-1, 1)], axis=1)

        # diagonal interpolation: z_t = (1-t)*x0 + t*x1
        z_t = (1 - t) * x0 + t * x1

        # target velocity: v_target = x1 - x0
        v_target = x1 - x0

        # predicted velocity
        v_pred = self._velocity(z_t, t, y_data.reshape(-1, 1))

        # MSE loss + backprop
        err = v_pred - v_target
        loss = 0.5 * np.mean(err**2)

        # manual backprop through 2-layer MLP
        grad_out = err / B
        inp = np.concatenate([z_t, t, y_data.reshape(-1, 1)], axis=1)
        h = np.maximum(inp @ self.W1 + self.b1, 0)

        self.W2 -= self.lr * (h.T @ grad_out)
        self.b2 -= self.lr * grad_out.sum(axis=0)
        grad_h = grad_out @ self.W2.T
        grad_h[h <= 0] = 0
        self.W1 -= self.lr * (inp.T @ grad_h)
        self.b1 -= self.lr * grad_h.sum(axis=0)

        return loss

    def sample(self, y_target, n_samples=100, n_steps=50):
        """
        Generate design candidates for a given target label.
        Returns: (n_samples, d) design coordinates.
        """
        d = self.d
        # start from noise (zero-anchored)
        z = np.concatenate([
            np.random.randn(n_samples, d),
            np.zeros((n_samples, 1))
        ], axis=1)

        y_cond = np.full((n_samples, 1), y_target)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t_val = np.full((n_samples, 1), step * dt)
            v = self._velocity(z, t_val, y_cond)
            z = z + v * dt

        return z[:, :d]  # design coordinates only

    def zero_deviation(self, samples, y_target):
        """
        Uncertainty metric: deviation of the label channel from target.
        Lower = more confident.
        """
        y_pred = forward_model(samples)
        return np.std(y_pred - y_target)

    def self_consistency(self, y_target, n_runs=5, n_samples=50, n_steps=50):
        """
        Uncertainty metric: variance across multiple independent generation runs.
        Lower = more consistent.
        """
        means = []
        for _ in range(n_runs):
            samples = self.sample(y_target, n_samples, n_steps)
            means.append(samples.mean(axis=0))
        return np.mean(np.std(means, axis=0))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    np.random.seed(42)
    d = 2

    # generate training data: x -> y = x0^2 + x1^2
    N = 1000
    x_data = np.random.randn(N, d) * 1.5
    y_data = forward_model(x_data)

    # train Diag-CFM
    model = DiagonalFlowMatching(d_design=d, hidden=64, lr=5e-4)
    print("Training Diagonal Flow Matching...")
    batch_size = 128
    for epoch in range(300):
        idx = np.random.choice(N, batch_size)
        loss = model.train_step(x_data[idx], y_data[idx])
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1}  loss={loss:.5f}")

    # inverse design: find x such that ||x||^2 = target
    target = 2.0
    print(f"\nInverse design for y_target = {target}")
    samples = model.sample(target, n_samples=200, n_steps=80)

    # evaluate round-trip accuracy
    y_pred = forward_model(samples)
    rt_error = np.mean(np.abs(y_pred - target))
    print(f"  Mean round-trip error: {rt_error:.4f}")
    print(f"  Sample radii: mean={np.sqrt(np.sum(samples**2, axis=1)).mean():.3f} "
          f"(expected: {np.sqrt(target):.3f})")

    # uncertainty / abstention
    zd = model.zero_deviation(samples, target)
    sc = model.self_consistency(target)
    print(f"  Zero-Deviation: {zd:.4f}")
    print(f"  Self-Consistency: {sc:.4f}")

    # OOD target (should have higher uncertainty)
    ood_target = 20.0
    ood_samples = model.sample(ood_target, n_samples=200, n_steps=80)
    zd_ood = model.zero_deviation(ood_samples, ood_target)
    sc_ood = model.self_consistency(ood_target)
    print(f"\nOOD target = {ood_target}")
    print(f"  Zero-Deviation: {zd_ood:.4f} (expected higher)")
    print(f"  Self-Consistency: {sc_ood:.4f} (expected higher)")

    if zd_ood > zd:
        print("  -> Abstention triggered for OOD target (correct)")


if __name__ == "__main__":
    demo()
