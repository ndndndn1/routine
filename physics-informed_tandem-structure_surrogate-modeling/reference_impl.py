"""
Reference implementation: Physics-Informed Tandem DNN for Metasurface Design
(arXiv:2603.15430)

Core idea: A tandem architecture where an Inverse DNN maps target radiation
patterns to reactive load distributions, and a physics-based forward solver
validates/corrects the prediction via cosine-similarity loss.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physics-based forward solver (simplified microwave network model)
# ---------------------------------------------------------------------------

def microwave_forward(loads, n_angles=60):
    """
    Simplified metasurface radiation pattern from reactive loads.
    loads: (N_cells,) reactive load values (impedance)
    Returns: (n_angles,) normalised far-field radiation pattern
    """
    N = len(loads)
    angles = np.linspace(-np.pi/2, np.pi/2, n_angles)
    wavelength = 1.0
    d = 0.5 * wavelength  # element spacing

    # array factor with reactive load phase shifts
    pattern = np.zeros(n_angles)
    for i, theta in enumerate(angles):
        steering = np.exp(1j * 2 * np.pi / wavelength * d
                         * np.arange(N) * np.sin(theta))
        phase_shift = np.exp(1j * np.arctan(loads))
        af = np.abs(np.sum(phase_shift * steering))
        pattern[i] = af

    # normalise
    pattern /= (pattern.max() + 1e-8)
    return pattern


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# ---------------------------------------------------------------------------
# Tandem DNN: Inverse network + Physics forward solver
# ---------------------------------------------------------------------------

class InverseDNN:
    """MLP: target pattern -> reactive loads."""

    def __init__(self, n_pattern, n_cells, hidden=64, lr=1e-3):
        self.n_cells = n_cells
        self.lr = lr
        s = np.sqrt(2.0 / n_pattern)
        self.W1 = np.random.randn(n_pattern, hidden) * s
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, n_cells) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(n_cells)

    def forward(self, pattern):
        """pattern: (B, n_pattern) -> (B, n_cells) predicted loads."""
        h = np.maximum(pattern @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2


class TandemMetasurface:
    """
    Tandem architecture: InverseDNN predicts loads, physics solver validates.
    Training loss = 1 - cosine_similarity(target_pattern, forward(predicted_loads))
    """

    def __init__(self, n_pattern, n_cells, hidden=64, lr=1e-3):
        self.inverse = InverseDNN(n_pattern, n_cells, hidden, lr)
        self.n_pattern = n_pattern
        self.n_cells = n_cells
        self.lr = lr

    def train_step(self, target_patterns):
        """
        Train using physics-in-the-loop.
        target_patterns: (B, n_pattern)
        """
        B = len(target_patterns)
        pred_loads = self.inverse.forward(target_patterns)

        total_loss = 0
        grad_loads = np.zeros_like(pred_loads)

        for i in range(B):
            target = target_patterns[i]
            loads = pred_loads[i]

            # forward solver
            realized = microwave_forward(loads, self.n_pattern)

            # cosine similarity loss
            cs = cosine_similarity(target, realized)
            loss = 1.0 - cs
            total_loss += loss

            # numerical gradient of loss w.r.t. loads
            eps = 1e-3
            for j in range(self.n_cells):
                loads_p = loads.copy()
                loads_p[j] += eps
                real_p = microwave_forward(loads_p, self.n_pattern)
                cs_p = cosine_similarity(target, real_p)

                loads_m = loads.copy()
                loads_m[j] -= eps
                real_m = microwave_forward(loads_m, self.n_pattern)
                cs_m = cosine_similarity(target, real_m)

                grad_loads[i, j] = -(cs_p - cs_m) / (2 * eps)

        total_loss /= B

        # backprop through inverse DNN
        h = np.maximum(target_patterns @ self.inverse.W1 + self.inverse.b1, 0)
        grad_out = grad_loads / B

        self.inverse.W2 -= self.lr * (h.T @ grad_out)
        self.inverse.b2 -= self.lr * grad_out.sum(axis=0)
        grad_h = grad_out @ self.inverse.W2.T
        grad_h[h <= 0] = 0
        self.inverse.W1 -= self.lr * (target_patterns.T @ grad_h)
        self.inverse.b1 -= self.lr * grad_h.sum(axis=0)

        return total_loss

    def design(self, target_pattern):
        """Predict reactive loads for a target pattern."""
        loads = self.inverse.forward(target_pattern.reshape(1, -1))[0]
        realized = microwave_forward(loads, self.n_pattern)
        cs = cosine_similarity(target_pattern, realized)
        return loads, realized, cs


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    np.random.seed(42)

    n_cells = 8
    n_pattern = 60

    # generate training patterns from random load configurations
    print("Generating training data...")
    N_train = 200
    train_patterns = np.zeros((N_train, n_pattern))
    for i in range(N_train):
        random_loads = np.random.randn(n_cells) * 2.0
        train_patterns[i] = microwave_forward(random_loads, n_pattern)

    # train tandem model
    model = TandemMetasurface(n_pattern, n_cells, hidden=64, lr=1e-3)
    print("Training Tandem DNN with physics-in-the-loop...")
    batch_size = 16
    for epoch in range(150):
        idx = np.random.choice(N_train, batch_size)
        loss = model.train_step(train_patterns[idx])
        if (epoch + 1) % 30 == 0:
            print(f"  epoch {epoch+1}  cosine-loss={loss:.4f}")

    # test: design for a shaped beam pattern
    print("\nInverse design test:")
    angles = np.linspace(-np.pi/2, np.pi/2, n_pattern)

    # target: beam steered to 20 degrees
    steer_angle = 20 * np.pi / 180
    target = np.exp(-((angles - steer_angle) / 0.2)**2)
    target /= target.max()

    loads, realized, cs = model.design(target)
    print(f"  Cosine similarity: {cs:.4f}")
    print(f"  Predicted loads: {np.round(loads, 3)}")
    print(f"  Target peak at: {angles[np.argmax(target)] * 180 / np.pi:.1f} deg")
    print(f"  Realized peak at: {angles[np.argmax(realized)] * 180 / np.pi:.1f} deg")

    # target: broadside beam
    target_broad = np.exp(-(angles / 0.3)**2)
    target_broad /= target_broad.max()
    loads_b, realized_b, cs_b = model.design(target_broad)
    print(f"\n  Broadside beam:")
    print(f"  Cosine similarity: {cs_b:.4f}")
    print(f"  Realized peak at: {angles[np.argmax(realized_b)] * 180 / np.pi:.1f} deg")


if __name__ == "__main__":
    demo()
