"""
Reference implementation: Active Learning of Model Discrepancy with Bayesian
Experimental Design (arXiv:2502.05372)

Core idea: Sequentially select experiments that maximise information gain about
model discrepancy between a physics solver and reality. Uses an MLP correction
term (surrogate for Neural ODE discrepancy) plus Ensemble-Kalman-style
gradient-free Bayesian inference.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Toy physics model with known discrepancy
# ---------------------------------------------------------------------------

def physics_model(x, theta):
    """
    Simplified physics solver: y = theta[0]*sin(theta[1]*x).
    """
    return theta[0] * np.sin(theta[1] * x)


def true_system(x, theta):
    """
    Reality includes a discrepancy term: cubic correction.
    """
    return physics_model(x, theta) + 0.3 * np.sin(3 * x) + 0.1 * x**2


# ---------------------------------------------------------------------------
# Neural discrepancy model (simple MLP)
# ---------------------------------------------------------------------------

class DiscrepancyMLP:
    """MLP that learns the gap: delta(x) = reality(x) - physics(x)."""

    def __init__(self, hidden=32, lr=1e-3):
        self.W1 = np.random.randn(1, hidden) * 0.5
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, 1) * 0.3
        self.b2 = np.zeros(1)
        self.lr = lr

    def forward(self, x):
        """x: (N, 1) -> (N, 1)."""
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def fit(self, X, Y, epochs=500):
        for ep in range(epochs):
            h = np.tanh(X @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            err = pred - Y
            loss = 0.5 * np.mean(err**2)

            grad_out = err / len(X)
            self.W2 -= self.lr * (h.T @ grad_out)
            self.b2 -= self.lr * grad_out.sum(axis=0)
            grad_h = grad_out @ self.W2.T * (1 - h**2)
            self.W1 -= self.lr * (X.T @ grad_h)
            self.b1 -= self.lr * grad_h.sum(axis=0)
        return loss


# ---------------------------------------------------------------------------
# Ensemble Kalman Inversion (EKI) — gradient-free Bayesian inference
# ---------------------------------------------------------------------------

class EnsembleKalmanInversion:
    """
    Simplified EKI for inferring discrepancy-model parameters.
    Ensemble of parameter vectors; update via Kalman gain.
    """

    def __init__(self, n_ensemble, param_dim, prior_mean, prior_std):
        self.J = n_ensemble
        self.ensemble = prior_mean + prior_std * np.random.randn(n_ensemble, param_dim)
        self.prior_std = prior_std

    def predict(self, x, disc_model_fn):
        """Run each ensemble member's discrepancy model on x."""
        preds = []
        for j in range(self.J):
            preds.append(disc_model_fn(x, self.ensemble[j]))
        return np.array(preds)  # (J, N_obs)

    def update(self, G, y_obs, noise_std):
        """
        Kalman update.
        G: (J, N_obs) predicted observations per ensemble member
        y_obs: (N_obs,) actual observations
        """
        J = self.J
        G_mean = G.mean(axis=0)
        G_pert = G - G_mean

        # covariance in observation space
        C_yy = (G_pert.T @ G_pert) / (J - 1) + noise_std**2 * np.eye(len(y_obs))

        # cross covariance
        ens_mean = self.ensemble.mean(axis=0)
        ens_pert = self.ensemble - ens_mean
        C_xy = (ens_pert.T @ G_pert) / (J - 1)

        # Kalman gain
        K = C_xy @ np.linalg.inv(C_yy)

        # update each member with perturbed observations
        for j in range(J):
            y_pert = y_obs + noise_std * np.random.randn(len(y_obs))
            self.ensemble[j] += K @ (y_pert - G[j])


# ---------------------------------------------------------------------------
# Expected Information Gain (EIG) for sequential BED
# ---------------------------------------------------------------------------

def estimate_eig(x_candidates, disc_model, ensemble, y_phys, noise_std):
    """
    Estimate EIG at each candidate location.
    Uses variance reduction in prediction across ensemble as proxy for info gain.
    """
    eig_vals = np.zeros(len(x_candidates))
    for i, xc in enumerate(x_candidates):
        xc_arr = np.array([[xc]])
        preds = []
        for j in range(ensemble.J):
            w_flat = ensemble.ensemble[j]
            # temporarily set discrepancy weights
            n_w1 = disc_model.W1.size
            n_b1 = disc_model.b1.size
            n_w2 = disc_model.W2.size
            W1 = w_flat[:n_w1].reshape(disc_model.W1.shape)
            b1 = w_flat[n_w1:n_w1+n_b1]
            W2 = w_flat[n_w1+n_b1:n_w1+n_b1+n_w2].reshape(disc_model.W2.shape)
            b2 = w_flat[n_w1+n_b1+n_w2:]
            h = np.tanh(xc_arr @ W1 + b1)
            pred = (h @ W2 + b2).ravel()
            preds.append(pred[0])
        preds = np.array(preds)
        eig_vals[i] = np.var(preds)
    return eig_vals


# ---------------------------------------------------------------------------
# Demo: sequential BED for model discrepancy learning
# ---------------------------------------------------------------------------

def demo():
    np.random.seed(42)
    theta = np.array([1.0, 2.0])
    noise_std = 0.05

    # candidate experiment locations
    x_candidates = np.linspace(0, 2 * np.pi, 50)

    # initialise discrepancy model
    disc = DiscrepancyMLP(hidden=16, lr=5e-3)
    param_dim = disc.W1.size + disc.b1.size + disc.W2.size + disc.b2.size

    # initialise EKI ensemble
    prior_mean = np.concatenate([disc.W1.ravel(), disc.b1, disc.W2.ravel(), disc.b2])
    prior_std = np.ones(param_dim) * 0.5
    eki = EnsembleKalmanInversion(n_ensemble=50, param_dim=param_dim,
                                  prior_mean=prior_mean, prior_std=prior_std)

    # sequential experiment loop
    n_rounds = 8
    selected_x = []
    selected_y = []

    print("Sequential BED for model discrepancy learning")
    print("=" * 55)

    for r in range(n_rounds):
        # compute EIG at each candidate
        y_phys = np.array([physics_model(xc, theta) for xc in x_candidates])
        eig = estimate_eig(x_candidates, disc, eki, y_phys, noise_std)

        # select highest EIG location
        best_idx = np.argmax(eig)
        x_new = x_candidates[best_idx]
        y_true = true_system(x_new, theta) + np.random.randn() * noise_std
        y_phys_new = physics_model(x_new, theta)
        delta_obs = y_true - y_phys_new

        selected_x.append(x_new)
        selected_y.append(delta_obs)

        print(f"  Round {r+1}: x={x_new:.3f}, observed discrepancy={delta_obs:.4f}, "
              f"EIG={eig[best_idx]:.4f}")

        # update discrepancy model with all collected data
        X_train = np.array(selected_x).reshape(-1, 1)
        Y_train = np.array(selected_y).reshape(-1, 1)
        disc_copy = DiscrepancyMLP(hidden=16, lr=5e-3)
        disc_copy.fit(X_train, Y_train, epochs=300)

        # EKI update
        def pred_fn(x_arr, w_flat):
            n_w1 = 16
            W1 = w_flat[:n_w1].reshape(1, 16)
            b1 = w_flat[n_w1:n_w1+16]
            W2 = w_flat[n_w1+16:n_w1+16+16].reshape(16, 1)
            b2 = w_flat[n_w1+16+16:]
            h = np.tanh(x_arr @ W1 + b1)
            return (h @ W2 + b2).ravel()

        G = np.zeros((eki.J, len(selected_x)))
        for j in range(eki.J):
            G[j] = pred_fn(X_train, eki.ensemble[j])
        eki.update(G, np.array(selected_y), noise_std)

    # evaluate final discrepancy model
    print("\nFinal evaluation:")
    x_eval = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
    true_disc = np.array([true_system(x, theta) - physics_model(x, theta)
                          for x in x_eval.ravel()])
    learned_disc = disc_copy.forward(x_eval).ravel()
    rmse = np.sqrt(np.mean((true_disc - learned_disc)**2))
    print(f"  Discrepancy RMSE: {rmse:.4f}")
    print(f"  Selected locations: {[f'{x:.2f}' for x in selected_x]}")


if __name__ == "__main__":
    demo()
