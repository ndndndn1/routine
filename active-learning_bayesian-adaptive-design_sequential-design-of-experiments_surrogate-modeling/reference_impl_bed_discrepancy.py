"""
Reference implementation of Active Learning of Model Discrepancy with Bayesian
Experimental Design (arXiv:2502.05372). Sequential BED loop with ensemble-based
information gain indicator and neural network discrepancy correction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Physics model (intentionally imperfect: linear approx of nonlinear truth)
# ---------------------------------------------------------------------------

def physics_model(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Linear physics model: y = theta[0] + theta[1]*x (ignores true nonlinearity)."""
    return theta[0] + theta[1] * x


def truth(x: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
    """Ground truth: nonlinear system y = sin(2*x) + 0.3*x^2, with observation noise."""
    return np.sin(2.0 * x) + 0.3 * x ** 2 + noise_std * np.random.randn(*x.shape)


# ---------------------------------------------------------------------------
# Neural network discrepancy correction delta(x)
# ---------------------------------------------------------------------------

class DiscrepancyNet(nn.Module):
    """Small MLP that approximates the additive model discrepancy delta(x)."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted discrepancy for input x (shape [N,1])."""
        return self.net(x)


def corrected_model(x: np.ndarray, theta: np.ndarray, delta_net: DiscrepancyNet) -> np.ndarray:
    """Physics model plus learned discrepancy correction."""
    phys = physics_model(x, theta)
    xt = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        delta = delta_net(xt).numpy().reshape(x.shape)
    return phys + delta


# ---------------------------------------------------------------------------
# Ensemble Kalman Inversion (EKI) – inference step
# ---------------------------------------------------------------------------

def eki_update(
    ensemble: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    delta_net: DiscrepancyNet,
    obs_noise_var: float,
) -> np.ndarray:
    """Run one EKI update step; returns updated ensemble of theta particles."""
    J = ensemble.shape[0]
    # Forward map for each particle
    G = np.stack([corrected_model(x_obs, ensemble[j], delta_net) for j in range(J)])  # [J, N_obs]

    # Ensemble means
    theta_bar = ensemble.mean(axis=0)
    G_bar = G.mean(axis=0)

    # Cross-covariance C_theta_G and auto-covariance C_GG
    dTheta = ensemble - theta_bar          # [J, d_theta]
    dG = G - G_bar                         # [J, N_obs]
    C_tG = (dTheta.T @ dG) / (J - 1)      # [d_theta, N_obs]
    C_GG = (dG.T @ dG) / (J - 1)          # [N_obs, N_obs]

    # Kalman gain
    obs_noise = obs_noise_var * np.eye(len(y_obs))
    K = C_tG @ np.linalg.solve((C_GG + obs_noise).T, np.eye(len(y_obs))).T  # [d_theta, N_obs]

    # Perturbed observations
    y_pert = y_obs[None, :] + np.sqrt(obs_noise_var) * np.random.randn(J, len(y_obs))

    # Update
    updated = ensemble + (K @ (y_pert - G).T).T  # [J, d_theta]
    return updated


# ---------------------------------------------------------------------------
# Ensemble-based information gain indicator
# ---------------------------------------------------------------------------

def ensemble_spread(ensemble: np.ndarray) -> float:
    """Return total variance (trace of covariance) as a spread measure."""
    return float(np.trace(np.cov(ensemble.T)))


def information_gain_indicator(
    x_cand: float,
    ensemble: np.ndarray,
    delta_net: DiscrepancyNet,
    obs_noise_var: float,
    n_eki_steps: int = 3,
) -> float:
    """Approximate expected information gain for a single candidate design point."""
    spread_before = ensemble_spread(ensemble)

    # Simulate a hypothetical observation using ensemble predictive mean
    x_arr = np.array([x_cand])
    y_hat = corrected_model(x_arr, ensemble.mean(axis=0), delta_net)

    # Run EKI with hypothetical observation
    ens_updated = ensemble.copy()
    for _ in range(n_eki_steps):
        ens_updated = eki_update(ens_updated, x_arr, y_hat, delta_net, obs_noise_var)

    spread_after = ensemble_spread(ens_updated)
    return max(spread_before - spread_after, 0.0)


# ---------------------------------------------------------------------------
# Online discrepancy model update
# ---------------------------------------------------------------------------

def update_discrepancy(
    delta_net: DiscrepancyNet,
    x_data: np.ndarray,
    y_data: np.ndarray,
    theta_mean: np.ndarray,
    n_epochs: int = 200,
    lr: float = 1e-3,
) -> None:
    """Retrain discrepancy net on residuals: delta(x) = y_obs - physics(x, theta)."""
    residuals = y_data - physics_model(x_data, theta_mean)
    Xt = torch.tensor(x_data.reshape(-1, 1), dtype=torch.float32)
    Rt = torch.tensor(residuals.reshape(-1, 1), dtype=torch.float32)

    optimizer = optim.Adam(delta_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(delta_net(Xt), Rt)
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Sequential BED loop
# ---------------------------------------------------------------------------

def sequential_bed_loop(
    n_iterations: int = 20,
    n_candidates: int = 50,
    ensemble_size: int = 100,
    obs_noise_var: float = 0.0025,
    n_eki_steps: int = 5,
    x_bounds: tuple = (0.0, 3.0),
    seed: int = 42,
) -> dict:
    """Run the full sequential BED loop, returning collected data and history."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # True parameter dimension: theta = [offset, slope]
    d_theta = 2
    # Initial prior ensemble: theta ~ N([0.5, 0.8], 0.3^2 * I)
    ensemble = np.random.randn(ensemble_size, d_theta) * 0.3 + np.array([0.5, 0.8])

    delta_net = DiscrepancyNet(hidden=32)

    # Seed dataset: two initial observations at boundaries
    x_init = np.array([x_bounds[0] + 0.1, x_bounds[1] - 0.1])
    y_init = truth(x_init)
    x_collected = x_init.copy()
    y_collected = y_init.copy()

    # Warm-start: initial EKI and discrepancy update
    for _ in range(n_eki_steps):
        ensemble = eki_update(ensemble, x_collected, y_collected, delta_net, obs_noise_var)
    update_discrepancy(delta_net, x_collected, y_collected, ensemble.mean(axis=0))

    ig_history = []
    design_history = list(x_init)

    candidates = np.linspace(x_bounds[0], x_bounds[1], n_candidates)

    for step in range(n_iterations):
        # Score all candidates
        ig_scores = np.array([
            information_gain_indicator(xc, ensemble, delta_net, obs_noise_var, n_eki_steps=3)
            for xc in candidates
        ])

        best_idx = int(np.argmax(ig_scores))
        x_new = candidates[best_idx]
        ig_history.append(ig_scores[best_idx])
        design_history.append(x_new)

        # Collect observation
        y_new = truth(np.array([x_new]))[0]
        x_collected = np.append(x_collected, x_new)
        y_collected = np.append(y_collected, y_new)

        # EKI update with new observation
        for _ in range(n_eki_steps):
            ensemble = eki_update(
                ensemble, np.array([x_new]), np.array([y_new]), delta_net, obs_noise_var
            )

        # Update discrepancy correction on all collected data
        update_discrepancy(delta_net, x_collected, y_collected, ensemble.mean(axis=0))

        theta_est = ensemble.mean(axis=0)
        print(
            f"Step {step+1:3d} | x*={x_new:.3f} | IG={ig_scores[best_idx]:.4f} "
            f"| theta=[{theta_est[0]:.3f},{theta_est[1]:.3f}] | N={len(x_collected)}"
        )

    return {
        "x_collected": x_collected,
        "y_collected": y_collected,
        "ensemble": ensemble,
        "delta_net": delta_net,
        "ig_history": ig_history,
        "design_history": design_history,
    }


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model(
    x_eval: np.ndarray,
    ensemble: np.ndarray,
    delta_net: DiscrepancyNet,
) -> tuple:
    """Return mean and std of corrected model predictions over the evaluation grid."""
    preds = np.stack([
        corrected_model(x_eval, ensemble[j], delta_net) for j in range(ensemble.shape[0])
    ])
    return preds.mean(axis=0), preds.std(axis=0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Sequential BED with Model Discrepancy Correction ===")
    print("Truth: y = sin(2x) + 0.3x^2 | Physics: y = theta0 + theta1*x\n")

    results = sequential_bed_loop(
        n_iterations=20,
        n_candidates=50,
        ensemble_size=100,
        obs_noise_var=0.0025,
        n_eki_steps=5,
        x_bounds=(0.0, 3.0),
        seed=42,
    )

    x_eval = np.linspace(0.0, 3.0, 200)
    y_true = np.sin(2.0 * x_eval) + 0.3 * x_eval ** 2
    mean_pred, std_pred = evaluate_model(x_eval, results["ensemble"], results["delta_net"])

    rmse = float(np.sqrt(np.mean((mean_pred - y_true) ** 2)))
    theta_final = results["ensemble"].mean(axis=0)

    print(f"\n--- Results ---")
    print(f"Final theta estimate : [{theta_final[0]:.4f}, {theta_final[1]:.4f}]")
    print(f"Corrected model RMSE : {rmse:.4f}")
    print(f"Observations collected: {len(results['x_collected'])}")
    print(f"Design history (last 5): {[f'{x:.3f}' for x in results['design_history'][-5:]]}")
