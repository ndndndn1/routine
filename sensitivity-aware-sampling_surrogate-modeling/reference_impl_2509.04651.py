"""Minimal reference implementation: Sensitivity-Driven Adaptive Surrogate
Modeling (arXiv:2509.04651).

Demonstrates combining solution sensitivity and surrogate error to adaptively
refine a surrogate for a 1D ODE-governed trajectory optimization.

Run:
    python reference_impl_2509.04651.py
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Expensive component function (e.g., aerodynamic coefficient from CFD) ------
# ---------------------------------------------------------------------------
def true_component(p: np.ndarray) -> np.ndarray:
    """True expensive component function g(p)."""
    return np.sin(2 * p) + 0.3 * np.cos(5 * p) + 0.1 * p


# ---------------------------------------------------------------------------
# RBF Kernel Interpolation Surrogate -----------------------------------------
# ---------------------------------------------------------------------------
class RBFSurrogate:
    def __init__(self, epsilon: float = 1.0):
        self.eps = epsilon
        self.P: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def _rbf(self, r):
        return np.exp(-(self.eps * r)**2)

    def fit(self, P, g):
        self.P = P.copy().flatten()
        self.g = g.copy()
        D = np.abs(P.reshape(-1, 1) - P.reshape(1, -1))
        Phi = self._rbf(D)
        self.weights = np.linalg.solve(Phi + 1e-10 * np.eye(len(P)), g)

    def predict(self, p_eval):
        p_eval = np.atleast_1d(p_eval)
        D = np.abs(p_eval.reshape(-1, 1) - self.P.reshape(1, -1))
        Phi = self._rbf(D)
        return Phi @ self.weights

    def error_estimate(self, p_eval):
        """Power function-based error estimate."""
        p_eval = np.atleast_1d(p_eval)
        D_eval = np.abs(p_eval.reshape(-1, 1) - self.P.reshape(1, -1))
        Phi_eval = self._rbf(D_eval)
        D_train = np.abs(self.P.reshape(-1, 1) - self.P.reshape(1, -1))
        Phi_train = self._rbf(D_train) + 1e-10 * np.eye(len(self.P))
        Phi_inv = np.linalg.inv(Phi_train)
        power = np.zeros(len(p_eval))
        for i in range(len(p_eval)):
            lam = Phi_inv @ Phi_eval[i]
            power[i] = max(1.0 - Phi_eval[i] @ lam, 1e-10)
        return np.sqrt(power)


# ---------------------------------------------------------------------------
# ODE system: ẋ = -g(x) * x + u(t) -----------------------------------------
# ---------------------------------------------------------------------------
def simulate_trajectory(surrogate, x0: float = 1.0, T: float = 3.0,
                        dt: float = 0.05):
    """Simulate ODE using surrogate component function."""
    ts = np.arange(0, T, dt)
    xs = np.zeros(len(ts))
    xs[0] = x0
    for i in range(len(ts) - 1):
        g_val = surrogate.predict(np.array([xs[i]]))[0]
        dxdt = -g_val * xs[i] + 0.5 * np.sin(ts[i])
        xs[i + 1] = xs[i] + dt * dxdt
    return ts, xs


def trajectory_objective(xs: np.ndarray) -> float:
    """J = integral of x² (minimize energy)."""
    return np.mean(xs**2)


# ---------------------------------------------------------------------------
# Adjoint-based sensitivity computation --------------------------------------
# ---------------------------------------------------------------------------
def compute_sensitivity(surrogate, ts: np.ndarray, xs: np.ndarray,
                        dt: float = 0.05):
    """Compute dJ/dg(p) via discrete adjoint at each state x_i.

    Returns sensitivity S(x_i) = how much g at state x_i affects J."""
    N = len(ts)
    lam = np.zeros(N)  # adjoint variable
    lam[-1] = 2 * xs[-1] / N  # terminal condition: dJ/dx_N

    for i in range(N - 2, -1, -1):
        g_val = surrogate.predict(np.array([xs[i]]))[0]
        dlam = lam[i + 1] * (1 - dt * g_val) + 2 * xs[i] / N
        lam[i] = dlam

    sensitivity = np.abs(lam * xs * dt)
    return sensitivity


# ---------------------------------------------------------------------------
# Sensitivity-driven acquisition function ------------------------------------
# ---------------------------------------------------------------------------
def acquisition(surrogate, xs: np.ndarray, sensitivity: np.ndarray):
    """alpha(x_i) = |S(x_i)| * |epsilon(x_i)|"""
    err = surrogate.error_estimate(xs)
    return sensitivity * err


# ---------------------------------------------------------------------------
# Adaptive refinement loop ---------------------------------------------------
# ---------------------------------------------------------------------------
def adaptive_refinement(n_init: int = 5, n_refine: int = 15, seed: int = 42):
    rng = np.random.default_rng(seed)

    P_train = rng.uniform(-2, 2, n_init)
    g_train = true_component(P_train)
    surr = RBFSurrogate(epsilon=1.0)
    surr.fit(P_train, g_train)

    print("Sensitivity-Driven Adaptive Surrogate Refinement")
    print("=" * 55)

    for step in range(n_refine):
        ts, xs = simulate_trajectory(surr)
        J = trajectory_objective(xs)
        sens = compute_sensitivity(surr, ts, xs)
        acq = acquisition(surr, xs, sens)

        best_idx = np.argmax(acq)
        p_new = xs[best_idx]
        g_new = true_component(np.array([p_new]))[0]

        P_train = np.append(P_train, p_new)
        g_train = np.append(g_train, g_new)
        surr.fit(P_train, g_train)

        if (step + 1) % 5 == 0:
            ts_t, xs_t = simulate_with_true(ts)
            J_true = trajectory_objective(xs_t)
            print(f"  Step {step+1:2d}: J_surr={J:.4f}, J_true={J_true:.4f}, "
                  f"gap={abs(J-J_true):.4f}, n_eval={len(P_train)}")

    return surr, P_train


def simulate_with_true(ts, x0: float = 1.0, dt: float = 0.05):
    """Simulate using true component function (ground truth)."""
    xs = np.zeros(len(ts))
    xs[0] = x0
    for i in range(len(ts) - 1):
        g_val = true_component(np.array([xs[i]]))[0]
        dxdt = -g_val * xs[i] + 0.5 * np.sin(ts[i])
        xs[i + 1] = xs[i] + dt * dxdt
    return ts, xs


# ---------------------------------------------------------------------------
# Baseline: uniform refinement -----------------------------------------------
# ---------------------------------------------------------------------------
def uniform_refinement(n_init: int = 5, n_refine: int = 15, seed: int = 42):
    rng = np.random.default_rng(seed)
    P_train = rng.uniform(-2, 2, n_init)
    g_train = true_component(P_train)
    surr = RBFSurrogate(epsilon=1.0)

    P_extra = rng.uniform(-2, 2, n_refine)
    g_extra = true_component(P_extra)
    P_all = np.concatenate([P_train, P_extra])
    g_all = np.concatenate([g_train, g_extra])
    surr.fit(P_all, g_all)

    ts, xs = simulate_trajectory(surr)
    J = trajectory_objective(xs)
    ts_t, xs_t = simulate_with_true(ts)
    J_true = trajectory_objective(xs_t)
    return J, J_true, abs(J - J_true)


if __name__ == "__main__":
    surr, P = adaptive_refinement()

    print("\n--- Comparison (same budget = 20 evals) ---")
    J_unif, J_true, gap_unif = uniform_refinement()
    print(f"  Uniform:  J_surr={J_unif:.4f}, J_true={J_true:.4f}, gap={gap_unif:.4f}")

    ts, xs = simulate_trajectory(surr)
    J_sens = trajectory_objective(xs)
    ts_t, xs_t = simulate_with_true(ts)
    J_t = trajectory_objective(xs_t)
    gap_sens = abs(J_sens - J_t)
    print(f"  Sens-drv: J_surr={J_sens:.4f}, J_true={J_t:.4f}, gap={gap_sens:.4f}")
    print(f"  Gap reduction: {gap_unif/(gap_sens+1e-30):.1f}x")
