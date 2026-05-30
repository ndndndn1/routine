"""
Reference implementation: Heteroscedastic GP for Plasma Etching UQ & Optimization
(arXiv:2511.04990)

Core idea: A heteroscedastic Gaussian process surrogate captures input-dependent
noise (process variation) in semiconductor plasma etching, then a reliability-based
design optimisation finds process parameters that minimise spatial etch-depth
variability across the wafer while maintaining reliability.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Simulated plasma etching process
# ---------------------------------------------------------------------------

def etch_depth(pressure, flow_rate, rf_power, wafer_pos, noise_scale=1.0):
    """
    Simulated etch depth at a wafer position given process parameters.
    wafer_pos: (2,) normalised (r, theta) position on wafer
    Returns: scalar etch depth (nm)
    """
    r, theta = wafer_pos
    base = 100 + 0.5 * pressure + 0.3 * flow_rate + 0.8 * rf_power
    spatial = 5 * r * np.cos(theta) + 3 * r**2
    # process-dependent noise (heteroscedastic)
    noise_var = 1.0 + 0.02 * pressure + 0.01 * rf_power**2 * 0.001
    noise = np.random.randn() * np.sqrt(noise_var) * noise_scale
    return base + spatial + noise


def wafer_positions(n=9):
    """Standard 9-point wafer measurement locations."""
    positions = []
    for r in [0.0, 0.5, 1.0]:
        if r == 0:
            positions.append((0.0, 0.0))
        else:
            for th in np.linspace(0, 2*np.pi, max(4, int(4*r)), endpoint=False):
                positions.append((r, th))
                if len(positions) >= n:
                    break
        if len(positions) >= n:
            break
    return positions[:n]


# ---------------------------------------------------------------------------
# Heteroscedastic Gaussian Process (simplified)
# ---------------------------------------------------------------------------

class HeteroscedasticGP:
    """
    GP with input-dependent noise variance.
    Uses two GPs: one for the mean function, one for log noise variance.
    """

    def __init__(self, length_scale=1.0, signal_var=1.0):
        self.ls = length_scale
        self.sv = signal_var
        self.X_train = None
        self.y_train = None
        self.log_noise_var = None

    def _kernel(self, X1, X2):
        """RBF kernel."""
        sq_dist = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
        return self.sv * np.exp(-0.5 * sq_dist / self.ls**2)

    def fit(self, X, y, n_noise_iter=5):
        """
        Fit GP with heteroscedastic noise estimation.
        X: (N, d), y: (N,)
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        N = len(X)

        # initial homoscedastic fit
        self.log_noise_var = np.zeros(N)

        for it in range(n_noise_iter):
            noise_var = np.exp(self.log_noise_var) + 1e-4
            K = self._kernel(X, X) + np.diag(noise_var)
            self._K_inv = np.linalg.inv(K + 1e-6 * np.eye(N))
            self._alpha = self._K_inv @ y

            # estimate local noise variance from residuals
            mu = K @ self._alpha
            residuals = (y - mu)**2
            # smooth residuals with local averaging
            for i in range(N):
                dists = np.sum((X - X[i])**2, axis=1)
                weights = np.exp(-dists / (2 * self.ls**2))
                weights /= weights.sum()
                self.log_noise_var[i] = np.log(np.dot(weights, residuals) + 1e-6)

    def predict(self, X_test):
        """Returns: mean, total_var (epistemic + aleatoric)."""
        k_star = self._kernel(X_test, self.X_train)
        k_ss = self._kernel(X_test, X_test)

        mu = k_star @ self._alpha
        v = k_star @ self._K_inv @ k_star.T
        epistemic_var = np.diag(k_ss - v)

        # interpolate aleatoric noise
        noise_var_interp = k_star @ self._K_inv @ np.exp(self.log_noise_var)
        noise_var_interp = np.clip(noise_var_interp, 1e-4, None)

        total_var = epistemic_var + noise_var_interp
        return mu, total_var, epistemic_var, noise_var_interp


# ---------------------------------------------------------------------------
# Reliability-Based Design Optimisation (RBDO)
# ---------------------------------------------------------------------------

def rbdo_objective(params, gp, wafer_pos_list, target_depth=150):
    """
    Objective: minimise spatial variability of etch depth across wafer.
    Constraint: mean depth should be close to target, with reliability.
    """
    n_pos = len(wafer_pos_list)
    X_test = np.zeros((n_pos, 3 + 2))

    for i, (r, th) in enumerate(wafer_pos_list):
        X_test[i] = np.concatenate([params, [r, th]])

    mu, total_var, _, aleatoric = gp.predict(X_test)

    # spatial variability (std across wafer positions)
    spatial_var = np.std(mu)

    # reliability: probability that mean depth is within tolerance
    mean_depth = np.mean(mu)
    mean_uncertainty = np.sqrt(np.mean(total_var))
    deviation = np.abs(mean_depth - target_depth)

    # combined objective: spatial uniformity + reliability penalty
    penalty = max(0, deviation - 2 * mean_uncertainty) * 10
    return spatial_var + penalty, spatial_var, mean_depth, mean_uncertainty


def optimise_process(gp, wafer_pos_list, target_depth=150, n_iter=200):
    """Simple random search RBDO."""
    best_params = None
    best_obj = np.inf

    for _ in range(n_iter):
        params = np.array([
            np.random.uniform(50, 200),   # pressure
            np.random.uniform(10, 100),   # flow_rate
            np.random.uniform(50, 300),   # rf_power
        ])
        obj, sv, md, mu = rbdo_objective(params, gp, wafer_pos_list, target_depth)
        if obj < best_obj:
            best_obj = obj
            best_params = params
            best_info = (sv, md, mu)

    return best_params, best_obj, best_info


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    np.random.seed(42)

    # wafer measurement positions
    positions = wafer_positions(9)

    # generate training data
    print("Generating plasma etching experimental data...")
    N_exp = 80
    X_all = []
    y_all = []
    for _ in range(N_exp):
        pressure = np.random.uniform(50, 200)
        flow_rate = np.random.uniform(10, 100)
        rf_power = np.random.uniform(50, 300)
        for pos in positions:
            x_row = np.array([pressure, flow_rate, rf_power, pos[0], pos[1]])
            depth = etch_depth(pressure, flow_rate, rf_power, pos)
            X_all.append(x_row)
            y_all.append(depth)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    print(f"  Total data points: {len(X_all)}")

    # subsample for GP (computational tractability)
    idx = np.random.choice(len(X_all), 200, replace=False)
    X_train = X_all[idx]
    y_train = y_all[idx]

    # fit heteroscedastic GP
    print("Fitting heteroscedastic GP surrogate...")
    gp = HeteroscedasticGP(length_scale=30.0, signal_var=100.0)
    gp.fit(X_train, y_train, n_noise_iter=3)

    # predict at a test point
    test_params = np.array([120, 50, 150])
    X_test = np.array([[120, 50, 150, 0.0, 0.0]])
    mu, total_var, epi, ale = gp.predict(X_test)
    print(f"\nTest prediction (center, P=120, F=50, RF=150):")
    print(f"  Mean depth: {mu[0]:.1f} nm")
    print(f"  Epistemic std: {np.sqrt(epi[0]):.2f}")
    print(f"  Aleatoric std: {np.sqrt(ale[0]):.2f}")
    print(f"  Total std: {np.sqrt(total_var[0]):.2f}")

    # RBDO
    print("\nRunning RBDO for optimal process parameters...")
    best_params, best_obj, (sv, md, mu_unc) = optimise_process(
        gp, positions, target_depth=150, n_iter=300
    )
    print(f"  Optimal params: P={best_params[0]:.1f}, F={best_params[1]:.1f}, "
          f"RF={best_params[2]:.1f}")
    print(f"  Spatial variability: {sv:.2f}")
    print(f"  Mean depth: {md:.1f} nm (target: 150)")
    print(f"  Mean uncertainty: {mu_unc:.2f}")


if __name__ == "__main__":
    demo()
