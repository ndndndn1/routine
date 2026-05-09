"""
Reference implementation: Invertible Generative Model via Triangular Map Composition
(arXiv:2509.03910 -- van Leeuwen, Brune, Carioni, 2025)

Core idea: Compose an upper triangular map T_U (posterior sampler) and a lower
triangular map T_L (likelihood sampler) into a single invertible map G = T_L o T_U
that maps latent (z', z) -> (x, y).  Forward pass samples the likelihood p(y|x),
inverse pass samples the posterior p(x|y).

This demo uses simple affine+nonlinear parameterisations (no torch/tensorflow).
We demonstrate on a 1D inverse problem: y = a*x^2 + noise, which has a
two-modal posterior for x given y (positive and negative root).

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.stats import norm


# -----------------------------------------------------------------------
# 1. Triangular map components (simple parametric forms)
# -----------------------------------------------------------------------

class LowerTriangularMap:
    """
    T_L: (x, z) -> (x, y)
    y = f(x, z) = mu_y(x) + sigma_y(x) * z

    This samples from the likelihood p(y|x) when z ~ N(0,1).
    mu_y(x) and sigma_y(x) define the conditional distribution y|x.
    """

    def __init__(self):
        # Parameters for the forward model: y = a*x^2 + noise
        self.a = 1.0
        self.noise_std = 0.3

    def mu_y(self, x):
        """Conditional mean of y given x."""
        return self.a * x**2

    def sigma_y(self, x):
        """Conditional std of y given x."""
        return self.noise_std * np.ones_like(x)

    def forward(self, x, z):
        """(x, z) -> (x, y): sample y from p(y|x)."""
        y = self.mu_y(x) + self.sigma_y(x) * z
        return x, y

    def inverse(self, x, y):
        """(x, y) -> (x, z): recover z from (x, y)."""
        z = (y - self.mu_y(x)) / self.sigma_y(x)
        return x, z

    def log_det_jac(self, x):
        """Log |det J| of the map (x,z)->(x,y) w.r.t. z.
        Since y = mu + sigma*z, dy/dz = sigma, so log|det| = log(sigma)."""
        return np.log(self.sigma_y(x))


class UpperTriangularMap:
    """
    T_U: (z', y) -> (x, y)
    x = g(z', y) = mu_x(y) + sigma_x(y) * z'

    This samples from the posterior p(x|y) when z' ~ N(0,1).
    We learn/set mu_x(y) and sigma_x(y) to approximate the true posterior.
    """

    def __init__(self, prior_std=2.0):
        self.prior_std = prior_std
        # These will be fit to approximate the posterior
        # For the quadratic model y = x^2 + noise, the posterior is bimodal.
        # We use a Gaussian mixture approximation parameterised per y.

    def mu_x(self, y):
        """Conditional mean(s) of x given y.
        For y = x^2, the MAP estimate is +/- sqrt(y).
        We use the positive root as the mean for a simple unimodal demo."""
        return np.sqrt(np.maximum(y, 0.0))

    def sigma_x(self, y):
        """Conditional std of x given y.
        Derived from linearisation: dx/dy = 1/(2x) near x=sqrt(y),
        so sigma_x ~ noise_std / (2*sqrt(y))."""
        return 0.3 / (2.0 * np.sqrt(np.maximum(y, 0.01)) + 1e-8)

    def forward(self, zp, y):
        """(z', y) -> (x, y): sample x from approximate p(x|y)."""
        x = self.mu_x(y) + self.sigma_x(y) * zp
        return x, y

    def inverse(self, x, y):
        """(x, y) -> (z', y): recover z' from (x, y)."""
        zp = (x - self.mu_x(y)) / self.sigma_x(y)
        return zp, y

    def log_det_jac(self, y):
        """Log |det J| of the map (z',y)->(x,y) w.r.t. z'.
        Since x = mu + sigma*z', dx/dz' = sigma."""
        return np.log(self.sigma_x(y))


# -----------------------------------------------------------------------
# 2. Composed invertible map G = T_L o T_U
# -----------------------------------------------------------------------

class InvertibleGenerativeModel:
    """
    G: (z', z) -> (x, y)
    G = T_L o T_U

    Step 1: T_U maps (z', z) -> (x_temp, z)  ... wait, need to be careful.

    Actually, the composition works as follows:
      - T_U: (z', y_placeholder) -> (x, y_placeholder)  [acts on first component]
      - T_L: (x, z) -> (x, y)                            [acts on second component]

    The correct composed map:
      Given (z', z):
        x = g(z', z)       [upper map: z plays role of "conditioning" variable]
        y = f(x, z)         [lower map: but z is reused... ]

    Per the paper, the composition is:
      G(z', z) = T_L(T_U(z', z))

    T_U interprets input as (z', z) and outputs (g(z', z), z) = (x, z)
    T_L interprets input as (x, z) and outputs (x, f(x, z)) = (x, y)

    So: G(z', z) = (g(z', z), f(g(z', z), z)) = (x, y)
    """

    def __init__(self, T_L, T_U):
        self.T_L = T_L
        self.T_U = T_U

    def forward(self, zp, z):
        """(z', z) -> (x, y)"""
        # Step 1: Upper map -- (z', z) -> (x, z)
        x, z_pass = self.T_U.forward(zp, z)
        # Step 2: Lower map -- (x, z) -> (x, y)
        x_pass, y = self.T_L.forward(x, z_pass)
        return x_pass, y

    def inverse(self, x, y):
        """(x, y) -> (z', z)"""
        # Step 1: Inverse lower map -- (x, y) -> (x, z)
        x_pass, z = self.T_L.inverse(x, y)
        # Step 2: Inverse upper map -- (x, z) -> (z', z)
        zp, z_pass = self.T_U.inverse(x_pass, z)
        return zp, z_pass

    def log_density(self, x, y):
        """
        log p(x, y) = log p_z(z) + log p_{z'}(z')
                     + log|det J_{T_L^{-1}}| + log|det J_{T_U^{-1}}|

        where (z', z) = G^{-1}(x, y) and p_z, p_{z'} are standard normal.
        For triangular maps, the log-det terms come from the diagonal blocks.
        """
        zp, z = self.inverse(x, y)
        # Log-density of reference distributions
        log_pz = norm.logpdf(z)
        log_pzp = norm.logpdf(zp)
        # Log-det of inverse maps (negative of forward log-dets)
        log_det_TL_inv = -self.T_L.log_det_jac(x)   # d(z)/d(y) = 1/sigma_y
        log_det_TU_inv = -self.T_U.log_det_jac(z)    # d(z')/d(x) = 1/sigma_x
        return log_pz + log_pzp + log_det_TL_inv + log_det_TU_inv

    def sample_likelihood(self, x, n_samples=1000):
        """Sample y ~ p(y|x) by fixing x, drawing z ~ N(0,1)."""
        z = np.random.randn(n_samples)
        _, y = self.T_L.forward(np.full(n_samples, x), z)
        return y

    def sample_posterior(self, y, n_samples=1000):
        """Sample x ~ p(x|y) by fixing y, drawing z' ~ N(0,1)."""
        zp = np.random.randn(n_samples)
        x, _ = self.T_U.forward(zp, np.full(n_samples, y))
        return x


# -----------------------------------------------------------------------
# 3. Training via maximum likelihood (simplified)
# -----------------------------------------------------------------------

def fit_upper_map_params(T_U, T_L, x_data, y_data, lr=0.01, n_iter=500):
    """
    Fit the upper triangular map parameters to approximate the true posterior.

    We minimise the negative log-likelihood of the joint model over training
    data {(x_i, y_i)}.  For simplicity we optimise a scale parameter for
    sigma_x(y) via gradient-free coordinate search.

    In a full implementation this would use automatic differentiation.
    """
    best_nll = np.inf
    best_scale = 1.0
    base_noise = T_L.noise_std

    for scale in np.linspace(0.1, 3.0, 60):
        # Temporarily set sigma_x scale
        original_sigma = T_U.sigma_x

        def new_sigma(y, s=scale, bn=base_noise):
            return s * bn / (2.0 * np.sqrt(np.maximum(y, 0.01)) + 1e-8)

        T_U.sigma_x = new_sigma

        # Compute NLL
        G = InvertibleGenerativeModel(T_L, T_U)
        nll = -np.mean(G.log_density(x_data, y_data))

        if nll < best_nll:
            best_nll = nll
            best_scale = scale

        T_U.sigma_x = original_sigma

    # Set best parameters
    base = T_L.noise_std

    def fitted_sigma(y, s=best_scale, bn=base):
        return s * bn / (2.0 * np.sqrt(np.maximum(y, 0.01)) + 1e-8)

    T_U.sigma_x = fitted_sigma
    print(f"  Fitted sigma_x scale = {best_scale:.2f}  (NLL = {best_nll:.3f})")


# -----------------------------------------------------------------------
# 4. Demo: 1D quadratic inverse problem
# -----------------------------------------------------------------------

def demo():
    print("=" * 65)
    print("Invertible Generative Model via Triangular Map Composition")
    print("(arXiv:2509.03910)")
    print("=" * 65)

    np.random.seed(42)

    # --- Ground truth forward model: y = x^2 + noise ---
    n_data = 2000
    x_true = np.random.randn(n_data) * 1.5          # prior: N(0, 1.5^2)
    noise_std = 0.3
    y_true = x_true**2 + noise_std * np.random.randn(n_data)

    # --- Build triangular maps ---
    T_L = LowerTriangularMap()
    T_L.noise_std = noise_std

    T_U = UpperTriangularMap()

    # --- Fit upper map to training data ---
    print("\n[1] Fitting upper triangular map (posterior approximation)...")
    fit_upper_map_params(T_U, T_L, x_true, y_true)

    # --- Compose into invertible model ---
    G = InvertibleGenerativeModel(T_L, T_U)

    # --- Test forward sampling (likelihood) ---
    print("\n[2] Forward sampling: p(y | x=1.0)")
    x_test = 1.0
    y_samples = G.sample_likelihood(x_test, n_samples=5000)
    print(f"  True mean:  {x_test**2:.3f}")
    print(f"  Sample mean: {np.mean(y_samples):.3f}")
    print(f"  Sample std:  {np.std(y_samples):.3f}  (true: {noise_std:.3f})")

    # --- Test inverse sampling (posterior) ---
    print("\n[3] Inverse sampling: p(x | y=1.0)")
    y_obs = 1.0
    x_posterior = G.sample_posterior(y_obs, n_samples=5000)
    print(f"  True roots:      +/-{np.sqrt(y_obs):.3f}")
    print(f"  Posterior mean:   {np.mean(x_posterior):.3f}")
    print(f"  Posterior std:    {np.std(x_posterior):.3f}")
    # Note: unimodal approximation centres on positive root

    # --- Test invertibility ---
    print("\n[4] Invertibility check: G^{-1}(G(z', z)) = (z', z)")
    zp_test = np.array([0.5, -1.2, 2.0, 0.0, -0.7])
    z_test = np.array([1.0, -0.3, 0.8, -1.5, 0.2])
    x_fwd, y_fwd = G.forward(zp_test, z_test)
    zp_rec, z_rec = G.inverse(x_fwd, y_fwd)
    err_zp = np.max(np.abs(zp_test - zp_rec))
    err_z = np.max(np.abs(z_test - z_rec))
    print(f"  Max reconstruction error |z' - z'_rec| = {err_zp:.2e}")
    print(f"  Max reconstruction error |z  - z_rec|  = {err_z:.2e}")
    assert err_zp < 1e-10 and err_z < 1e-10, "Invertibility failed!"
    print("  PASSED: exact invertibility verified.")

    # --- Joint density evaluation ---
    print("\n[5] Joint log-density evaluation at (x, y) = (1.0, 1.0)")
    log_p = G.log_density(np.array([1.0]), np.array([1.0]))
    print(f"  log p(x=1, y=1) = {log_p[0]:.4f}")

    # --- Posterior variance as function of y ---
    print("\n[6] Posterior uncertainty vs observation value:")
    for y_val in [0.25, 1.0, 4.0, 9.0]:
        x_post = G.sample_posterior(y_val, n_samples=10000)
        print(f"  y={y_val:5.2f}  ->  E[x|y]={np.mean(x_post):+.3f}, "
              f"Std[x|y]={np.std(x_post):.4f}")

    print("\n" + "=" * 65)
    print("Demo complete. The composed triangular map G = T_L o T_U provides")
    print("an invertible generative model for both forward and inverse problems.")
    print("=" * 65)


if __name__ == "__main__":
    demo()
