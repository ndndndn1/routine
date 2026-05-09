"""
Reference implementation: Variational Deep Material Network (VDMN)
(arXiv: 2512.18104)

Core idea: A hierarchical binary-tree homogenization model (Deep Material
Network) with variational encoding of microstructure uncertainty. The VDMN
propagates manufacturing process variations through the material model
to quantify uncertainty in macroscopic mechanical response, and solves
inverse problems (response -> microstructure) via the variational posterior.

Demonstrates:
  1. Binary-tree Deep Material Network (DMN) for 2-phase composite
     homogenization (Voigt-Reuss mixing).
  2. Variational encoder-decoder (numpy MLP) mapping microstructure
     descriptors to DMN parameters with learned uncertainty.
  3. Forward uncertainty propagation via Monte Carlo sampling.
  4. Inverse problem: infer microstructure from observed stress-strain.

Dependencies: numpy, scipy
Usage: python 2512.18104_reference_impl.py
"""

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 1. Deep Material Network (DMN) -- simplified 2D plane-stress
# ---------------------------------------------------------------------------

def isotropic_stiffness_2d(E, nu):
    """
    2D plane-stress stiffness matrix (3x3 Voigt notation).
    C = E/(1-nu^2) * [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]
    """
    factor = E / (1.0 - nu ** 2)
    C = factor * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])
    return C


def rotation_matrix_2d(theta):
    """
    Rotation transformation for Voigt stress/strain (2D).
    Transforms stiffness: C_rot = T @ C @ T^T
    """
    c, s = np.cos(theta), np.sin(theta)
    T = np.array([
        [c * c, s * s, 2 * c * s],
        [s * s, c * c, -2 * c * s],
        [-c * s, c * s, c * c - s * s]
    ])
    return T


def voigt_reuss_mix(C1, C2, w, theta=0.0):
    """
    Voigt-Reuss mixed homogenization at a single DMN node.

    C_eff = alpha * C_Voigt + (1-alpha) * C_Reuss
    where alpha = 0.5 (Hill average), and w is volume fraction of phase 1.

    theta rotates the local frame before mixing.
    """
    T = rotation_matrix_2d(theta)
    C1_rot = T @ C1 @ T.T
    C2_rot = T @ C2 @ T.T

    # Voigt (upper bound): volume-weighted stiffness
    C_voigt = w * C1_rot + (1.0 - w) * C2_rot

    # Reuss (lower bound): volume-weighted compliance
    S1 = np.linalg.inv(C1_rot + 1e-10 * np.eye(3))
    S2 = np.linalg.inv(C2_rot + 1e-10 * np.eye(3))
    S_reuss = w * S1 + (1.0 - w) * S2
    C_reuss = np.linalg.inv(S_reuss + 1e-10 * np.eye(3))

    # Hill average
    C_eff = 0.5 * (C_voigt + C_reuss)
    return C_eff


class DeepMaterialNetwork:
    """
    Binary-tree DMN with n_levels levels.
    Leaves = 2^n_levels material phases.
    Each internal node has parameters (w, theta) for mixing.

    For a 2-phase composite, leaves alternate between phase 1 and phase 2
    based on local volume fractions.
    """

    def __init__(self, n_levels=3):
        self.n_levels = n_levels
        self.n_leaves = 2 ** n_levels
        self.n_internal = self.n_leaves - 1

    def forward(self, C_phase1, C_phase2, weights, thetas, vf_assign=None):
        """
        Bottom-up homogenization through the binary tree.

        C_phase1, C_phase2: (3,3) stiffness of each phase
        weights: (n_internal,) volume fractions at each node
        thetas:  (n_internal,) rotation angles at each node
        vf_assign: (n_leaves,) fraction of phase1 at each leaf (0 or 1 typical)

        Returns: C_eff (3,3) effective stiffness
        """
        if vf_assign is None:
            # Default: alternating phases
            vf_assign = np.array([i % 2 for i in range(self.n_leaves)], dtype=float)

        # Leaf stiffness
        C_nodes = []
        for i in range(self.n_leaves):
            vf = vf_assign[i]
            C_leaf = vf * C_phase1 + (1.0 - vf) * C_phase2
            C_nodes.append(C_leaf)

        # Bottom-up homogenization
        node_idx = 0
        current_level = list(C_nodes)

        for level in range(self.n_levels):
            next_level = []
            n_pairs = len(current_level) // 2
            for j in range(n_pairs):
                w = weights[node_idx]
                theta = thetas[node_idx]
                C_left = current_level[2 * j]
                C_right = current_level[2 * j + 1]
                C_mixed = voigt_reuss_mix(C_left, C_right, w, theta)
                next_level.append(C_mixed)
                node_idx += 1
            current_level = next_level

        return current_level[0]  # root = effective stiffness

    def predict_stress(self, C_phase1, C_phase2, weights, thetas, strain, vf_assign=None):
        """Predict stress for given strain using homogenized stiffness."""
        C_eff = self.forward(C_phase1, C_phase2, weights, thetas, vf_assign)
        stress = C_eff @ strain
        return stress


# ---------------------------------------------------------------------------
# 2. Variational Encoder-Decoder (numpy MLP)
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def softplus(x):
    return np.log1p(np.exp(np.clip(x, -20, 20)))


class VariationalMLP:
    """
    Variational encoder-decoder for microstructure -> DMN parameters.

    Encoder: microstructure descriptors (e.g., volume fraction, aspect ratio)
             -> latent mean mu and log-variance log_var
    Decoder: latent z -> DMN parameters (weights, thetas)
    """

    def __init__(self, d_input, d_latent, n_dmn_params, hidden=32):
        self.d_input = d_input
        self.d_latent = d_latent
        self.n_dmn_params = n_dmn_params
        self.hidden = hidden

        sc1 = np.sqrt(2.0 / d_input)
        sc2 = np.sqrt(2.0 / hidden)
        sc3 = np.sqrt(2.0 / d_latent)

        # Encoder
        self.W_enc1 = np.random.randn(d_input, hidden) * sc1
        self.b_enc1 = np.zeros(hidden)
        self.W_mu = np.random.randn(hidden, d_latent) * sc2 * 0.5
        self.b_mu = np.zeros(d_latent)
        self.W_logvar = np.random.randn(hidden, d_latent) * sc2 * 0.5
        self.b_logvar = np.zeros(d_latent) - 1.0  # init to small variance

        # Decoder
        self.W_dec1 = np.random.randn(d_latent, hidden) * sc3
        self.b_dec1 = np.zeros(hidden)
        self.W_dec2 = np.random.randn(hidden, n_dmn_params) * sc2 * 0.5
        self.b_dec2 = np.zeros(n_dmn_params)

    def encode(self, x):
        """x: (d_input,) -> mu, log_var: (d_latent,)"""
        h = np.maximum(x @ self.W_enc1 + self.b_enc1, 0)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Sample z = mu + sigma * epsilon, epsilon ~ N(0,I)."""
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        return mu + std * eps

    def decode(self, z):
        """z: (d_latent,) -> dmn_params: (n_dmn_params,)"""
        h = np.maximum(z @ self.W_dec1 + self.b_dec1, 0)
        params = h @ self.W_dec2 + self.b_dec2
        return params

    def split_dmn_params(self, raw_params, n_internal):
        """Split raw decoder output into weights (sigmoid) and thetas."""
        w_raw = raw_params[:n_internal]
        t_raw = raw_params[n_internal:2 * n_internal]
        weights = sigmoid(w_raw)  # [0, 1]
        thetas = t_raw * 0.1     # small angles
        return weights, thetas

    def forward(self, x, n_internal):
        """Full forward: encode -> sample -> decode -> DMN params."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        raw = self.decode(z)
        weights, thetas = self.split_dmn_params(raw, n_internal)
        return weights, thetas, mu, log_var, z

    def _get_all_params(self):
        return [self.W_enc1, self.b_enc1, self.W_mu, self.b_mu,
                self.W_logvar, self.b_logvar,
                self.W_dec1, self.b_dec1, self.W_dec2, self.b_dec2]


# ---------------------------------------------------------------------------
# 3. Synthetic data generation for 2-phase composite
# ---------------------------------------------------------------------------

def generate_composite_data(n_samples=200, n_strains=5):
    """
    Generate synthetic stress-strain data for a 2-phase composite
    with varying volume fraction and fiber orientation.

    Returns:
      micro_desc: (n_samples, 2) -- [volume_fraction, orientation_angle]
      strains:    (n_strains, 3) -- applied strain states
      stresses:   (n_samples, n_strains, 3) -- resulting stress
    """
    # Phase properties (fiber and matrix)
    E_fiber, nu_fiber = 200.0, 0.25   # GPa-like
    E_matrix, nu_matrix = 5.0, 0.35

    C_fiber = isotropic_stiffness_2d(E_fiber, nu_fiber)
    C_matrix = isotropic_stiffness_2d(E_matrix, nu_matrix)

    # Varying microstructure: volume fraction and orientation
    vf = np.random.uniform(0.15, 0.65, n_samples)
    orient = np.random.uniform(-np.pi / 6, np.pi / 6, n_samples)
    micro_desc = np.column_stack([vf, orient])

    # Strain states to probe
    all_strains = np.array([
        [0.01, 0.0, 0.0],     # uniaxial x
        [0.0, 0.01, 0.0],     # uniaxial y
        [0.005, 0.005, 0.0],  # biaxial
        [0.0, 0.0, 0.01],     # shear
        [0.01, -0.003, 0.005] # combined
    ])
    strains = all_strains[:n_strains]

    stresses = np.zeros((n_samples, n_strains, 3))

    for i in range(n_samples):
        # Effective stiffness via rotation + Voigt-Reuss
        T = rotation_matrix_2d(orient[i])
        C_f_rot = T @ C_fiber @ T.T
        C_eff = vf[i] * C_f_rot + (1.0 - vf[i]) * C_matrix
        # Add slight nonlinear correction for realism
        S_f = np.linalg.inv(C_f_rot + 1e-10 * np.eye(3))
        S_m = np.linalg.inv(C_matrix + 1e-10 * np.eye(3))
        S_reuss = vf[i] * S_f + (1.0 - vf[i]) * S_m
        C_reuss = np.linalg.inv(S_reuss + 1e-10 * np.eye(3))
        C_eff = 0.5 * (C_eff + C_reuss)  # Hill average

        for j in range(n_strains):
            stresses[i, j] = C_eff @ strains[j]

    # Add measurement noise
    noise_std = 0.02 * np.abs(stresses).mean()
    stresses += np.random.randn(*stresses.shape) * noise_std

    return micro_desc, strains, stresses, C_fiber, C_matrix


# ---------------------------------------------------------------------------
# 4. VDMN Training
# ---------------------------------------------------------------------------

def _compute_loss_single(vae, dmn, micro_desc_i, strains, stresses_i,
                         C_fiber, C_matrix, n_internal, beta):
    """Compute ELBO loss for a single sample (helper for finite-diff)."""
    w, th, mu, log_var, z = vae.forward(micro_desc_i, n_internal)
    recon = 0.0
    for j in range(strains.shape[0]):
        sp = dmn.predict_stress(C_fiber, C_matrix, w, th, strains[j])
        recon += np.mean((sp - stresses_i[j]) ** 2)
    recon /= strains.shape[0]
    kl = -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var))
    return recon + beta * kl, recon, kl


def train_vdmn(vae, dmn, micro_desc, strains, stresses,
               C_fiber, C_matrix,
               epochs=100, lr=5e-4, beta=0.01, batch_size=16):
    """
    Train VDMN by maximizing ELBO:
      L = E_q[log p(sigma|z)] - beta * KL(q||p)

    Uses stochastic coordinate-wise finite-difference gradients.
    Each epoch processes one mini-batch for speed in this reference demo.
    """
    n_samples = len(micro_desc)
    n_internal = dmn.n_internal

    print(f"Training VDMN (epochs={epochs}, lr={lr}, beta={beta})")

    for epoch in range(epochs):
        # Pick a small mini-batch (2 samples for speed)
        batch_idx = np.random.choice(n_samples, min(batch_size, 4), replace=False)
        bs = len(batch_idx)

        # Compute base loss
        base_loss = 0.0
        base_recon = 0.0
        base_kl = 0.0
        for bi in batch_idx:
            l, r, k = _compute_loss_single(
                vae, dmn, micro_desc[bi], strains, stresses[bi],
                C_fiber, C_matrix, n_internal, beta)
            base_loss += l
            base_recon += r
            base_kl += k
        base_loss /= bs
        base_recon /= bs
        base_kl /= bs

        # Stochastic finite-diff: perturb 2 random coords per param array
        eps = 1e-4
        all_params = vae._get_all_params()

        for param_arr in all_params:
            n_p = min(2, param_arr.size)
            flat_idxs = np.random.choice(param_arr.size, n_p, replace=False)

            for fi in flat_idxs:
                idx = np.unravel_index(fi, param_arr.shape)
                old_val = param_arr[idx]
                param_arr[idx] = old_val + eps

                lp = 0.0
                for bi in batch_idx:
                    l, _, _ = _compute_loss_single(
                        vae, dmn, micro_desc[bi], strains, stresses[bi],
                        C_fiber, C_matrix, n_internal, beta)
                    lp += l
                lp /= bs

                grad = (lp - base_loss) / eps
                param_arr[idx] = old_val - lr * grad

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  "
                  f"recon={base_recon:.6f}  KL={base_kl:.4f}  "
                  f"ELBO={-base_loss:.6f}")

    return vae


# ---------------------------------------------------------------------------
# 5. Forward Uncertainty Propagation
# ---------------------------------------------------------------------------

def forward_uq(vae, dmn, micro_desc_sample, strains, C_fiber, C_matrix,
               n_mc=100):
    """
    Propagate microstructure uncertainty to stress response via Monte Carlo.
    """
    n_internal = dmn.n_internal
    n_strains = strains.shape[0]
    stress_samples = np.zeros((n_mc, n_strains, 3))

    mu, log_var = vae.encode(micro_desc_sample)

    for k in range(n_mc):
        z = vae.reparameterize(mu, log_var)
        raw = vae.decode(z)
        w, th = vae.split_dmn_params(raw, n_internal)

        for j in range(n_strains):
            stress_samples[k, j] = dmn.predict_stress(
                C_fiber, C_matrix, w, th, strains[j]
            )

    stress_mean = stress_samples.mean(axis=0)
    stress_std = stress_samples.std(axis=0)

    return stress_mean, stress_std, stress_samples


# ---------------------------------------------------------------------------
# 6. Inverse Problem: infer microstructure from observed response
# ---------------------------------------------------------------------------

def inverse_problem(vae, dmn, sigma_obs, strains, C_fiber, C_matrix,
                    n_candidates=200, n_internal=7):
    """
    Given observed stress responses sigma_obs (n_strains, 3),
    sample latent codes and rank by likelihood to find microstructure
    candidates.
    """
    # Sample from prior p(z) = N(0, I)
    d_latent = vae.d_latent
    z_candidates = np.random.randn(n_candidates, d_latent)

    likelihoods = np.zeros(n_candidates)

    for k in range(n_candidates):
        raw = vae.decode(z_candidates[k])
        w, th = vae.split_dmn_params(raw, n_internal)

        residual = 0.0
        for j in range(strains.shape[0]):
            sp = dmn.predict_stress(C_fiber, C_matrix, w, th, strains[j])
            residual += np.sum((sp - sigma_obs[j]) ** 2)

        likelihoods[k] = -residual  # log-likelihood (unnormalized)

    # Top candidates
    top_idx = np.argsort(likelihoods)[-10:][::-1]

    # Decode top candidates to DMN params and estimate microstructure
    results = []
    for idx in top_idx:
        raw = vae.decode(z_candidates[idx])
        w, th = vae.split_dmn_params(raw, n_internal)
        # Effective volume fraction estimate from weights
        vf_est = np.mean(w)
        theta_est = np.mean(np.abs(th))
        results.append({
            'z': z_candidates[idx],
            'vf_est': vf_est,
            'theta_est': theta_est,
            'log_lik': likelihoods[idx]
        })

    return results


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    print("=" * 65)
    print("VDMN Reference Implementation")
    print("2-phase composite: fiber-reinforced polymer")
    print("=" * 65)

    # --- Generate data ---
    print("\n[1] Generating synthetic composite dataset...")
    n_train = 80
    micro_desc, strains, stresses, C_fiber, C_matrix = \
        generate_composite_data(n_train, n_strains=3)
    print(f"  {n_train} samples, microstructure descriptors: "
          f"[volume_fraction, orientation]")
    print(f"  Volume fraction range: [{micro_desc[:,0].min():.2f}, "
          f"{micro_desc[:,0].max():.2f}]")
    print(f"  Orientation range: [{micro_desc[:,1].min():.3f}, "
          f"{micro_desc[:,1].max():.3f}] rad")

    # --- Setup DMN and VAE ---
    n_levels = 3
    dmn = DeepMaterialNetwork(n_levels=n_levels)
    n_internal = dmn.n_internal  # 7 for 3 levels
    n_dmn_params = 2 * n_internal  # weights + thetas

    d_input = 2   # (vf, orientation)
    d_latent = 4
    vae = VariationalMLP(d_input, d_latent, n_dmn_params, hidden=24)

    print(f"\n  DMN: {n_levels} levels, {dmn.n_leaves} leaves, "
          f"{n_internal} internal nodes")
    print(f"  VAE: d_input={d_input}, d_latent={d_latent}, "
          f"n_dmn_params={n_dmn_params}")

    # --- Train ---
    print("\n[2] Training VDMN...")
    vae = train_vdmn(vae, dmn, micro_desc, strains, stresses,
                     C_fiber, C_matrix,
                     epochs=60, lr=5e-4, beta=0.005, batch_size=4)

    # --- Forward UQ ---
    print("\n[3] Forward Uncertainty Propagation...")
    test_micro = np.array([0.40, 0.05])  # vf=0.40, orient=0.05 rad
    print(f"  Test microstructure: vf={test_micro[0]:.2f}, "
          f"orient={test_micro[1]:.3f} rad")

    stress_mean, stress_std, stress_samples = forward_uq(
        vae, dmn, test_micro, strains, C_fiber, C_matrix, n_mc=200
    )

    all_strain_labels = ["uniaxial-x", "uniaxial-y", "biaxial",
                         "shear", "combined"]
    strain_labels = all_strain_labels[:strains.shape[0]]
    print(f"\n  {'Strain state':<14s} {'sigma_11 mean':>14s} {'+-std':>10s} "
          f"{'sigma_22 mean':>14s} {'+-std':>10s}")
    print("  " + "-" * 62)
    for j in range(len(strain_labels)):
        print(f"  {strain_labels[j]:<14s} "
              f"{stress_mean[j,0]:14.4f} {stress_std[j,0]:10.4f} "
              f"{stress_mean[j,1]:14.4f} {stress_std[j,1]:10.4f}")

    # Coefficient of variation
    cv = np.mean(stress_std / (np.abs(stress_mean) + 1e-10)) * 100
    print(f"\n  Mean coefficient of variation: {cv:.1f}%")

    # --- Inverse Problem ---
    print("\n[4] Inverse Problem: infer microstructure from stress response...")

    # Generate observation from a known microstructure
    true_vf, true_orient = 0.35, 0.08
    print(f"  True microstructure: vf={true_vf:.2f}, orient={true_orient:.3f} rad")

    T_true = rotation_matrix_2d(true_orient)
    C_f_rot = T_true @ C_fiber @ T_true.T
    C_eff_true = true_vf * C_f_rot + (1.0 - true_vf) * C_matrix
    S_f_true = np.linalg.inv(C_f_rot + 1e-10 * np.eye(3))
    S_m_true = np.linalg.inv(C_matrix + 1e-10 * np.eye(3))
    S_r = true_vf * S_f_true + (1.0 - true_vf) * S_m_true
    C_eff_true = 0.5 * (C_eff_true + np.linalg.inv(S_r + 1e-10 * np.eye(3)))

    sigma_obs = np.zeros((strains.shape[0], 3))
    for j in range(strains.shape[0]):
        sigma_obs[j] = C_eff_true @ strains[j]
        sigma_obs[j] += np.random.randn(3) * 0.01  # noise

    results = inverse_problem(vae, dmn, sigma_obs, strains,
                              C_fiber, C_matrix,
                              n_candidates=500, n_internal=n_internal)

    print(f"\n  Top 5 microstructure candidates:")
    print(f"  {'Rank':>4s} {'vf_est':>8s} {'theta_est':>10s} {'log_lik':>10s}")
    print("  " + "-" * 36)
    vf_estimates = []
    for rank, r in enumerate(results[:5]):
        print(f"  {rank+1:4d} {r['vf_est']:8.3f} {r['theta_est']:10.4f} "
              f"{r['log_lik']:10.4f}")
        vf_estimates.append(r['vf_est'])

    vf_mean_est = np.mean(vf_estimates)
    vf_std_est = np.std(vf_estimates)
    print(f"\n  Estimated vf (top-5 mean): {vf_mean_est:.3f} +- {vf_std_est:.3f} "
          f"(true: {true_vf:.2f})")
    print(f"  Note: multiple solutions reflect the one-to-many nature "
          f"of the inverse problem")

    # --- Summary ---
    print("\n" + "=" * 65)
    print("Summary:")
    print("  - DMN provides physics-based hierarchical homogenization")
    print("  - Variational encoding captures microstructure uncertainty")
    print("  - Forward UQ propagates process variation to stress response")
    print("  - Inverse problem yields probabilistic microstructure estimates")
    print("=" * 65)


if __name__ == "__main__":
    main()
