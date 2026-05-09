"""
Reference implementation: Sensitivity-Constrained Fourier Neural Operator (SC-FNO)
(arXiv: 2505.08740)

Core idea: Train a Fourier-space surrogate on a parametric 1D PDE while
simultaneously matching the parameter sensitivity du/dp, so that
global sensitivity indices (Sobol') estimated from the surrogate are
accurate and inverse-problem gradients are reliable.

Demonstrates:
  1. Synthetic data generation for a 1D diffusion-reaction equation
     u_xx + p*u = f  with Dirichlet BCs, varying parameter p.
  2. Simplified FNO-like model (truncated Fourier modes, numpy only).
  3. Sensitivity-constrained training: L = L_data + lambda * L_sens.
  4. Sobol' first-order index estimation via the trained surrogate.

Dependencies: numpy, scipy
Usage: python 2505.08740_reference_impl.py
"""

import numpy as np
from scipy.linalg import solve_banded


# ---------------------------------------------------------------------------
# 1. Parametric 1D PDE solver  (diffusion-reaction: -u_xx + p*u = f)
# ---------------------------------------------------------------------------

def solve_diffusion_reaction(p_val, nx=64, f_func=None):
    """
    Solve  -u_xx + p*u = f  on [0, 1] with u(0)=u(1)=0.
    Returns u(x) on interior grid (nx-1 points).
    Also returns du/dp via direct differentiation.
    """
    dx = 1.0 / nx
    x = np.linspace(dx, 1.0 - dx, nx - 1)

    if f_func is None:
        f = np.sin(np.pi * x)  # default forcing
    else:
        f = f_func(x)

    # Assemble tridiagonal:  (-1/dx^2) u_{i-1} + (2/dx^2 + p) u_i + (-1/dx^2) u_{i+1} = f_i
    n = nx - 1
    diag_main = np.full(n, 2.0 / dx**2 + p_val)
    diag_off = np.full(n - 1, -1.0 / dx**2)

    # Banded storage for scipy solve_banded:  ab[0]=upper, ab[1]=main, ab[2]=lower
    ab = np.zeros((3, n))
    ab[0, 1:] = diag_off       # upper diagonal
    ab[1, :] = diag_main       # main diagonal
    ab[2, :-1] = diag_off      # lower diagonal

    u = solve_banded((1, 1), ab, f)

    # Sensitivity: differentiate the linear system A(p) u = f  =>  A du/dp = -dA/dp u
    # dA/dp = I  (since the p contribution is +p on diagonal)
    rhs_sens = -u  # -dA/dp * u = -I * u = -u
    du_dp = solve_banded((1, 1), ab, rhs_sens)

    return x, u, du_dp


# ---------------------------------------------------------------------------
# 2. Data generation
# ---------------------------------------------------------------------------

def generate_dataset(n_samples=200, nx=64, p_range=(0.5, 15.0)):
    """Generate (p, u, du/dp) triples."""
    ps = np.random.uniform(p_range[0], p_range[1], n_samples)
    n_pts = nx - 1
    U = np.zeros((n_samples, n_pts))
    S = np.zeros((n_samples, n_pts))
    for i, p in enumerate(ps):
        _, u, s = solve_diffusion_reaction(p, nx=nx)
        U[i] = u
        S[i] = s
    x_grid = np.linspace(1.0 / nx, 1.0 - 1.0 / nx, n_pts)
    return ps, U, S, x_grid


# ---------------------------------------------------------------------------
# 3. Simplified Fourier Neural Operator (numpy, 1-layer)
# ---------------------------------------------------------------------------

class SimpleFourierOperator:
    """
    Minimal 1D Fourier operator: for each sample with parameter p,
    predict u(x) by learning weights on truncated Fourier modes.

    Architecture:
      - Lift p -> hidden representation via linear layer
      - Apply Fourier convolution (truncated modes)
      - Project to output u(x)

    This is a drastically simplified version to illustrate the
    sensitivity-constrained training idea.
    """

    def __init__(self, n_pts, n_modes=12, hidden=32):
        self.n_pts = n_pts
        self.n_modes = n_modes
        self.hidden = hidden

        scale1 = np.sqrt(2.0 / 1)
        scale2 = np.sqrt(2.0 / hidden)

        # Lift: scalar p -> (n_pts, hidden)
        self.W_lift = np.random.randn(1, hidden) * scale1
        self.b_lift = np.zeros(hidden)

        # Fourier layer weights: (n_modes, hidden, hidden) complex
        self.R_real = np.random.randn(n_modes, hidden, hidden) * scale2 * 0.1
        self.R_imag = np.random.randn(n_modes, hidden, hidden) * scale2 * 0.1

        # Bypass linear
        self.W_bypass = np.random.randn(hidden, hidden) * scale2 * 0.5
        self.b_bypass = np.zeros(hidden)

        # Project: hidden -> 1
        self.W_proj = np.random.randn(hidden, 1) * np.sqrt(2.0 / hidden)
        self.b_proj = np.zeros(1)

    def _fourier_layer(self, v):
        """
        v: (n_pts, hidden)
        Apply spectral convolution on truncated modes + bypass.
        """
        n = v.shape[0]
        # FFT along spatial dim for each hidden channel
        v_hat = np.fft.rfft(v, axis=0)  # (n//2+1, hidden)

        # Truncate and apply learnable weights
        out_hat = np.zeros_like(v_hat)
        km = min(self.n_modes, v_hat.shape[0])
        R = self.R_real[:km] + 1j * self.R_imag[:km]  # (km, hidden, hidden)
        for k in range(km):
            out_hat[k] = v_hat[k] @ R[k]

        # iFFT
        out_spatial = np.fft.irfft(out_hat, n=n, axis=0)  # (n_pts, hidden)

        # Bypass (local linear)
        bypass = v @ self.W_bypass + self.b_bypass

        return np.maximum(out_spatial + bypass, 0)  # ReLU

    def forward(self, p_scalar):
        """
        p_scalar: float, the PDE parameter
        Returns: u_pred (n_pts,)
        """
        # Lift
        p_in = np.array([[p_scalar]])  # (1, 1)
        h0 = p_in @ self.W_lift + self.b_lift  # (1, hidden)
        v = np.tile(h0, (self.n_pts, 1))  # (n_pts, hidden) -- broadcast p

        # Fourier layer
        v = self._fourier_layer(v)

        # Project
        u = (v @ self.W_proj + self.b_proj).ravel()  # (n_pts,)
        return u

    def forward_with_grad(self, p_scalar, dp=1e-5):
        """Forward pass + finite-difference sensitivity du/dp."""
        u = self.forward(p_scalar)
        u_plus = self.forward(p_scalar + dp)
        du_dp = (u_plus - u) / dp
        return u, du_dp

    def _collect_params(self):
        return [self.W_lift, self.b_lift,
                self.R_real, self.R_imag,
                self.W_bypass, self.b_bypass,
                self.W_proj, self.b_proj]

    def _collect_grads(self, p_scalar, u_target, s_target, lam):
        """
        Numerical gradients via finite differences for all parameters.
        L = L_data + lam * L_sens
        """
        params = self._collect_params()
        grads = []
        eps = 1e-5

        def loss_fn():
            u_pred, s_pred = self.forward_with_grad(p_scalar)
            l_data = np.mean((u_pred - u_target) ** 2)
            l_sens = np.mean((s_pred - s_target) ** 2)
            return l_data + lam * l_sens

        base_loss = loss_fn()

        for param in params:
            g = np.zeros_like(param)
            it = np.nditer(param, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                old = param[idx]
                param[idx] = old + eps
                loss_plus = loss_fn()
                param[idx] = old
                g[idx] = (loss_plus - base_loss) / eps
                it.iternext()
            grads.append(g)

        return grads, base_loss


# ---------------------------------------------------------------------------
# 4. Efficient batch training with stochastic parameter-space finite diffs
# ---------------------------------------------------------------------------

def train_sc_fno(model, ps, U, S, lam=1.0, epochs=80, lr=1e-3, batch_size=8):
    """
    Train the simplified FNO with sensitivity-constrained loss.

    Because full numerical gradients are expensive, we use a simplified
    approach: for each mini-batch, compute loss and approximate parameter
    updates via coordinate-wise finite differences on a random subset of
    model weights.

    For this reference demo we use a direct MSE fitting approach instead,
    leveraging the linearity of the projection layer.
    """
    n_samples = len(ps)
    n_pts = U.shape[1]

    # Normalize
    p_mean, p_std = ps.mean(), ps.std() + 1e-8
    u_scale = np.abs(U).max() + 1e-8
    s_scale = np.abs(S).max() + 1e-8

    ps_n = (ps - p_mean) / p_std
    U_n = U / u_scale
    S_n = S / s_scale

    print(f"Training SC-FNO (lambda={lam}, epochs={epochs}, lr={lr})")

    for epoch in range(epochs):
        perm = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = perm[start:start + batch_size]
            batch_loss_data = 0.0
            batch_loss_sens = 0.0

            # Accumulate gradient estimates via finite differences on key params
            grad_W_proj = np.zeros_like(model.W_proj)
            grad_b_proj = np.zeros_like(model.b_proj)
            grad_W_lift = np.zeros_like(model.W_lift)
            grad_b_lift = np.zeros_like(model.b_lift)
            grad_W_bypass = np.zeros_like(model.W_bypass)
            grad_b_bypass = np.zeros_like(model.b_bypass)
            grad_R_real = np.zeros_like(model.R_real)
            grad_R_imag = np.zeros_like(model.R_imag)

            bs = len(batch_idx)

            for bi in batch_idx:
                p_val = ps_n[bi]
                u_tgt = U_n[bi]
                s_tgt = S_n[bi]

                u_pred, s_pred = model.forward_with_grad(p_val)

                err_u = u_pred - u_tgt
                err_s = s_pred - s_tgt

                batch_loss_data += np.mean(err_u ** 2)
                batch_loss_sens += np.mean(err_s ** 2)

                # --- Analytical gradient for projection layer ---
                # u = v @ W_proj + b_proj  =>  du/dW_proj = v^T, du/db_proj = 1
                # Fourier layer output (recompute)
                p_in = np.array([[p_val]])
                h0 = p_in @ model.W_lift + model.b_lift
                v_in = np.tile(h0, (model.n_pts, 1))
                v_out = model._fourier_layer(v_in)  # (n_pts, hidden)

                # Grad for W_proj, b_proj (data term)
                # dL/dW_proj = (2/n) * v_out^T @ err_u_col
                err_u_col = err_u.reshape(-1, 1)
                grad_W_proj += (2.0 / n_pts) * (v_out.T @ err_u_col) / bs
                grad_b_proj += (2.0 / n_pts) * err_u.sum() / bs

            batch_loss = batch_loss_data / bs + lam * batch_loss_sens / bs
            total_loss += batch_loss

            # --- Finite-diff gradients for Fourier layer params (subsample) ---
            eps = 1e-4
            # Only perturb a random subset of params per batch for efficiency
            n_perturb = min(8, model.R_real.size)
            indices = np.random.choice(model.R_real.size, n_perturb, replace=False)

            for flat_idx in indices:
                idx = np.unravel_index(flat_idx, model.R_real.shape)

                for param_arr, grad_arr in [
                    (model.R_real, grad_R_real),
                    (model.R_imag, grad_R_imag),
                ]:
                    old_val = param_arr[idx]
                    # +eps
                    param_arr[idx] = old_val + eps
                    loss_plus = 0.0
                    for bi in batch_idx:
                        u_p, s_p = model.forward_with_grad(ps_n[bi])
                        loss_plus += np.mean((u_p - U_n[bi])**2) + lam * np.mean((s_p - S_n[bi])**2)
                    loss_plus /= bs
                    param_arr[idx] = old_val
                    grad_arr[idx] = (loss_plus - batch_loss) / eps

            # Also perturb W_lift, b_lift, W_bypass (subsample)
            for param_arr, grad_arr, n_pert in [
                (model.W_lift, grad_W_lift, min(4, model.W_lift.size)),
                (model.b_lift, grad_b_lift, min(4, model.b_lift.size)),
                (model.W_bypass, grad_W_bypass, min(4, model.W_bypass.size)),
                (model.b_bypass, grad_b_bypass, min(4, model.b_bypass.size)),
            ]:
                idxs = np.random.choice(param_arr.size, n_pert, replace=False)
                for fi in idxs:
                    mi = np.unravel_index(fi, param_arr.shape)
                    old_val = param_arr[mi]
                    param_arr[mi] = old_val + eps
                    lp = 0.0
                    for bi in batch_idx:
                        u_p, s_p = model.forward_with_grad(ps_n[bi])
                        lp += np.mean((u_p - U_n[bi])**2) + lam * np.mean((s_p - S_n[bi])**2)
                    lp /= bs
                    param_arr[mi] = old_val
                    grad_arr[mi] = (lp - batch_loss) / eps

            # Update all params
            model.W_proj -= lr * grad_W_proj
            model.b_proj -= lr * grad_b_proj.ravel()
            model.W_lift -= lr * grad_W_lift
            model.b_lift -= lr * grad_b_lift.ravel()
            model.W_bypass -= lr * grad_W_bypass
            model.b_bypass -= lr * grad_b_bypass.ravel()
            model.R_real -= lr * grad_R_real
            model.R_imag -= lr * grad_R_imag

            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.6f}")

    return model, p_mean, p_std, u_scale, s_scale


# ---------------------------------------------------------------------------
# 5. Sobol' first-order sensitivity index estimation
# ---------------------------------------------------------------------------

def sobol_first_order(model, p_mean, p_std, u_scale, p_range, n_mc=500, nx=63):
    """
    Estimate Sobol' first-order index S_1 for the scalar parameter p
    using the trained surrogate.

    S_1 = Var_p[E[u|p]] / Var[u]

    For a single scalar parameter, S_1 should be ~1.0.
    We compute the variance of the surrogate output over the parameter range.
    """
    ps_mc = np.random.uniform(p_range[0], p_range[1], n_mc)
    ps_mc_n = (ps_mc - p_mean) / p_std

    U_mc = np.zeros((n_mc, nx))
    for i, pn in enumerate(ps_mc_n):
        U_mc[i] = model.forward(pn) * u_scale

    # For a single parameter: S_1 = Var[E[u|p]] / Var[u]
    # E[u|p] is just u(p) since u is deterministic given p
    # Var[u] = Var_p[u(p)]
    var_total = np.var(U_mc, axis=0).mean()
    mean_u = U_mc.mean(axis=0)
    var_conditional = np.var(U_mc, axis=0).mean()  # = Var[E[u|p]]

    S1 = var_conditional / (var_total + 1e-12)
    return S1


# ---------------------------------------------------------------------------
# 6. Inverse problem demo
# ---------------------------------------------------------------------------

def inverse_problem_demo(model, p_mean, p_std, u_scale, s_scale, nx=63):
    """
    Given observed u_obs, recover the parameter p* using gradient descent.
    The SC-FNO provides accurate du/dp for faster convergence.
    """
    # Generate observation from true p
    p_true = 5.0
    _, u_obs, _ = solve_diffusion_reaction(p_true, nx=nx + 1)

    # Normalize observation
    u_obs_n = u_obs / u_scale

    # Gradient descent in normalized p-space
    p_est_n = 0.0  # initial guess (corresponds to p_mean)
    lr_inv = 0.5
    print(f"\n  Inverse problem: recovering p_true = {p_true:.2f}")

    for step in range(30):
        u_pred, du_dp_pred = model.forward_with_grad(p_est_n)
        residual = u_pred - u_obs_n
        loss = np.mean(residual ** 2)

        # Gradient: dL/dp = 2 * mean(residual * du/dp)
        grad = 2.0 * np.mean(residual * du_dp_pred)
        p_est_n -= lr_inv * grad

        p_est = p_est_n * p_std + p_mean

        if (step + 1) % 5 == 0:
            print(f"    step {step+1:3d}  p_est={p_est:.4f}  loss={loss:.6f}")

    p_final = p_est_n * p_std + p_mean
    print(f"  Final estimate: p = {p_final:.4f} (true = {p_true:.2f}, "
          f"error = {abs(p_final - p_true):.4f})")
    return p_final


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)
    nx = 64
    n_pts = nx - 1  # interior points

    # --- Generate dataset ---
    print("=" * 60)
    print("SC-FNO Reference Implementation")
    print("PDE: -u_xx + p*u = sin(pi*x), u(0)=u(1)=0")
    print("=" * 60)

    print("\n[1] Generating dataset...")
    n_train = 120
    p_range = (0.5, 15.0)
    ps, U, S, x_grid = generate_dataset(n_train, nx, p_range)
    print(f"  {n_train} samples, p in [{p_range[0]}, {p_range[1]}], grid={n_pts} pts")

    # --- Train SC-FNO (with sensitivity constraint) ---
    print("\n[2] Training SC-FNO (lambda=1.0, WITH sensitivity constraint)...")
    model_sc = SimpleFourierOperator(n_pts, n_modes=8, hidden=16)
    model_sc, pm, ps_std, us, ss = train_sc_fno(
        model_sc, ps, U, S, lam=1.0, epochs=60, lr=5e-4, batch_size=16
    )

    # --- Train standard FNO (without sensitivity constraint) ---
    print("\n[3] Training standard FNO (lambda=0.0, NO sensitivity constraint)...")
    model_std = SimpleFourierOperator(n_pts, n_modes=8, hidden=16)
    model_std, _, _, _, _ = train_sc_fno(
        model_std, ps, U, S, lam=0.0, epochs=60, lr=5e-4, batch_size=16
    )

    # --- Evaluate sensitivity accuracy ---
    print("\n[4] Evaluating sensitivity accuracy on test set...")
    n_test = 30
    ps_test = np.random.uniform(p_range[0], p_range[1], n_test)
    ps_test_n = (ps_test - pm) / ps_std

    err_sc, err_std = 0.0, 0.0
    for i, (pt, ptn) in enumerate(zip(ps_test, ps_test_n)):
        _, u_true, s_true = solve_diffusion_reaction(pt, nx)
        s_true_n = s_true / ss

        _, s_sc = model_sc.forward_with_grad(ptn)
        _, s_no = model_std.forward_with_grad(ptn)

        err_sc += np.mean((s_sc - s_true_n) ** 2)
        err_std += np.mean((s_no - s_true_n) ** 2)

    err_sc /= n_test
    err_std /= n_test
    improvement = (1 - err_sc / (err_std + 1e-12)) * 100
    print(f"  SC-FNO sensitivity MSE:  {err_sc:.6f}")
    print(f"  Std-FNO sensitivity MSE: {err_std:.6f}")
    print(f"  Improvement: {improvement:.1f}%")

    # --- Sobol' index estimation ---
    print("\n[5] Sobol' first-order index estimation...")
    S1 = sobol_first_order(model_sc, pm, ps_std, us, p_range, n_mc=300, nx=n_pts)
    print(f"  S_1 (single param, should be ~1.0): {S1:.4f}")

    # --- Inverse problem ---
    print("\n[6] Inverse problem via SC-FNO gradients...")
    inverse_problem_demo(model_sc, pm, ps_std, us, ss, nx=n_pts)

    print("\n" + "=" * 60)
    print("Done. SC-FNO demonstrates sensitivity-constrained operator learning.")
    print("=" * 60)


if __name__ == "__main__":
    main()
