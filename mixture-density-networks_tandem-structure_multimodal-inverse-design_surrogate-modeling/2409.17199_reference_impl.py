"""
Reference implementation: Tandem Network + MDN for Optical Multilayer Thin Film Inverse Design
Based on: arXiv 2409.17199 — "Optical Multilayer Thin Film Structure Inverse Design:
From Optimization to Deep Learning" (Ma, Ma, Guo, 2024)

Demonstrates:
  1. Synthetic forward model (simplified thin-film reflectance)
  2. MLP surrogate forward model with manual backpropagation
  3. Tandem network: inverse model + frozen forward model
  4. Mixture Density Network (MDN): multimodal conditional p(d|R)

All implemented with numpy/scipy only. Run: python reference_impl.py
"""

import numpy as np

np.random.seed(42)


# ============================================================================
# 1. Synthetic Forward Model (simplified thin-film reflectance)
# ============================================================================

def thin_film_reflectance(thicknesses, n_layers, wavelengths):
    """
    Simplified reflectance for a multilayer thin film stack.
    Uses Fabry-Perot interference per layer.
    """
    r_total = np.zeros(len(wavelengths), dtype=float)
    for d, n in zip(thicknesses, n_layers):
        phase = 2.0 * np.pi * n * d / wavelengths
        r_total += 0.1 * (n - 1.0) ** 2 * np.sin(phase) ** 2
    return np.clip(r_total, 0.0, 1.0)


def thin_film_batch(thicknesses_batch, n_layers, wavelengths):
    """Vectorized batch version."""
    return np.array([thin_film_reflectance(t, n_layers, wavelengths)
                     for t in thicknesses_batch])


# ============================================================================
# 2. MLP with Analytical Backpropagation (numpy-only)
# ============================================================================

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def softplus(x):
    return np.log1p(np.exp(np.clip(x, -20, 20)))

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


class MLP:
    """MLP with manual backpropagation and SGD/Adam training."""

    def __init__(self, layer_sizes, seed=0):
        rng = np.random.RandomState(seed)
        self.weights = []
        self.biases = []
        self.layer_sizes = layer_sizes
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            w = rng.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)
        self._init_adam()

    def _init_adam(self):
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

    def forward(self, x, store=False):
        """Forward pass. If store=True, cache activations for backprop."""
        if store:
            self.pre_acts = []
            self.acts = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = x @ w + b
            if store:
                self.pre_acts.append(z)
            if i < len(self.weights) - 1:
                x = relu(z)
            else:
                x = z  # linear output
            if store:
                self.acts.append(x)
        return x

    def backward(self, d_output):
        """Backward pass given gradient of loss w.r.t. output.
        Returns gradient w.r.t. input x."""
        self.grad_w = []
        self.grad_b = []
        N = d_output.shape[0]
        delta = d_output

        for i in reversed(range(len(self.weights))):
            self.grad_w.insert(0, self.acts[i].T @ delta / N)
            self.grad_b.insert(0, np.mean(delta, axis=0))
            delta = delta @ self.weights[i].T
            if i > 0:
                delta = delta * relu_grad(self.pre_acts[i - 1])

        return delta  # gradient w.r.t. input x

    def adam_step(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.weights)):
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * self.grad_w[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * self.grad_w[i] ** 2
            m_hat = self.m_w[i] / (1 - beta1 ** self.t)
            v_hat = self.v_w[i] / (1 - beta2 ** self.t)
            self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * self.grad_b[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * self.grad_b[i] ** 2
            m_hat = self.m_b[i] / (1 - beta1 ** self.t)
            v_hat = self.v_b[i] / (1 - beta2 ** self.t)
            self.biases[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_params_from(self, other):
        for i in range(len(self.weights)):
            self.weights[i] = other.weights[i].copy()
            self.biases[i] = other.biases[i].copy()


# ============================================================================
# 3. Generate Synthetic Dataset
# ============================================================================

NUM_LAYERS = 3
REFRACTIVE_INDICES = np.array([1.5, 2.0, 1.8])
WAVELENGTHS = np.linspace(400, 800, 20)  # 20 wavelength points
D_MIN, D_MAX = 20.0, 200.0

print("=" * 70)
print("Generating synthetic thin-film dataset...")
N_TRAIN, N_TEST = 1000, 200

d_train = np.random.uniform(D_MIN, D_MAX, (N_TRAIN, NUM_LAYERS))
r_train = thin_film_batch(d_train, REFRACTIVE_INDICES, WAVELENGTHS)
d_test = np.random.uniform(D_MIN, D_MAX, (N_TEST, NUM_LAYERS))
r_test = thin_film_batch(d_test, REFRACTIVE_INDICES, WAVELENGTHS)

# Normalize thicknesses to [0, 1]
d_train_n = (d_train - D_MIN) / (D_MAX - D_MIN)
d_test_n = (d_test - D_MIN) / (D_MAX - D_MIN)

print(f"  Train: {N_TRAIN}, Test: {N_TEST}")
print(f"  Layers: {NUM_LAYERS}, Wavelengths: {len(WAVELENGTHS)}")


# ============================================================================
# 4. Train Forward Surrogate Model
# ============================================================================

print("\n" + "=" * 70)
print("Step 1: Training Forward Surrogate Model f(d) -> R(lambda)...")

fwd = MLP([NUM_LAYERS, 32, 32, len(WAVELENGTHS)], seed=1)

EPOCHS_FWD = 300
BS = 128
for epoch in range(EPOCHS_FWD):
    idx = np.random.choice(N_TRAIN, BS, replace=False)
    x_b, y_b = d_train_n[idx], r_train[idx]
    pred = fwd.forward(x_b, store=True)
    d_loss = 2.0 * (pred - y_b) / BS  # grad of MSE
    fwd.backward(d_loss)
    fwd.adam_step(lr=1e-3)

r_pred = fwd.forward(d_test_n)
fwd_mse = np.mean((r_pred - r_test) ** 2)
print(f"  Forward model test MSE: {fwd_mse:.6f}")


# ============================================================================
# 5. Tandem Network: Inverse Model + Frozen Forward Model
# ============================================================================

print("\n" + "=" * 70)
print("Step 2: Training Tandem Network (Inverse + Frozen Forward)...")
print("  Architecture: R -> InverseNet -> d_pred -> FrozenForward -> R_recon")
print("  Loss: ||R_recon - R_target||^2 (spectrum reconstruction)")

inv = MLP([len(WAVELENGTHS), 64, 32, NUM_LAYERS], seed=2)

# Frozen copy of forward model
fwd_frozen = MLP([NUM_LAYERS, 32, 32, len(WAVELENGTHS)], seed=1)
fwd_frozen.copy_params_from(fwd)

EPOCHS_TANDEM = 500
for epoch in range(EPOCHS_TANDEM):
    idx = np.random.choice(N_TRAIN, BS, replace=False)
    r_b = r_train[idx]

    # Forward through inverse model
    d_pred = inv.forward(r_b, store=True)
    d_pred_clip = np.clip(d_pred, 0, 1)

    # Forward through frozen forward model
    r_recon = fwd_frozen.forward(d_pred_clip, store=True)

    # Reconstruction loss gradient
    d_recon = 2.0 * (r_recon - r_b) / BS

    # Backprop through frozen forward model to get gradient w.r.t. d_pred
    d_d = fwd_frozen.backward(d_recon)

    # Mask gradient where clipping occurred
    d_d = d_d * ((d_pred >= 0) & (d_pred <= 1)).astype(float)

    # Backprop through inverse model
    inv.backward(d_d)
    inv.adam_step(lr=1e-3)

# Evaluate
d_pred_test = np.clip(inv.forward(r_test), 0, 1)
r_recon_test = fwd_frozen.forward(d_pred_test)
tandem_mse = np.mean((r_recon_test - r_test) ** 2)
d_denorm = d_pred_test * (D_MAX - D_MIN) + D_MIN
tandem_d_mse = np.mean((d_denorm - d_test) ** 2)
print(f"  Tandem spectrum reconstruction MSE: {tandem_mse:.6f}")
print(f"  Tandem thickness prediction MSE:    {tandem_d_mse:.2f} nm^2")


# ============================================================================
# 6. Mixture Density Network (MDN) for Multimodal Inverse Design
# ============================================================================

print("\n" + "=" * 70)
print("Step 3: Training Mixture Density Network p(d|R)...")
print("  p(d|R) = sum_k pi_k(R) * N(d; mu_k(R), sigma_k^2(R))")

K = 5  # mixture components
D = NUM_LAYERS
W = len(WAVELENGTHS)
OUT_DIM = K + K * D + K * D  # pi, mu, sigma

mdn = MLP([W, 64, 64, OUT_DIM], seed=3)


def mdn_forward(model, x, store=False):
    """Parse MDN output into (pi, mu, sigma)."""
    raw = model.forward(x, store=store)
    N = x.shape[0]
    pi = softmax(raw[:, :K])
    mu = raw[:, K:K + K * D].reshape(N, K, D)
    sigma = softplus(raw[:, K + K * D:].reshape(N, K, D)) + 1e-4
    return pi, mu, sigma


def mdn_nll(pi, mu, sigma, y):
    """NLL loss and gradient of raw output."""
    N = y.shape[0]
    y_exp = y[:, np.newaxis, :]  # (N, 1, D)

    # Log Gaussian per component
    log_gauss = (-0.5 * np.sum(((y_exp - mu) / sigma) ** 2, axis=2)
                 - np.sum(np.log(sigma), axis=2)
                 - 0.5 * D * np.log(2 * np.pi))  # (N, K)

    log_pi = np.log(pi + 1e-10)
    log_mix = log_pi + log_gauss  # (N, K)
    max_lm = np.max(log_mix, axis=1, keepdims=True)
    log_sum = max_lm + np.log(np.sum(np.exp(log_mix - max_lm), axis=1, keepdims=True))
    nll = -np.mean(log_sum)

    # Responsibilities: gamma_k = pi_k * N_k / sum(...)
    gamma = np.exp(log_mix - log_sum)  # (N, K)

    # Gradients w.r.t. raw outputs
    # d(NLL)/d(pi_logits) => gamma - pi
    d_pi_logits = (pi - gamma) / N  # (N, K)

    # d(NLL)/d(mu) => gamma_k * (mu_k - y) / sigma_k^2
    d_mu = gamma[:, :, np.newaxis] * (mu - y_exp) / (sigma ** 2)  # (N, K, D)
    d_mu = d_mu.reshape(N, K * D) / N

    # d(NLL)/d(sigma_raw): via chain rule through softplus
    # d(NLL)/d(sigma) = gamma_k * (1/sigma - (y-mu)^2/sigma^3)
    d_sigma = gamma[:, :, np.newaxis] * (1.0 / sigma - (y_exp - mu) ** 2 / sigma ** 3)
    # Chain rule: d(sigma)/d(raw) = sigmoid(raw) = sigma * (1 - exp(-sigma))...
    # For softplus: d(softplus(x))/dx = sigmoid(x) = 1 - exp(-softplus(x))
    sig_raw_grad = 1.0 - np.exp(-sigma + 1e-4)  # approx sigmoid
    d_sigma_raw = (d_sigma * sig_raw_grad).reshape(N, K * D) / N

    d_raw = np.concatenate([d_pi_logits, d_mu, d_sigma_raw], axis=1)
    return nll, d_raw


EPOCHS_MDN = 500
for epoch in range(EPOCHS_MDN):
    idx = np.random.choice(N_TRAIN, BS, replace=False)
    r_b, d_b = r_train[idx], d_train_n[idx]

    pi, mu, sigma = mdn_forward(mdn, r_b, store=True)
    nll, d_raw = mdn_nll(pi, mu, sigma, d_b)

    mdn.backward(d_raw)
    mdn.adam_step(lr=1e-3)

    if (epoch + 1) % 100 == 0:
        pi_t, mu_t, sig_t = mdn_forward(mdn, r_test)
        nll_t, _ = mdn_nll(pi_t, mu_t, sig_t, d_test_n)
        print(f"    Epoch {epoch + 1}: test NLL = {nll_t:.4f}")

# Final evaluation
pi_t, mu_t, sig_t = mdn_forward(mdn, r_test)
nll_final, _ = mdn_nll(pi_t, mu_t, sig_t, d_test_n)
print(f"  MDN final test NLL: {nll_final:.4f}")

# Sample multiple solutions
print(f"\n  Example: sampling 5 solutions for test spectrum #0")
print(f"  True thicknesses: {np.round(d_test[0], 1)}")
pi_s, mu_s, sig_s = mdn_forward(mdn, r_test[0:1])
for i in range(5):
    k = np.random.choice(K, p=pi_s[0])
    d_sample = np.random.normal(mu_s[0, k], sig_s[0, k])
    d_sample_denorm = np.clip(d_sample * (D_MAX - D_MIN) + D_MIN, D_MIN, D_MAX)
    r_sample = thin_film_reflectance(d_sample_denorm, REFRACTIVE_INDICES, WAVELENGTHS)
    err = np.mean((r_sample - r_test[0]) ** 2)
    print(f"    Sample {i + 1}: d={np.round(d_sample_denorm, 1)}, recon MSE={err:.6f}")


# ============================================================================
# 7. Comparison Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary: Tandem Network vs MDN for Inverse Design")
print("=" * 70)

# Tandem: single prediction
tandem_errs = []
for i in range(min(20, N_TEST)):
    d_t = np.clip(inv.forward(r_test[i:i + 1]), 0, 1)
    d_t = d_t[0] * (D_MAX - D_MIN) + D_MIN
    r_t = thin_film_reflectance(d_t, REFRACTIVE_INDICES, WAVELENGTHS)
    tandem_errs.append(np.mean((r_t - r_test[i]) ** 2))

# MDN: best of 10 samples
mdn_errs = []
for i in range(min(20, N_TEST)):
    pi_s, mu_s, sig_s = mdn_forward(mdn, r_test[i:i + 1])
    best = float('inf')
    for _ in range(10):
        k = np.random.choice(K, p=pi_s[0])
        d_s = np.random.normal(mu_s[0, k], sig_s[0, k])
        d_s = np.clip(d_s * (D_MAX - D_MIN) + D_MIN, D_MIN, D_MAX)
        r_s = thin_film_reflectance(d_s, REFRACTIVE_INDICES, WAVELENGTHS)
        best = min(best, np.mean((r_s - r_test[i]) ** 2))
    mdn_errs.append(best)

print(f"  Tandem avg recon MSE (20 test):      {np.mean(tandem_errs):.6f}")
print(f"  MDN best-of-10 recon MSE (20 test):  {np.mean(mdn_errs):.6f}")
print(f"\n  Tandem: deterministic, single solution per query")
print(f"  MDN:    probabilistic, multiple valid solutions via sampling")
print(f"\nDone.")
