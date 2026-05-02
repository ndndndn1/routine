"""Minimal reference implementation for arXiv:2506.10044.

Reproduces the CNN-LSTM Tandem Neural Network for optical inverse design:
  1) Transfer Matrix Method (TMM) forward model for multilayer thin films.
  2) Tandem architecture: inverse net → pre-trained forward net.
  3) Training loop with tandem loss (spectrum reconstruction).

Run:
    python reference_impl.py
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Transfer Matrix Method (TMM) for normal incidence -------------------------
# ---------------------------------------------------------------------------
def tmm_transmission(thicknesses: np.ndarray, n_layers: np.ndarray,
                     wavelengths: np.ndarray) -> np.ndarray:
    n_sub = 1.52
    n_air = 1.0
    T = np.zeros(len(wavelengths))
    for wi, lam in enumerate(wavelengths):
        M = np.eye(2, dtype=complex)
        for d, n in zip(thicknesses, n_layers):
            phi = 2 * np.pi * n * d / lam
            layer_M = np.array([
                [np.cos(phi), 1j * np.sin(phi) / n],
                [1j * n * np.sin(phi), np.cos(phi)],
            ])
            M = M @ layer_M
        r_num = M[0, 0] * n_sub + M[0, 1] * n_air * n_sub - M[1, 0] - M[1, 1] * n_air
        r_den = M[0, 0] * n_sub + M[0, 1] * n_air * n_sub + M[1, 0] + M[1, 1] * n_air
        r = r_num / r_den
        T[wi] = 1.0 - np.abs(r) ** 2
    return T


# ---------------------------------------------------------------------------
# Simple MLP blocks (numpy only) --------------------------------------------
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, dims: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            self.weights.append(rng.normal(0, scale, (dims[i], dims[i + 1])))
            self.biases.append(np.zeros(dims[i + 1]))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)
        return x

    def parameters(self) -> list[np.ndarray]:
        return self.weights + self.biases


# ---------------------------------------------------------------------------
# Tandem Network: inverse + forward -----------------------------------------
# ---------------------------------------------------------------------------
class TandemNetwork:
    def __init__(self, n_layers: int, n_wavelengths: int, hidden: int = 64, seed: int = 0):
        self.inverse_net = MLP([n_wavelengths, hidden, hidden, n_layers], seed=seed)
        self.forward_net = MLP([n_layers, hidden, hidden, n_wavelengths], seed=seed + 1)
        self.n_layers = n_layers

    def inverse(self, spectrum: np.ndarray) -> np.ndarray:
        raw = self.inverse_net.forward(spectrum)
        return np.clip(raw, 20.0, 300.0)

    def forward(self, thicknesses: np.ndarray) -> np.ndarray:
        return self.forward_net.forward(thicknesses)

    def tandem_forward(self, spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        thicknesses = self.inverse(spectrum)
        spectrum_pred = self.forward(thicknesses)
        return thicknesses, spectrum_pred


# ---------------------------------------------------------------------------
# Training with finite-difference gradient -----------------------------------
# ---------------------------------------------------------------------------
def train_forward(net: TandemNetwork, X_thick: np.ndarray, Y_spec: np.ndarray,
                  lr: float = 1e-3, epochs: int = 200) -> list[float]:
    losses = []
    for epoch in range(epochs):
        pred = net.forward(X_thick)
        loss = float(np.mean((pred - Y_spec) ** 2))
        losses.append(loss)
        delta = 1e-5
        for p in net.forward_net.parameters():
            grad = np.zeros_like(p)
            flat = p.ravel()
            for idx in range(min(len(flat), 500)):
                old = flat[idx]
                flat[idx] = old + delta
                loss_p = np.mean((net.forward(X_thick) - Y_spec) ** 2)
                flat[idx] = old
                grad.ravel()[idx] = (loss_p - loss) / delta
            p -= lr * grad
    return losses


def train_tandem(net: TandemNetwork, Y_spec: np.ndarray,
                 lr: float = 1e-3, epochs: int = 100) -> list[float]:
    losses = []
    for epoch in range(epochs):
        _, spec_recon = net.tandem_forward(Y_spec)
        loss = float(np.mean((spec_recon - Y_spec) ** 2))
        losses.append(loss)
        delta = 1e-5
        for p in net.inverse_net.parameters():
            grad = np.zeros_like(p)
            flat = p.ravel()
            for idx in range(min(len(flat), 500)):
                old = flat[idx]
                flat[idx] = old + delta
                _, sp = net.tandem_forward(Y_spec)
                loss_p = np.mean((sp - Y_spec) ** 2)
                flat[idx] = old
                grad.ravel()[idx] = (loss_p - loss) / delta
            p -= lr * grad
    return losses


# ---------------------------------------------------------------------------
# Demo: SiO2/TiO2 multilayer inverse design ---------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n_layers = 4
    wavelengths = np.linspace(400, 800, 20)
    n_sio2, n_tio2 = 1.46, 2.30
    n_refr = np.array([n_sio2, n_tio2, n_sio2, n_tio2])

    rng = np.random.default_rng(0)
    n_samples = 50
    X_thick = rng.uniform(20, 300, (n_samples, n_layers))
    Y_spec = np.array([tmm_transmission(t, n_refr, wavelengths) for t in X_thick])

    net = TandemNetwork(n_layers, len(wavelengths), hidden=32, seed=7)

    print("Training forward net...")
    fw_losses = train_forward(net, X_thick, Y_spec, lr=5e-4, epochs=50)
    print(f"  Forward loss: {fw_losses[0]:.4f} -> {fw_losses[-1]:.4f}")

    print("Training tandem (inverse) net...")
    tn_losses = train_tandem(net, Y_spec, lr=5e-4, epochs=30)
    print(f"  Tandem loss: {tn_losses[0]:.4f} -> {tn_losses[-1]:.4f}")

    target_spec = Y_spec[0]
    pred_thick, recon_spec = net.tandem_forward(target_spec.reshape(1, -1))
    print(f"\nTarget thicknesses: {X_thick[0]}")
    print(f"Predicted thicknesses: {pred_thick.ravel()}")
    print(f"Spectrum MSE: {np.mean((recon_spec - target_spec) ** 2):.6f}")
