"""Minimal reference implementation for arXiv:2510.22176.

Reproduces the core pipeline:
  1) A 2D CNN surrogate mapping binary device masks to figures of merit (FoM).
  2) Integrated Gradients (IG) attribution on the surrogate.
  3) Fabrication perturbation analysis using IG sensitivity maps.

Run:
    python reference_impl.py
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Tiny 2D convolution helpers (no torch dependency) -------------------------
# ---------------------------------------------------------------------------
def _conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    kh, kw = w.shape[2], w.shape[3]
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    n, ci, h, ww = x.shape
    co = w.shape[0]
    out = np.zeros((n, co, h, ww))
    for i in range(kh):
        for j in range(kw):
            out += xp[:, np.newaxis, :, i:i + h, j:j + ww] * w[np.newaxis, :, :, i:i + 1, j:j + 1]
    return out.sum(axis=2).reshape(n, co, h, ww) + b[np.newaxis, :, np.newaxis, np.newaxis]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _global_avg_pool(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=(2, 3))


# ---------------------------------------------------------------------------
# Simple CNN surrogate: mask (1, H, W) -> FoM scalar -----------------------
# ---------------------------------------------------------------------------
class CNNSurrogate:
    def __init__(self, h: int = 16, w: int = 16, seed: int = 42):
        rng = np.random.default_rng(seed)
        s = 0.3
        self.w1 = rng.normal(0, s, (4, 1, 3, 3))
        self.b1 = np.zeros(4)
        self.w2 = rng.normal(0, s, (8, 4, 3, 3))
        self.b2 = np.zeros(8)
        self.fc = rng.normal(0, s, (1, 8))
        self.fc_b = np.zeros(1)

    def forward(self, mask: np.ndarray) -> float:
        x = mask[np.newaxis, np.newaxis, :, :]
        x = _relu(_conv2d_simple(x, self.w1, self.b1))
        x = _relu(_conv2d_simple(x, self.w2, self.b2))
        x = _global_avg_pool(x)
        return float((x @ self.fc.T + self.fc_b).item())


def _conv2d_simple(x, w, b):
    n, ci, h, ww = x.shape
    co, _, kh, kw = w.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    out = np.zeros((n, co, h, ww))
    for oc in range(co):
        for ic in range(ci):
            for ki in range(kh):
                for kj in range(kw):
                    out[:, oc] += xp[:, ic, ki:ki + h, kj:kj + ww] * w[oc, ic, ki, kj]
        out[:, oc] += b[oc]
    return out


# ---------------------------------------------------------------------------
# Integrated Gradients attribution ------------------------------------------
# ---------------------------------------------------------------------------
def integrated_gradients(
    model: CNNSurrogate,
    mask: np.ndarray,
    baseline: np.ndarray | None = None,
    steps: int = 50,
    delta: float = 1e-4,
) -> np.ndarray:
    if baseline is None:
        baseline = np.zeros_like(mask)
    h, w = mask.shape
    ig = np.zeros_like(mask)
    for step in range(steps):
        alpha = step / steps
        interp = baseline + alpha * (mask - baseline)
        grad = np.zeros_like(mask)
        f0 = model.forward(interp)
        for i in range(h):
            for j in range(w):
                perturbed = interp.copy()
                perturbed[i, j] += delta
                grad[i, j] = (model.forward(perturbed) - f0) / delta
        ig += grad
    ig = (mask - baseline) * ig / steps
    return ig


# ---------------------------------------------------------------------------
# Fabrication perturbation analysis -----------------------------------------
# ---------------------------------------------------------------------------
def fab_perturbation_analysis(
    model: CNNSurrogate,
    mask: np.ndarray,
    ig_map: np.ndarray,
    n_perturb: int = 20,
    flip_frac: float = 0.05,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    h, w = mask.shape
    n_flip = max(1, int(h * w * flip_frac))
    abs_ig = np.abs(ig_map).ravel()
    high_sens_idx = np.argsort(abs_ig)[-n_flip:]
    low_sens_idx = np.argsort(abs_ig)[:n_flip]

    fom_base = model.forward(mask)
    delta_high, delta_low = [], []
    for _ in range(n_perturb):
        m_high = mask.copy()
        for idx in high_sens_idx:
            r, c = divmod(idx, w)
            m_high[r, c] = 1.0 - m_high[r, c]
        delta_high.append(abs(model.forward(m_high) - fom_base))

        m_low = mask.copy()
        for idx in low_sens_idx:
            r, c = divmod(idx, w)
            m_low[r, c] = 1.0 - m_low[r, c]
        delta_low.append(abs(model.forward(m_low) - fom_base))

    return {
        "fom_base": fom_base,
        "mean_delta_high_sens": float(np.mean(delta_high)),
        "mean_delta_low_sens": float(np.mean(delta_low)),
        "sensitivity_ratio": float(np.mean(delta_high) / (np.mean(delta_low) + 1e-12)),
    }


# ---------------------------------------------------------------------------
# Demo ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    H, W = 8, 8
    rng = np.random.default_rng(123)
    mask = (rng.random((H, W)) > 0.5).astype(float)

    model = CNNSurrogate(H, W, seed=42)
    print("FoM =", model.forward(mask))

    ig_map = integrated_gradients(model, mask, steps=20)
    print("IG map range:", ig_map.min(), "~", ig_map.max())

    result = fab_perturbation_analysis(model, mask, ig_map)
    print("Perturbation analysis:")
    for k, v in result.items():
        print(f"  {k}: {v:.6f}")
