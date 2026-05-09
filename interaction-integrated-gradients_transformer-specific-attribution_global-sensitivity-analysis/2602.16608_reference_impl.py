"""Minimal reference implementation: Context-Aware Layer-Wise Integrated
Gradients for Explaining Transformer Models — CA-LIG (arXiv:2602.16608).

Demonstrates layer-wise integrated gradients with attention-gradient weighting
on a simplified 2-layer single-head attention model.

Key ideas:
  - Compute IG separately within each Transformer layer (layer-wise IG).
  - Weight token attributions by class-specific attention gradients.
  - Aggregate across layers for final token-level attribution.

Run:
    python 2602.16608_reference_impl.py

Dependencies: numpy, scipy
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax


# ---------------------------------------------------------------------------
# Simplified 2-Layer Attention Model ----------------------------------------
# ---------------------------------------------------------------------------
class SimpleTransformer:
    """Minimal 2-layer single-head attention + FFN model.

    Architecture:
      Input: (seq_len, d_model)
      Layer 1: Attention + FFN
      Layer 2: Attention + FFN
      Classifier: mean-pool -> linear -> softmax

    All weights are fixed random for reproducibility.
    """

    def __init__(self, seq_len: int = 6, d_model: int = 8,
                 n_classes: int = 3, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_layers = 2

        # Attention weights (Q, K, V) for each layer
        self.Wq = [rng.standard_normal((d_model, d_model)) * 0.3
                    for _ in range(self.n_layers)]
        self.Wk = [rng.standard_normal((d_model, d_model)) * 0.3
                    for _ in range(self.n_layers)]
        self.Wv = [rng.standard_normal((d_model, d_model)) * 0.3
                    for _ in range(self.n_layers)]

        # FFN weights for each layer (single hidden layer)
        d_ff = d_model * 2
        self.W_ff1 = [rng.standard_normal((d_model, d_ff)) * 0.3
                      for _ in range(self.n_layers)]
        self.b_ff1 = [rng.standard_normal(d_ff) * 0.1
                      for _ in range(self.n_layers)]
        self.W_ff2 = [rng.standard_normal((d_ff, d_model)) * 0.3
                      for _ in range(self.n_layers)]
        self.b_ff2 = [rng.standard_normal(d_model) * 0.1
                      for _ in range(self.n_layers)]

        # Classifier
        self.W_cls = rng.standard_normal((d_model, n_classes)) * 0.3
        self.b_cls = rng.standard_normal(n_classes) * 0.1

    def attention_layer(self, h: np.ndarray, layer: int):
        """Single-head scaled dot-product attention.

        h: (seq_len, d_model)
        Returns: (output, attention_weights)
        """
        Q = h @ self.Wq[layer]  # (S, D)
        K = h @ self.Wk[layer]
        V = h @ self.Wv[layer]

        scale = np.sqrt(self.d_model)
        scores = Q @ K.T / scale  # (S, S)
        attn = softmax(scores, axis=-1)  # (S, S)

        out = attn @ V  # (S, D)
        return out, attn

    def ffn_layer(self, h: np.ndarray, layer: int):
        """Feed-forward network with ReLU."""
        hidden = np.maximum(0, h @ self.W_ff1[layer] + self.b_ff1[layer])
        return hidden @ self.W_ff2[layer] + self.b_ff2[layer]

    def forward(self, x: np.ndarray, return_intermediates: bool = False):
        """Full forward pass.

        x: (seq_len, d_model)
        Returns: logits (n_classes,), and optionally intermediates.
        """
        intermediates = {"h": [x.copy()], "attn": []}

        h = x.copy()
        for l in range(self.n_layers):
            attn_out, attn_w = self.attention_layer(h, l)
            h = h + attn_out  # residual connection
            ffn_out = self.ffn_layer(h, l)
            h = h + ffn_out  # residual connection
            intermediates["h"].append(h.copy())
            intermediates["attn"].append(attn_w.copy())

        # Classification: mean pool + linear
        pooled = h.mean(axis=0)  # (d_model,)
        logits = pooled @ self.W_cls + self.b_cls  # (n_classes,)

        if return_intermediates:
            return logits, intermediates
        return logits

    def forward_from_layer(self, h: np.ndarray, from_layer: int):
        """Forward pass starting from a given layer's output.

        h: (seq_len, d_model) — intermediate representation.
        from_layer: index of the layer to start from (0-indexed, after this layer).
        """
        for l in range(from_layer, self.n_layers):
            attn_out, _ = self.attention_layer(h, l)
            h = h + attn_out
            ffn_out = self.ffn_layer(h, l)
            h = h + ffn_out

        pooled = h.mean(axis=0)
        logits = pooled @ self.W_cls + self.b_cls
        return logits


# ---------------------------------------------------------------------------
# Layer-Wise Integrated Gradients (LIG) -------------------------------------
# ---------------------------------------------------------------------------
def _numerical_grad_wrt_h(model, h: np.ndarray, from_layer: int,
                           target_class: int, eps: float = 1e-5):
    """Compute gradient of logit[target_class] w.r.t. intermediate h.

    h: (seq_len, d_model)
    Returns: gradient of same shape.
    """
    grad = np.zeros_like(h)
    h_flat = h.flatten()
    for idx in range(len(h_flat)):
        h_p = h_flat.copy(); h_p[idx] += eps
        h_m = h_flat.copy(); h_m[idx] -= eps
        logits_p = model.forward_from_layer(h_p.reshape(h.shape), from_layer)
        logits_m = model.forward_from_layer(h_m.reshape(h.shape), from_layer)
        grad.flat[idx] = (logits_p[target_class] - logits_m[target_class]) / (2 * eps)
    return grad


def layer_wise_ig(model, h_actual: np.ndarray, h_baseline: np.ndarray,
                  from_layer: int, target_class: int,
                  steps: int = 30) -> np.ndarray:
    """Compute layer-wise IG at a given layer.

    LIG_i^(l) = (h_i - h'_i) * (1/M) * sum_m grad_i(h' + m/M * (h - h'))

    Returns: attribution of shape (seq_len, d_model).
    """
    delta = h_actual - h_baseline
    grad_sum = np.zeros_like(delta)

    for m in range(1, steps + 1):
        alpha = m / steps
        h_interp = h_baseline + alpha * delta
        grad = _numerical_grad_wrt_h(model, h_interp, from_layer, target_class)
        grad_sum += grad

    lig = delta * grad_sum / steps
    return lig


# ---------------------------------------------------------------------------
# Context-Aware Attention Weighting -----------------------------------------
# ---------------------------------------------------------------------------
def attention_gradient_weights(model, x: np.ndarray, target_class: int,
                                eps: float = 1e-5):
    """Compute context-aware attention weights for each layer.

    w_t^(l) = sum_j ReLU(dY_c/dA_{t,j}^(l)) * A_{t,j}^(l)

    Returns: list of weight arrays, each (seq_len,).
    """
    logits, intermediates = model.forward(x, return_intermediates=True)
    weights = []

    for l in range(model.n_layers):
        attn = intermediates["attn"][l]  # (S, S)
        S = attn.shape[0]

        # Numerical gradient of logit w.r.t. attention weights
        grad_attn = np.zeros((S, S))
        for i in range(S):
            for j in range(S):
                # Perturb attention weight A[i,j]
                # We recompute forward with perturbed attention
                # For simplicity, use finite difference on the full forward
                attn_p = attn.copy(); attn_p[i, j] += eps
                attn_m = attn.copy(); attn_m[i, j] -= eps

                # Recompute V for this layer
                h_prev = intermediates["h"][l]
                V = h_prev @ model.Wv[l]

                # After attention with perturbed weights
                h_after_attn_p = h_prev + attn_p @ V
                h_after_attn_m = h_prev + attn_m @ V

                # Continue forward from after FFN of this layer
                ffn_out_p = model.ffn_layer(h_after_attn_p, l)
                h_out_p = h_after_attn_p + ffn_out_p
                ffn_out_m = model.ffn_layer(h_after_attn_m, l)
                h_out_m = h_after_attn_m + ffn_out_m

                logits_p = model.forward_from_layer(h_out_p, l + 1)
                logits_m = model.forward_from_layer(h_out_m, l + 1)

                grad_attn[i, j] = (logits_p[target_class] -
                                   logits_m[target_class]) / (2 * eps)

        # Grad-CAM-like aggregation: w_t = sum_j ReLU(grad_A_{t,j}) * A_{t,j}
        grad_relu = np.maximum(0, grad_attn)
        w = np.sum(grad_relu * attn, axis=1)  # (S,)

        # Normalize per layer
        w_sum = w.sum()
        if w_sum > 1e-12:
            w = w / w_sum

        weights.append(w)

    return weights


# ---------------------------------------------------------------------------
# CA-LIG: Full Algorithm ----------------------------------------------------
# ---------------------------------------------------------------------------
def ca_lig(model, x: np.ndarray, baseline: np.ndarray,
           target_class: int, steps: int = 30):
    """Context-Aware Layer-Wise Integrated Gradients.

    Algorithm:
      1. Forward pass to get intermediates h^(l) and attention A^(l).
      2. For each layer l, compute LIG^(l) and attention-gradient weight w^(l).
      3. Aggregate: CA-LIG_t = sum_l w_t^(l) * ||LIG_t^(l)||_2

    Returns:
      - token_attribution: (seq_len,) — normalized token-level attributions.
      - layer_contributions: (n_layers,) — per-layer contribution ratios.
      - lig_per_layer: list of (seq_len, d_model) — raw LIG per layer.
    """
    # Get intermediates for actual input
    _, inter_actual = model.forward(x, return_intermediates=True)

    # Get intermediates for baseline
    _, inter_baseline = model.forward(baseline, return_intermediates=True)

    # Attention-gradient weights
    attn_weights = attention_gradient_weights(model, x, target_class)

    # Layer-wise IG
    lig_per_layer = []
    lig_norms = []  # (n_layers, seq_len)

    for l in range(model.n_layers):
        h_act = inter_actual["h"][l]
        h_bas = inter_baseline["h"][l]
        lig = layer_wise_ig(model, h_act, h_bas,
                            from_layer=l, target_class=target_class,
                            steps=steps)
        lig_per_layer.append(lig)

        # Token-level L2 norm of LIG
        token_norms = np.linalg.norm(lig, axis=1)  # (seq_len,)
        lig_norms.append(token_norms)

    lig_norms = np.array(lig_norms)  # (n_layers, seq_len)

    # CA-LIG aggregation
    ca_lig_scores = np.zeros(model.seq_len)
    layer_contributions = np.zeros(model.n_layers)

    for l in range(model.n_layers):
        weighted = attn_weights[l] * lig_norms[l]  # (seq_len,)
        ca_lig_scores += weighted
        layer_contributions[l] = weighted.sum()

    # Normalize
    total = ca_lig_scores.sum()
    if total > 1e-12:
        token_attribution = ca_lig_scores / total
    else:
        token_attribution = np.ones(model.seq_len) / model.seq_len

    lc_total = layer_contributions.sum()
    if lc_total > 1e-12:
        layer_contributions = layer_contributions / lc_total

    return token_attribution, layer_contributions, lig_per_layer


# ---------------------------------------------------------------------------
# Comparison: Standard IG vs CA-LIG -----------------------------------------
# ---------------------------------------------------------------------------
def standard_ig_tokens(model, x: np.ndarray, baseline: np.ndarray,
                       target_class: int, steps: int = 30):
    """Standard (non-layer-wise) IG for token-level attribution.

    Treats the full model as F(x) and computes IG on the flattened input.
    Token attribution = L2 norm of per-token IG vector.
    """
    delta = x - baseline
    grad_sum = np.zeros_like(x)

    for m in range(1, steps + 1):
        alpha = m / steps
        x_interp = baseline + alpha * delta

        # Gradient of logit[target_class] w.r.t. input x
        grad = np.zeros_like(x)
        x_flat = x_interp.flatten()
        for idx in range(len(x_flat)):
            xp = x_flat.copy(); xp[idx] += 1e-5
            xm = x_flat.copy(); xm[idx] -= 1e-5
            lp = model.forward(xp.reshape(x.shape))[target_class]
            lm = model.forward(xm.reshape(x.shape))[target_class]
            grad.flat[idx] = (lp - lm) / (2e-5)
        grad_sum += grad

    ig = delta * grad_sum / steps
    token_attr = np.linalg.norm(ig, axis=1)  # (seq_len,)
    total = token_attr.sum()
    if total > 1e-12:
        token_attr = token_attr / total
    return token_attr


# ---------------------------------------------------------------------------
# Main demo -----------------------------------------------------------------
# ---------------------------------------------------------------------------
def main():
    np.set_printoptions(precision=4, suppress=True)

    seq_len = 6
    d_model = 8
    n_classes = 3

    print("CA-LIG: Context-Aware Layer-Wise Integrated Gradients (arXiv:2602.16608)")
    print("=" * 70)

    model = SimpleTransformer(seq_len=seq_len, d_model=d_model,
                               n_classes=n_classes, seed=42)

    # Create synthetic input and baseline
    rng = np.random.default_rng(123)
    x = rng.standard_normal((seq_len, d_model)) * 0.5
    baseline = np.zeros((seq_len, d_model))

    # Forward pass
    logits = model.forward(x)
    probs = softmax(logits)
    target_class = int(np.argmax(logits))

    print(f"\n[Input]")
    print(f"  Sequence length: {seq_len}, Model dim: {d_model}")
    print(f"  Logits: {logits}")
    print(f"  Predicted class: {target_class} (prob={probs[target_class]:.4f})")

    # --- CA-LIG ---
    print(f"\n[1] CA-LIG Attribution")
    print("-" * 50)
    token_attr, layer_contrib, lig_layers = ca_lig(
        model, x, baseline, target_class, steps=20
    )

    print(f"  Token attributions:")
    token_labels = [f"tok_{i}" for i in range(seq_len)]
    for i in range(seq_len):
        bar = "#" * int(token_attr[i] * 40)
        print(f"    {token_labels[i]}: {token_attr[i]:.4f} |{bar}")

    print(f"\n  Layer contributions:")
    for l in range(model.n_layers):
        bar = "#" * int(layer_contrib[l] * 40)
        print(f"    Layer {l+1}: {layer_contrib[l]:.4f} |{bar}")

    # --- Standard IG for comparison ---
    print(f"\n[2] Standard IG Attribution (baseline comparison)")
    print("-" * 50)
    std_attr = standard_ig_tokens(model, x, baseline, target_class, steps=20)
    for i in range(seq_len):
        bar = "#" * int(std_attr[i] * 40)
        print(f"    {token_labels[i]}: {std_attr[i]:.4f} |{bar}")

    # --- Rank comparison ---
    print(f"\n[3] Rank Comparison")
    print("-" * 50)
    rank_calig = np.argsort(-token_attr) + 1
    rank_std = np.argsort(-std_attr) + 1
    print(f"  CA-LIG ranking:     {rank_calig}")
    print(f"  Standard IG ranking: {rank_std}")

    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(token_attr, std_attr)
    print(f"  Spearman correlation: rho={rho:.4f}, p={pval:.4f}")

    # --- Layer-wise analysis ---
    print(f"\n[4] Layer-Wise Sensitivity Decomposition")
    print("-" * 50)
    for l in range(model.n_layers):
        lig_norms = np.linalg.norm(lig_layers[l], axis=1)
        print(f"  Layer {l+1} LIG norms per token: {lig_norms}")

    # --- Attention weights analysis ---
    print(f"\n[5] Context-Aware Attention Weights")
    print("-" * 50)
    attn_weights = attention_gradient_weights(model, x, target_class)
    for l in range(model.n_layers):
        print(f"  Layer {l+1} weights: {attn_weights[l]}")

    print(f"\n{'=' * 70}")
    print("Done. CA-LIG provides layer-resolved, context-aware token attributions.")


if __name__ == "__main__":
    main()
