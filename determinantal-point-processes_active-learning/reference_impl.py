"""
Reference implementation: DPP-based Data Curation for ML Interatomic Potentials
(arXiv:2603.22160)

Core idea: Use Determinantal Point Processes to select diverse, informative
subsets of atomic configurations for training MLIPs. The DPP kernel is
constructed from molecular descriptors to promote conformational diversity.
"""

import numpy as np
from scipy.linalg import det


# ---------------------------------------------------------------------------
# Toy molecular descriptors (SOAP-like)
# ---------------------------------------------------------------------------

def compute_descriptors(configs, n_desc=16):
    """
    Compute descriptor vectors for atomic configurations.
    In practice this would be SOAP or ACE; here we use a random feature map.
    configs: (N, n_atoms, 3) array of atomic positions
    Returns: (N, n_desc) descriptor matrix
    """
    N = len(configs)
    descriptors = np.zeros((N, n_desc))
    for i, conf in enumerate(configs):
        # radial distribution features
        n_atoms = len(conf)
        dists = []
        for a in range(n_atoms):
            for b in range(a + 1, n_atoms):
                dists.append(np.linalg.norm(conf[a] - conf[b]))
        dists = np.array(dists) if dists else np.array([1.0])

        # histogram-like features
        bins = np.linspace(0, 5, n_desc + 1)
        hist, _ = np.histogram(dists, bins=bins, density=True)
        descriptors[i] = hist

    return descriptors


# ---------------------------------------------------------------------------
# DPP kernel from descriptors
# ---------------------------------------------------------------------------

def build_dpp_kernel(descriptors, sigma=1.0):
    """
    Build L-ensemble DPP kernel from descriptor similarity.
    L_ij = exp(-||d_i - d_j||^2 / (2 * sigma^2))
    """
    sq_dists = (
        np.sum(descriptors**2, axis=1, keepdims=True)
        - 2 * descriptors @ descriptors.T
        + np.sum(descriptors**2, axis=1)
    )
    return np.exp(-0.5 * sq_dists / sigma**2)


# ---------------------------------------------------------------------------
# DPP sampling (eigendecomposition-based)
# ---------------------------------------------------------------------------

def sample_dpp_k(L, k):
    """Sample a subset of size k from L-ensemble DPP."""
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals = np.maximum(eigvals, 0)

    # phase 1: select eigenvectors proportionally
    probs = eigvals / (eigvals + 1)
    selected = np.where(np.random.rand(len(probs)) < probs)[0]

    if len(selected) == 0:
        return np.random.choice(L.shape[0], size=k, replace=False)

    V = eigvecs[:, selected].copy()
    n = L.shape[0]
    chosen = []

    # phase 2: sequential item selection
    for _ in range(min(k, V.shape[1])):
        probs_i = np.sum(V**2, axis=1)
        probs_i = np.maximum(probs_i, 0)
        total = probs_i.sum()
        if total < 1e-12:
            break
        probs_i /= total
        idx = np.random.choice(n, p=probs_i)
        chosen.append(idx)

        if V.shape[1] <= 1:
            break
        j = np.argmax(np.abs(V[idx]))
        Vj = V[:, j: j + 1].copy()
        denom = Vj[idx] + 1e-12
        V -= (V[idx: idx + 1, :] / denom) * Vj
        V = np.delete(V, j, axis=1)

    # fill remaining slots if needed
    remaining = list(set(range(n)) - set(chosen))
    while len(chosen) < k and remaining:
        chosen.append(remaining.pop(np.random.randint(len(remaining))))

    return np.array(chosen[:k])


# ---------------------------------------------------------------------------
# Baseline: Farthest Point Sampling (FPS)
# ---------------------------------------------------------------------------

def farthest_point_sampling(descriptors, k):
    """Greedy FPS in descriptor space."""
    N = len(descriptors)
    selected = [np.random.randint(N)]
    min_dists = np.full(N, np.inf)

    for _ in range(k - 1):
        last = descriptors[selected[-1]]
        dists = np.sum((descriptors - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1
        selected.append(np.argmax(min_dists))

    return np.array(selected)


# ---------------------------------------------------------------------------
# Toy MLIP training & evaluation
# ---------------------------------------------------------------------------

def train_evaluate_mlip(X_train, y_train, X_test, y_test):
    """Simple kernel ridge regression as MLIP proxy."""
    sigma = 1.0
    lam = 1e-3
    K_train = np.exp(-0.5 * (
        np.sum(X_train**2, 1, keepdims=True)
        - 2 * X_train @ X_train.T
        + np.sum(X_train**2, 1)
    ) / sigma**2)
    alpha = np.linalg.solve(K_train + lam * np.eye(len(K_train)), y_train)

    K_test = np.exp(-0.5 * (
        np.sum(X_test**2, 1, keepdims=True)
        - 2 * X_test @ X_train.T
        + np.sum(X_train**2, 1)
    ) / sigma**2)
    y_pred = K_test @ alpha
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    return rmse


# ---------------------------------------------------------------------------
# Demo: compare DPP vs FPS vs Random selection
# ---------------------------------------------------------------------------

def demo():
    np.random.seed(42)

    # generate synthetic atomic configurations
    N_pool = 500
    N_test = 100
    n_atoms = 8
    configs = [np.random.randn(n_atoms, 3) * (1 + 0.3 * np.random.randn()) for _ in range(N_pool + N_test)]

    # compute descriptors
    descs = compute_descriptors(configs)
    descs_pool = descs[:N_pool]
    descs_test = descs[N_pool:]

    # synthetic "energies" (target for MLIP)
    def energy_fn(desc):
        return np.sin(desc[:4].sum()) + 0.5 * np.cos(desc[4:8].sum()) + 0.1 * desc.sum()

    energies_pool = np.array([energy_fn(d) for d in descs_pool])
    energies_test = np.array([energy_fn(d) for d in descs_test])

    k = 50  # training set size

    print(f"Pool size: {N_pool}, Test size: {N_test}, Selection size: {k}\n")

    # DPP selection
    L = build_dpp_kernel(descs_pool, sigma=0.5)
    dpp_idx = sample_dpp_k(L, k)
    rmse_dpp = train_evaluate_mlip(descs_pool[dpp_idx], energies_pool[dpp_idx], descs_test, energies_test)
    print(f"DPP selection:    RMSE = {rmse_dpp:.4f}")

    # FPS selection
    fps_idx = farthest_point_sampling(descs_pool, k)
    rmse_fps = train_evaluate_mlip(descs_pool[fps_idx], energies_pool[fps_idx], descs_test, energies_test)
    print(f"FPS selection:    RMSE = {rmse_fps:.4f}")

    # random selection (average over 5 trials)
    rmses_rand = []
    for _ in range(5):
        rand_idx = np.random.choice(N_pool, k, replace=False)
        rmses_rand.append(
            train_evaluate_mlip(descs_pool[rand_idx], energies_pool[rand_idx], descs_test, energies_test)
        )
    print(f"Random selection: RMSE = {np.mean(rmses_rand):.4f} +/- {np.std(rmses_rand):.4f}")


if __name__ == "__main__":
    demo()
