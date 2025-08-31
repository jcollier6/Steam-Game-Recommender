import os
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from .utils import load_json, save_numpy, save_torch


def run_tag_pca(games_df: pd.DataFrame, dim: int = 128, sppmi_shift: float = 5.0) -> None:
    """Build game embeddings from a bipartite game–tag graph via SPPMI + SVD.

    - Edge weight uses a rank-based weight w = (21 - rank) / 20 (same as before)
    - SPPMI(g, t) = max(log p(g,t) - log p(g) - log p(t) - log(k), 0), k = sppmi_shift
    - TruncatedSVD to fixed dim, then row L2-normalization
    """
    tag_id_map = load_json('ml/data/tag_id_map.json')
    V = len(tag_id_map)
    N = len(games_df)

    app_ids = games_df['app_id'].astype(int).to_numpy()

    # Build sparse game–tag weight matrix W (CSR)
    rows = []
    cols = []
    data = []
    for row_idx, row in games_df.iterrows():
        tags = row.get('tags') or []
        for tag_id, rank in tags:
            key = str(tag_id)
            j = tag_id_map.get(key)
            if j is None:
                continue
            w = (21 - int(rank)) / 20.0
            if w <= 0:
                continue
            rows.append(row_idx)
            cols.append(int(j))
            data.append(float(w))

    if not data:
        raise RuntimeError('No tag edges found to build bipartite embeddings')

    W = sparse.csr_matrix((data, (rows, cols)), shape=(N, V), dtype=np.float64)

    # Compute SPPMI matrix on edges
    total_mass = float(W.sum())
    p_g = np.asarray(W.sum(axis=1)).flatten() / max(total_mass, 1e-12)
    p_t = np.asarray(W.sum(axis=0)).flatten() / max(total_mass, 1e-12)
    log_k = np.log(sppmi_shift)

    # For each nonzero entry, compute SPPMI value
    W = W.tocoo()
    sppmi_vals = []
    for i, j, w in zip(W.row, W.col, W.data):
        p_gt = w / total_mass
        if p_gt <= 0 or p_g[i] <= 0 or p_t[j] <= 0:
            sppmi_vals.append(0.0)
        else:
            val = np.log(p_gt) - np.log(p_g[i]) - np.log(p_t[j]) - log_k
            sppmi_vals.append(val if val > 0 else 0.0)
    S = sparse.csr_matrix((sppmi_vals, (W.row, W.col)), shape=(N, V), dtype=np.float64)

    # Truncated SVD to fixed dimensions
    k = min(dim, min(N, V) - 1) if min(N, V) > 1 else 1
    svd = TruncatedSVD(n_components=k, random_state=42)
    G_emb = svd.fit_transform(S)

    # Row L2-normalize
    norms = np.linalg.norm(G_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    G_norm = (G_emb / norms).astype('float32')

    os.makedirs('ml/data', exist_ok=True)
    save_numpy(G_norm, 'ml/data/T_pca_norm.npy')  # keep filename for downstream
    save_torch(svd, 'ml/data/pca_tags.pkl')        # store SVD model for reproducibility
    save_numpy(app_ids, 'ml/data/tag_pca_app_ids.npy')
    global_tag_mean = G_norm.mean(axis=0)
    save_numpy(global_tag_mean, 'ml/data/global_tag_mean.npy')
    print(f'✅ Chunk 3 bipartite SPPMI+SVD saved (dim={k})')

