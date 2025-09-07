import os
import math
import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy import sparse

from .utils import load_json, save_numpy, save_torch, save_json


def _rank_weight(r: int, Rmax: int, w_min: float, alpha: float) -> float:
    x = 1.0 - (r - 1) / max(Rmax - 1, 1)
    return w_min + (1.0 - w_min) * (x ** alpha)


def run_tag_pca(
    games_df: pd.DataFrame,
    svd_dim: int = 128,
    R_max: int = 20,
    w_min: float = 0.8,
    alpha: float = 1.2,
    min_tag_count: int = 3,
    min_tag_df: int = 10,
    beta: float = 0.15,
    beta_max: float = 0.35,
    m_pair: int = 20,
    alpha_cds: float = 0.75,
    sppmi_shift: float = 10.0,
) -> None:
    """Build robust tag-based game embeddings.

    IDF weighting, co-tag PMI boosts and CDS-smoothed SPPMI produce dense
    embeddings whose artifacts are written to ``ml/data``.
    """

    tag_id_map = load_json("ml/data/tag_id_map.json")
    N_games = len(games_df)

    # ------------------------------------------------------------------
    # Document frequency over full corpus
    # ------------------------------------------------------------------
    V = len(tag_id_map)
    df_counts = np.zeros(V, dtype=np.int64)
    for _, row in games_df.iterrows():
        tags = row.get("tags") or []
        seen = set()
        for tag_id, _ in tags[:R_max]:
            key = str(tag_id)
            j = tag_id_map.get(key)
            if j is not None and j not in seen:
                df_counts[int(j)] += 1
                seen.add(j)
    idf = np.log((N_games + 1) / (df_counts + 1)) + 1.0

    keep_mask = df_counts >= int(min_tag_df)
    kept_indices = np.where(keep_mask)[0]
    idf_kept = idf[kept_indices].astype("float32")
    index_map = {int(orig): int(new) for new, orig in enumerate(kept_indices)}
    kept_tag_map = {
        tid: index_map[int(j)]
        for tid, j in tag_id_map.items()
        if keep_mask[int(j)]
    }
    V_kept = len(kept_indices)

    # ------------------------------------------------------------------
    # Co-tag PMI table (binary counts)
    # ------------------------------------------------------------------
    tag_counts = np.zeros(V_kept, dtype=np.int64)
    pair_counts: dict[tuple[int, int], int] = {}
    for _, row in games_df.iterrows():
        tags = row.get("tags") or []
        idxs = sorted({kept_tag_map[str(tid)] for tid, _ in tags[:R_max] if str(tid) in kept_tag_map})
        for j in idxs:
            tag_counts[j] += 1
        for a, b in combinations(idxs, 2):
            if a > b:
                a, b = b, a
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    p_j = tag_counts / max(N_games, 1)

    rows_pmi, cols_pmi, data_pmi = [], [], []
    pmi_dict: dict[int, dict[int, float]] = {}
    for (a, b), c in pair_counts.items():
        if c < m_pair:
            continue
        p_ab = c / N_games
        denom = p_j[a] * p_j[b]
        if denom <= 0:
            continue
        val = math.log(p_ab / denom)
        if val <= 0:
            continue
        rows_pmi.append(a)
        cols_pmi.append(b)
        data_pmi.append(val)
        pmi_dict.setdefault(a, {})[b] = val
        pmi_dict.setdefault(b, {})[a] = val
    pmi_matrix = sparse.coo_matrix((data_pmi, (rows_pmi, cols_pmi)), shape=(V_kept, V_kept))

    # ------------------------------------------------------------------
    # Build weighted game-tag matrix on training subset
    # ------------------------------------------------------------------
    games_train = games_df[games_df["tags"].apply(lambda t: len(t or []) >= min_tag_count)].reset_index(drop=True)
    N_train = len(games_train)
    rows, cols, data = [], [], []
    app_ids = games_train["app_id"].astype(int).to_numpy()
    for i, row in games_train.iterrows():
        tags = row.get("tags") or []
        for tag_id, rank in tags[:R_max]:
            key = str(tag_id)
            j = kept_tag_map.get(key)
            if j is None:
                continue
            w = _rank_weight(int(rank), R_max, w_min, alpha)
            rows.append(i)
            cols.append(j)
            data.append(float(w * idf_kept[j]))
    if not data:
        raise RuntimeError("No tag edges found to build embeddings")
    W = sparse.csr_matrix((data, (rows, cols)), shape=(N_train, V_kept), dtype=np.float64)

    # Co-tag PMI boost per game
    for i in range(N_train):
        start, end = W.indptr[i], W.indptr[i + 1]
        cols_i = W.indices[start:end]
        data_i = W.data[start:end]
        for pos, j in enumerate(cols_i):
            neigh = [pmi_dict.get(j, {}).get(k, 0.0) for k in cols_i if k != j]
            if neigh:
                boost = sum(neigh) / len(neigh)
                factor = 1.0 + beta * boost
                factor = min(factor, 1.0 + beta_max)
                data_i[pos] *= factor
        row_sum = data_i.sum()
        if row_sum > 0:
            W.data[start:end] = data_i / row_sum

    # ------------------------------------------------------------------
    # SPPMI with Context Distribution Smoothing
    # ------------------------------------------------------------------
    total_mass = float(W.sum())
    p_i = np.asarray(W.sum(axis=1)).flatten() / max(total_mass, 1e-12)
    p_t = np.asarray(W.sum(axis=0)).flatten() / max(total_mass, 1e-12)
    log_k = math.log(sppmi_shift)

    W_coo = W.tocoo()
    sppmi_vals = []
    for i, j, w in zip(W_coo.row, W_coo.col, W_coo.data):
        p_ij = w / total_mass
        if p_ij <= 0 or p_i[i] <= 0 or p_t[j] <= 0:
            sppmi_vals.append(0.0)
        else:
            val = math.log(p_ij) - math.log(p_i[i]) - alpha_cds * math.log(p_t[j]) - log_k
            sppmi_vals.append(val if val > 0 else 0.0)
    S = sparse.csr_matrix((sppmi_vals, (W_coo.row, W_coo.col)), shape=(N_train, V_kept), dtype=np.float64)

    # ------------------------------------------------------------------
    # Truncated SVD and normalization
    # ------------------------------------------------------------------
    k = min(svd_dim, max(min(N_train, V_kept) - 1, 1))
    svd = TruncatedSVD(n_components=k, random_state=42)
    G = svd.fit_transform(S)
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    G_norm = (G / norms).astype("float32")

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    os.makedirs("ml/data", exist_ok=True)
    save_numpy(G_norm, "ml/data/tag_game_emb.npy")
    save_torch(svd, "ml/data/tag_svd.pkl")
    save_numpy(app_ids, "ml/data/tag_app_ids.npy")
    save_numpy(idf_kept, "ml/data/tag_idf.npy")
    sparse.save_npz("ml/data/tag_pair_pmi_plus.npz", pmi_matrix.tocsr())
    params = {
        "svd_dim": int(svd_dim),
        "R_max": int(R_max),
        "w_min": float(w_min),
        "alpha": float(alpha),
        "min_tag_count": int(min_tag_count),
        "min_tag_df": int(min_tag_df),
        "beta": float(beta),
        "beta_max": float(beta_max),
        "m_pair": int(m_pair),
        "alpha_cds": float(alpha_cds),
        "sppmi_shift": float(sppmi_shift),
    }
    save_json(params, "ml/data/tag_params.json")
    global_tag_mean = G_norm.mean(axis=0)
    save_numpy(global_tag_mean, "ml/data/tag_global_mean.npy")
    print(f"✅ Chunk 3 tag embeddings saved (dim={k}, games={N_train}, tags={V_kept})")
