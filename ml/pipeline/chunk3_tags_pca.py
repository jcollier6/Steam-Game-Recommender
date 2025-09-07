import os
import math
import json
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score
from scipy import sparse

from .utils import load_json, save_numpy, save_torch, save_json


def _rank_weight(r: int, Rmax: int, w_min: float, alpha: float) -> float:
    x = 1.0 - (r - 1) / max(Rmax - 1, 1)
    return w_min + (1.0 - w_min) * (x ** alpha)


def run_tag_pca(
    games_df: pd.DataFrame,
    svd_dim: int = 192,
    svd_k_list: Optional[Sequence[int]] = None,
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

    Optionally sweeps over several ``k`` values when performing Truncated SVD
    to choose a suitable embedding dimensionality. The function emits
    progress metrics – tag coverage, training matrix density, and SVD
    explained variance – to help monitor embedding quality. IDF weighting,
    co-tag PMI boosts and CDS-smoothed SPPMI produce dense embeddings whose
    artifacts are written to ``ml/data``.
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
    kept_tags_fraction = V_kept / max(V, 1)
    print(
        f"[chunk3] Keeping {V_kept}/{V} tags after min_df>={int(min_tag_df)} filter",
        flush=True,
    )
    metrics: dict[str, object] = {}
    metrics["matrix_health"] = {"kept_tags_fraction": float(kept_tags_fraction)}

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
    mean_tags = float(W.getnnz(axis=1).mean()) if N_train > 0 else 0.0
    metrics["matrix_health"]["avg_tags_per_game"] = float(mean_tags)
    print(
        f"[chunk3] Training matrix built for {N_train} games (avg tags/game={mean_tags:.2f})",
        flush=True,
    )
    game_tag_sets = [
        {str(tid) for tid, _ in (row.get("tags") or [])[:R_max]}
        for _, row in games_train.iterrows()
    ]

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
    density = S.nnz / max(S.shape[0] * S.shape[1], 1)
    metrics["matrix_health"]["sppmi_density"] = float(density)
    print(
        f"[chunk3] SPPMI matrix {S.shape} with {S.nnz} non-zeros (density={density:.6f})",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Truncated SVD sweep and normalization
    # ------------------------------------------------------------------
    k_candidates = sorted(set(svd_k_list or [svd_dim]))
    sweep_results: dict[str, dict[str, float]] = {}
    best_k = None
    best_svd: Optional[TruncatedSVD] = None
    best_G = None
    for k in k_candidates:
        k_eff = min(k, max(min(N_train, V_kept) - 1, 1))
        svd_tmp = TruncatedSVD(n_components=k_eff, random_state=42)
        G_tmp = svd_tmp.fit_transform(S)
        evr = svd_tmp.explained_variance_ratio_
        cum_evr = float(evr.sum())
        comp1 = float(evr[0]) if evr.size else 0.0
        sweep_results[str(k_eff)] = {
            "cumulative_evr": cum_evr,
            "comp1": comp1,
        }
        if best_k is None and cum_evr >= 0.80 and comp1 <= 0.10:
            best_k = k_eff
            best_svd = svd_tmp
            best_G = G_tmp
    if best_k is None:
        best_k = k_eff
        best_svd = svd_tmp
        best_G = G_tmp

    svd = best_svd
    G = best_G
    norms = np.linalg.norm(G, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    G_norm = (G / norms).astype("float32")
    exp_var = float(svd.explained_variance_ratio_.sum())
    top5 = np.round(svd.explained_variance_ratio_[:5], 4).tolist()
    comp1 = float(svd.explained_variance_ratio_[0]) if svd.explained_variance_ratio_.size else 0.0
    comp2 = float(svd.explained_variance_ratio_[1]) if svd.explained_variance_ratio_.size > 1 else 0.0
    print(
        f"[chunk3] SVD explained variance {exp_var:.2%} (first 5 comps: {top5})",
        flush=True,
    )
    if comp1 > 0.10:
        print(
            f"[chunk3] ⚠️ First component explains {comp1:.2%} variance (>10%)",
            flush=True,
        )
    if comp1 + comp2 > 0.18:
        print(
            f"[chunk3] ⚠️ First two components explain {(comp1 + comp2):.2%} variance (>18%)",
            flush=True,
        )
    popularity_leak = exp_var >= 0.90 and comp1 > 0.12
    if popularity_leak:
        print(
            f"[chunk3] ⚠️ Potential popularity leakage (EVR≥90%, comp1>{comp1:.2%})",
            flush=True,
        )
    metrics["svd_dim_used"] = int(best_k)
    metrics["explained_variance"] = float(exp_var)
    metrics["top5_shares"] = [float(x) for x in top5]
    metrics["sweep_results"] = sweep_results
    metrics["spectral_warnings"] = []
    if comp1 > 0.10:
        metrics["spectral_warnings"].append("comp1_gt_10pct")
    if comp1 + comp2 > 0.18:
        metrics["spectral_warnings"].append("comp1_plus_comp2_gt_18pct")
    metrics["popularity_leakage"] = bool(popularity_leak)

    tag_vecs = svd.components_.T
    tag_norms = np.linalg.norm(tag_vecs, axis=1, keepdims=True)
    tag_norms[tag_norms == 0] = 1e-6
    T_norm = (tag_vecs / tag_norms).astype("float32")

    # ------------------------------------------------------------------
    # Nearest-neighbor coherence
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    sample_size = min(200, N_train)
    if sample_size > 0:
        sample_idx = rng.choice(N_train, size=sample_size, replace=False)
        jaccards = []
        for idx in sample_idx:
            sims = G_norm[idx] @ G_norm.T
            sims[idx] = -1.0
            neigh_idx = np.argpartition(-sims, 10)[:10]
            neigh_idx = neigh_idx[np.argsort(-sims[neigh_idx])]
            A = game_tag_sets[idx]
            for j in neigh_idx:
                B = game_tag_sets[j]
                inter = len(A & B)
                union = len(A | B)
                jaccards.append(inter / union if union > 0 else 0.0)
        mean_j = float(np.mean(jaccards)) if jaccards else 0.0
        p50_j = float(np.median(jaccards)) if jaccards else 0.0
    else:
        mean_j = 0.0
        p50_j = 0.0
    metrics["nn_coherence"] = {"mean_jaccard": mean_j, "p50_jaccard": p50_j}

    # ------------------------------------------------------------------
    # Held-out co-tag edge prediction
    # ------------------------------------------------------------------
    pairs = list(pair_counts.keys())
    if pairs:
        rng.shuffle(pairs)
        n_test = max(1, int(0.1 * len(pairs)))
        test_pairs = pairs[:n_test]
        existing = set(pairs)
        pos_sims = [float(T_norm[a] @ T_norm[b]) for a, b in test_pairs]
        neg_sims = []
        while len(neg_sims) < n_test:
            a = int(rng.integers(V_kept))
            b = int(rng.integers(V_kept))
            if a == b:
                continue
            p = (a, b) if a < b else (b, a)
            if p in existing:
                continue
            neg_sims.append(float(T_norm[p[0]] @ T_norm[p[1]]))
        y_true = np.array([1] * n_test + [0] * n_test)
        y_scores = np.array(pos_sims + neg_sims)
        edge_ap = float(average_precision_score(y_true, y_scores))
    else:
        edge_ap = 0.0
    metrics["edge_prediction_ap"] = edge_ap

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
    os.makedirs("ml/metrics", exist_ok=True)
    save_json(metrics, "ml/metrics/tag_metrics.json")
    print(
        f"✅ Chunk 3 tag embeddings saved (dim={k}, games={N_train}, tags={V_kept}, variance={exp_var:.2%})"
    )
