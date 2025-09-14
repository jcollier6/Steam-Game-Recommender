import os
import json
import numpy as np
import pandas as pd
import torch
import faiss

from .utils import load_numpy, load_json, load_torch
from .chunk9_model import UserMetaFC, ScoreMLP
from .chunk8_embeddings import EmbeddingTables
from .chunk7_user_profile import compute_user_meta_raw


def _load_structured_schema(data_dir: str):
    schema_path = os.path.join(data_dir, 'structured_schema.json')
    if not os.path.exists(schema_path):
        return None
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _filter_unique(candidates, K, exclude_ids):
    seen = set(int(x) for x in exclude_ids)
    results = []
    for c in candidates:
        app_id = int(c['app_id']) if isinstance(c, dict) else int(c)
        if app_id in seen:
            continue
        seen.add(app_id)
        results.append(c)
        if len(results) >= K:
            break
    return results


def recommend(
    user_id: int,
    interactions_df,
    K: int = 10,
    data_dir: str | None = None,
    user_library_df: pd.DataFrame | None = None,
):
    # Load structured scaling schema for reproducibility logs and potential transforms
    schema = _load_structured_schema(data_dir or os.getenv('ML_DATA_DIR', 'ml/data'))
    if schema is not None:
        snap = schema.get('snapshot_date')
        cols = schema.get('columns', [])
        print(f"structured schema loaded: snapshot_date={snap}, columns={len(cols)}", flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if data_dir is None:
        data_dir = os.getenv('ML_DATA_DIR', 'ml/data')
    umeta_state = load_torch(os.path.join(data_dir, 'user_meta_fc.pth'), device)
    user_meta_dim = umeta_state['fc1.weight'].shape[1]
    user_meta_fc = UserMetaFC(user_meta_dim).to(device)
    user_meta_fc.load_state_dict(umeta_state)
    score_mlp = ScoreMLP().to(device)
    score_mlp.load_state_dict(load_torch(os.path.join(data_dir, 'score_mlp.pth'), device))
    tables = EmbeddingTables(0, 0)
    tables.load(os.path.join(data_dir, 'emb_tables'), device=device)
    map_path = os.path.join(data_dir, 'user_id_to_index.json')
    user_id_map = load_json(map_path) if os.path.exists(map_path) else {}
    item_meta_np = load_numpy(os.path.join(data_dir, 'item_meta_embs.npy')).astype(np.float32)
    item_meta_np = np.vstack([
        np.zeros((1, item_meta_np.shape[1]), dtype=item_meta_np.dtype),
        item_meta_np,
    ])
    item_meta_embs = item_meta_np
    index_meta = faiss.read_index(os.path.join(data_dir, 'faiss_meta.index'))
    index_to_app = load_json(os.path.join(data_dir, 'index_to_app_id_meta.json'))
    app_id_to_index = load_json(os.path.join(data_dir, 'app_id_to_meta_index.json'))

    def _app_idx(app_id: int) -> int:
        idx = app_id_to_index.get(str(app_id))
        return idx + 1 if idx is not None else 0

    def _user_idx(uid: int) -> int:
        idx = user_id_map.get(str(uid))
        return idx + 1 if idx is not None else 0

    user_df = interactions_df[interactions_df['user_id'] == user_id]
    if user_df.empty and user_library_df is not None:
        lib = user_library_df.copy()
        if 'user_id' not in lib.columns:
            lib['user_id'] = user_id
        required_cols = [
            'playtime_forever',
            'playtime_2weeks',
            'wishlisted',
            'wishlist_priority',
            'date_added_to_wishlist',
        ]
        for col in required_cols:
            if col not in lib.columns:
                if col == 'wishlisted':
                    lib[col] = False
                else:
                    lib[col] = 0
        pos_mask = (
            (lib['playtime_forever'] > 0)
            | (lib['playtime_2weeks'] > 0)
            | (lib['wishlisted'])
        )
        pos_count = int(pos_mask.sum())
        if pos_count >= 20:
            user_df = lib
        else:
            popular = load_json(os.path.join(data_dir, 'global_popular_games.json'))
            return _filter_unique(popular, K, lib['app_id'].astype(int).tolist())
    elif user_df.empty:
        popular = load_json(os.path.join(data_dir, 'global_popular_games.json'))
        return _filter_unique(popular, K, [])

    pos_mask_df = (
        (user_df['playtime_forever'] > 0)
        | (user_df['playtime_2weeks'] > 0)
        | (user_df['wishlisted'])
    )
    if pos_mask_df.sum() < 20:
        popular = load_json(os.path.join(data_dir, 'global_popular_games.json'))
        return _filter_unique(popular, K, user_df['app_id'].astype(int).tolist())

    owned_ids = user_df['app_id'].astype(int).tolist()
    exclude_ids: set[int] = set(owned_ids)

    _, _, user_meta_raw = compute_user_meta_raw(user_id, user_df, data_dir)
    with torch.no_grad():
        user_meta_emb = user_meta_fc(torch.tensor(user_meta_raw).float().unsqueeze(0).to(device))
    user_meta_norm = user_meta_emb / user_meta_emb.norm(dim=1, keepdim=True)
    query_vec = user_meta_norm.cpu().numpy().astype('float32')

    # Query FAISS in batches until we gather at least K unique, unowned items
    batch_size = max(500, K * 10)
    candidate_ids: list[int] = []
    seen_cands: set[int] = set(exclude_ids)
    search_k = 0
    while len(candidate_ids) < K and search_k < index_meta.ntotal:
        search_k = min(search_k + batch_size, index_meta.ntotal)
        _, cand_idxs = index_meta.search(query_vec, search_k)
        start = max(0, search_k - batch_size)
        new_batch = []
        for idx in cand_idxs[0][start:search_k]:
            app_id = index_to_app[idx] if isinstance(index_to_app, list) else index_to_app[str(idx)]
            if app_id in seen_cands:
                continue
            seen_cands.add(app_id)
            candidate_ids.append(app_id)
            new_batch.append(app_id)
            if len(candidate_ids) >= K:
                break
        print(f"searched {search_k}, new apps {new_batch}")


    scores = []
    with torch.no_grad():
        idx_u = _user_idx(user_id)
        if idx_u != 0 and idx_u < tables.user_emb.num_embeddings:
            u_emb = tables.user_emb(torch.tensor([idx_u], dtype=torch.long, device=device))
        else:
            pos_apps = user_df[
                (user_df['playtime_forever'] > 0)
                | (user_df['playtime_2weeks'] > 0)
                | (user_df['wishlisted'])
            ]['app_id'].unique()
            item_indices = []
            for a in pos_apps:
                idx = _app_idx(a)
                if idx != 0 and idx < tables.item_emb.num_embeddings:
                    item_indices.append(idx)
            if item_indices:
                emb = tables.item_emb(torch.tensor(item_indices, device=device))
                u_emb = emb.mean(dim=0, keepdim=True)
            else:
                u_emb = tables.mean_user_emb.unsqueeze(0).to(device)
        for app_id in candidate_ids:
            idx = _app_idx(app_id)
            if idx != 0 and idx < tables.item_emb.num_embeddings:
                i_emb = tables.item_emb(torch.tensor([idx], device=device))
            else:
                i_emb = tables.mean_item_emb.unsqueeze(0).to(device)
            if idx != 0 and idx < len(item_meta_embs):
                i_meta = (
                    torch.from_numpy(item_meta_embs[idx])
                    .float()
                    .unsqueeze(0)
                    .to(device)
                )
            else:
                i_meta = torch.zeros((1, item_meta_embs.shape[1]), device=device)
            feat = torch.cat([u_emb, user_meta_emb, i_emb, i_meta], dim=1)
            s = score_mlp(feat)
            scores.append(s.item())

    ranked = [x for _, x in sorted(zip(scores, candidate_ids), key=lambda t: t[0], reverse=True)]

    # Remove duplicates and items the user already owns.
    recs = _filter_unique(ranked, K, exclude_ids)

    # If FAISS produced too many duplicates, backfill with popular titles to
    # still return K unique recommendations.
    if len(recs) < K:
        popular = load_json(os.path.join(data_dir, 'global_popular_games.json'))
        exclude = owned_ids + [int(r['app_id']) if isinstance(r, dict) else int(r) for r in recs]
        recs.extend(_filter_unique(popular, K - len(recs), exclude))

    return recs
