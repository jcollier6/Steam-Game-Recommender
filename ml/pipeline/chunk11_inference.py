import os
import numpy as np
import pandas as pd
import torch
import faiss

from .utils import load_numpy, load_json, load_torch
from .chunk9_model import UserMetaFC, ScoreMLP
from .chunk8_embeddings import EmbeddingTables
from .chunk7_user_profile import compute_user_meta_raw


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if data_dir is None:
        data_dir = os.getenv('ML_DATA_DIR', 'ml/data')
    tag_dim = load_numpy(os.path.join(data_dir, 'T_pca_norm.npy')).shape[1]
    user_meta_fc = UserMetaFC(tag_dim + 1).to(device)
    score_mlp = ScoreMLP().to(device)
    user_meta_fc.load_state_dict(load_torch(os.path.join(data_dir, 'user_meta_fc.pth'), device))
    score_mlp.load_state_dict(load_torch(os.path.join(data_dir, 'score_mlp.pth'), device))
    tables = EmbeddingTables(0, 0)
    tables.load(os.path.join(data_dir, 'emb_tables'), device=device)
    map_path = os.path.join(data_dir, 'user_id_to_index.json')
    user_id_map = load_json(map_path) if os.path.exists(map_path) else {}
    item_meta_embs = load_numpy(os.path.join(data_dir, 'item_meta_embs.npy'))
    index_meta = faiss.read_index(os.path.join(data_dir, 'faiss_meta.index'))
    index_to_app = load_json(os.path.join(data_dir, 'index_to_app_id_meta.json'))
    app_id_to_index = load_json(os.path.join(data_dir, 'app_id_to_meta_index.json'))

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

    _, _, user_meta_raw = compute_user_meta_raw(user_id, user_df, data_dir)
    with torch.no_grad():
        user_meta_emb = user_meta_fc(torch.tensor(user_meta_raw).float().unsqueeze(0).to(device))
    user_meta_norm = user_meta_emb / user_meta_emb.norm(dim=1, keepdim=True)
    query_vec = user_meta_norm.cpu().numpy().astype('float32')
    _, cand_idxs = index_meta.search(query_vec, 500)
    candidate_ids = [index_to_app[i] if isinstance(index_to_app, list) else index_to_app[str(i)] for i in cand_idxs[0]]

    scores = []
    with torch.no_grad():
        idx_u = user_id_map.get(str(user_id))
        if idx_u is not None and idx_u < tables.user_emb.num_embeddings:
            u_emb = tables.user_emb(torch.tensor([idx_u], device=device))
        else:
            pos_apps = user_df[
                (user_df['playtime_forever'] > 0)
                | (user_df['playtime_2weeks'] > 0)
                | (user_df['wishlisted'])
            ]['app_id'].unique()
            item_indices = [
                app_id_to_index.get(str(a))
                for a in pos_apps
                if app_id_to_index.get(str(a)) is not None
                and app_id_to_index.get(str(a)) < tables.item_emb.num_embeddings
            ]
            if item_indices:
                emb = tables.item_emb(torch.tensor(item_indices, device=device))
                u_emb = emb.mean(dim=0, keepdim=True)
            else:
                u_emb = tables.mean_user_emb.unsqueeze(0).to(device)
        for app_id in candidate_ids:
            idx = app_id_to_index.get(str(app_id), None)
            if idx is not None and idx < tables.item_emb.num_embeddings:
                i_emb = tables.item_emb(torch.tensor([idx], device=device))
            else:
                i_emb = tables.mean_item_emb.unsqueeze(0).to(device)
            if idx is not None and idx < len(item_meta_embs):
                i_meta = torch.from_numpy(item_meta_embs[idx]).float().unsqueeze(0).to(device)
            else:
                i_meta = torch.zeros((1, item_meta_embs.shape[1]), device=device)
            feat = torch.cat([u_emb, user_meta_emb, i_emb, i_meta], dim=1)
            s = score_mlp(feat)
            scores.append(s.item())

    ranked = [x for _, x in sorted(zip(scores, candidate_ids), key=lambda t: t[0], reverse=True)]
    return _filter_unique(ranked, K, user_df['app_id'].astype(int).tolist())
