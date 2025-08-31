import os
import json
import numpy as np
import faiss
from .utils import load_numpy, save_json


def build_faiss_indices() -> None:
    # Load meta embeddings and alignment mapping
    meta_embs = load_numpy('ml/data/item_meta_embs.npy').astype('float32')
    with open('ml/data/app_id_to_meta_index.json', 'r', encoding='utf-8') as f:
        app_id_to_meta_index = {int(k): int(v) for k, v in json.load(f).items()}

    # Build ordered app_id list by meta index
    max_idx = max(app_id_to_meta_index.values()) if app_id_to_meta_index else -1
    app_ids_ordered = [None] * (max_idx + 1)
    for aid, idx in app_id_to_meta_index.items():
        if 0 <= idx <= max_idx:
            app_ids_ordered[idx] = int(aid)

    # Load tag PCA and align to the same app_id ordering
    T_pca_all = load_numpy('ml/data/T_pca_norm.npy').astype('float32')
    tag_app_ids = load_numpy('ml/data/tag_pca_app_ids.npy')
    tag_idx_map = {int(a): i for i, a in enumerate(tag_app_ids)}
    T_f = []
    kept_app_ids = []
    for aid in app_ids_ordered:
        if aid is None:
            continue
        t_idx = tag_idx_map.get(int(aid))
        if t_idx is None:
            continue
        T_f.append(T_pca_all[t_idx])
        kept_app_ids.append(int(aid))
    T_final = np.array(T_f, dtype='float32')

    # Build tags FAISS index
    index_tags = faiss.IndexFlatIP(T_final.shape[1])
    index_tags.add(T_final)
    faiss.write_index(index_tags, 'ml/data/faiss_tags.index')
    save_json(kept_app_ids, 'ml/data/index_to_app_id_tags.json')

    # Normalize and build meta FAISS index (match kept_app_ids length)
    meta_kept = []
    for aid in kept_app_ids:
        idx = app_id_to_meta_index.get(int(aid))
        if idx is not None and idx < len(meta_embs):
            meta_kept.append(meta_embs[idx])
    meta_kept = np.array(meta_kept, dtype='float32')
    norms = np.linalg.norm(meta_kept, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    meta_norm = meta_kept / norms
    index_meta = faiss.IndexFlatIP(meta_norm.shape[1])
    index_meta.add(meta_norm.astype('float32'))
    faiss.write_index(index_meta, 'ml/data/faiss_meta.index')
    save_json(kept_app_ids, 'ml/data/index_to_app_id_meta.json')
    print('✅ Chunk 6 FAISS indices saved')

