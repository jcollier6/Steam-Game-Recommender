import os
import json
import numpy as np
import faiss

from .utils import load_numpy, save_json


def build_faiss_indices() -> None:
    """Build FAISS indices for tag and item metadata embeddings.

    The function aligns tag-based PCA embeddings and metadata-based embeddings
    on a common ``app_id`` ordering, then writes cosine-similarity FAISS indices
    (inner-product over L2-normalized vectors) along with mappings from index
    positions back to ``app_id`` values.
    """

    os.makedirs('ml/data', exist_ok=True)

    # Load meta embeddings and alignment mapping
    meta_embs = load_numpy('ml/data/item_meta_embs.npy').astype(np.float32)
    with open('ml/data/app_id_to_meta_index.json', 'r', encoding='utf-8') as f:
        app_id_to_meta_index = {int(k): int(v) for k, v in json.load(f).items()}

    meta_app_ids = np.array(list(app_id_to_meta_index.keys()), dtype=int)
    meta_indices = np.array(list(app_id_to_meta_index.values()), dtype=int)

    # Load tag PCA embeddings and align using intersection of app_ids
    T_pca_all = load_numpy('ml/data/T_pca_norm.npy').astype(np.float32)
    tag_app_ids = load_numpy('ml/data/tag_pca_app_ids.npy').astype(int)
    common_app_ids, tag_idx, meta_pos = np.intersect1d(
        tag_app_ids, meta_app_ids, return_indices=True
    )
    T_final = T_pca_all[tag_idx]
    kept_app_ids = common_app_ids.astype(int).tolist()

    # Build tags FAISS index
    index_tags = faiss.IndexFlatIP(T_final.shape[1])
    index_tags.add(T_final)
    faiss.write_index(index_tags, 'ml/data/faiss_tags.index')
    save_json(kept_app_ids, 'ml/data/index_to_app_id_tags.json')

    # Normalize and build metadata FAISS index
    meta_selected = meta_embs[meta_indices[meta_pos]]
    norms = np.linalg.norm(meta_selected, axis=1, keepdims=True)
    norms[norms == 0] = 1e-6
    meta_norm = meta_selected / norms
    index_meta = faiss.IndexFlatIP(meta_norm.shape[1])
    index_meta.add(meta_norm)
    if T_final.shape[0] != meta_norm.shape[0]:
        raise ValueError('Tag and meta arrays differ in length; alignment failed.')
    faiss.write_index(index_meta, 'ml/data/faiss_meta.index')
    save_json(kept_app_ids, 'ml/data/index_to_app_id_meta.json')

    print('✅ Chunk 6 FAISS indices saved')

