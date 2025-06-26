import os
import numpy as np
import faiss
from .utils import load_numpy, save_json


def build_faiss_indices() -> None:
    T_pca_norm = load_numpy('ml/data/T_pca_norm.npy').astype('float32')
    meta_embs = load_numpy('ml/data/item_meta_embs.npy').astype('float32')
    app_ids = load_numpy('ml/data/structured_app_ids.npy').tolist()
    index_tags = faiss.IndexFlatIP(T_pca_norm.shape[1])
    index_tags.add(T_pca_norm)
    faiss.write_index(index_tags, 'ml/data/faiss_tags.index')
    save_json(app_ids, 'ml/data/index_to_app_id_tags.json')

    norms = np.linalg.norm(meta_embs, axis=1, keepdims=True)
    norms[norms==0] = 1e-6
    meta_norm = meta_embs / norms
    index_meta = faiss.IndexFlatIP(meta_norm.shape[1])
    index_meta.add(meta_norm.astype('float32'))
    faiss.write_index(index_meta, 'ml/data/faiss_meta.index')
    save_json(app_ids, 'ml/data/index_to_app_id_meta.json')
    print('✅ Chunk 6 FAISS indices saved')

