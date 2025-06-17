import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .utils import load_json, save_numpy, save_torch


def run_tag_pca(games_df: pd.DataFrame) -> None:
    tag_id_map = load_json('ml/data/tag_id_map.json')
    V = len(tag_id_map)
    app_ids = []
    T_raw_all = np.zeros((len(games_df), V), dtype=float)

    for row_idx, row in games_df.iterrows():
        tags = row['tags'] or []
        for tag_id, rank in tags:
            idx = tag_id_map[str(tag_id)]
            weight = (21 - rank) / 20.0
            T_raw_all[row_idx, idx] = weight
        app_ids.append(row['app_id'])

    pca = PCA(n_components=256)
    T_pca = pca.fit_transform(T_raw_all)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    if cumsum[255] < 0.90:
        raise RuntimeError(f'Tag-PCA covers only {cumsum[255]*100:.1f}% variance')

    norms = np.linalg.norm(T_pca, axis=1, keepdims=True)
    norms[norms==0] = 1e-6
    T_pca_norm = T_pca / norms

    os.makedirs('data', exist_ok=True)
    save_numpy(T_pca_norm.astype('float32'), 'data/T_pca_norm.npy')
    save_torch(pca, 'data/pca_tags.pkl')
    save_numpy(np.array(app_ids), 'data/tag_pca_app_ids.npy')
    global_tag_mean = T_pca_norm.mean(axis=0)
    save_numpy(global_tag_mean, 'data/global_tag_mean.npy')
    print('✅ Chunk 3 tag PCA saved')

