import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from .utils import load_json, save_numpy, save_torch


def run_tag_pca(games_df: pd.DataFrame) -> None:
    tag_id_map = load_json('ml/data/tag_id_map.json')
    V = len(tag_id_map)
    N = len(games_df)
    app_ids = []

    # Compute document frequency for each tag
    df_counts = np.zeros(V, dtype=int)
    for tags in games_df['tags']:
        if not tags:
            continue
        seen = set()
        for tag_id, _ in tags:
            key = str(tag_id)
            if key in tag_id_map:
                seen.add(tag_id_map[key])
        for idx in seen:
            df_counts[idx] += 1

    # Smooth IDF as in sklearn: log((N + 1) / (df + 1)) + 1
    idf = np.log((N + 1) / (df_counts + 1)) + 1.0

    T_raw_all = np.zeros((N, V), dtype=float)

    for row_idx, row in games_df.iterrows():
        tags = row['tags'] or []
        for tag_id, rank in tags:
            key = str(tag_id)
            if key not in tag_id_map:
                # Skip tags that are missing from the tag_id_map
                continue
            idx = tag_id_map[key]
            tf = (21 - rank) / 20.0
            T_raw_all[row_idx, idx] = tf * idf[idx]
        app_ids.append(row['app_id'])

    pca = PCA(n_components=256)
    T_pca = pca.fit_transform(T_raw_all)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    if cumsum[255] < 0.90:
        raise RuntimeError(f'Tag-PCA covers only {cumsum[255]*100:.1f}% variance')

    norms = np.linalg.norm(T_pca, axis=1, keepdims=True)
    norms[norms==0] = 1e-6
    T_pca_norm = T_pca / norms

    os.makedirs('ml/data', exist_ok=True)
    save_numpy(T_pca_norm.astype('float32'), 'ml/data/T_pca_norm.npy')
    save_torch(pca, 'ml/data/pca_tags.pkl')
    save_numpy(np.array(app_ids), 'ml/data/tag_pca_app_ids.npy')
    global_tag_mean = T_pca_norm.mean(axis=0)
    save_numpy(global_tag_mean, 'ml/data/global_tag_mean.npy')
    print('✅ Chunk 3 tag PCA saved')

