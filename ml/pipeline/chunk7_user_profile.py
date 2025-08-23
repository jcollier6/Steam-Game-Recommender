import os
import numpy as np
import pandas as pd
import math
from datetime import timedelta

from .utils import (
    load_numpy,
    compute_alpha,
    build_id_to_index_map,
    get_current_utc_date,
)

ALPHA = compute_alpha(30)


def compute_user_meta_raw(user_id: int, interactions_df: pd.DataFrame, data_dir: str | None = None):
    if data_dir is None:
        data_dir = os.getenv("ML_DATA_DIR", "ml/data")
    T_pca_norm = load_numpy(os.path.join(data_dir, 'T_pca_norm.npy'))
    app_ids = load_numpy(os.path.join(data_dir, 'structured_app_ids.npy')).tolist()
    app_id_to_index = build_id_to_index_map(app_ids)
    global_tag_mean = load_numpy(os.path.join(data_dir, 'global_tag_mean.npy'))
    now = get_current_utc_date()
    owned = interactions_df[(interactions_df['user_id']==user_id) & ((interactions_df['playtime_forever']>0) | (interactions_df['playtime_2weeks']>0))]
    if not owned.empty:
        vectors = []
        for app in owned['app_id'].unique():
            idx = app_id_to_index.get(app)
            if idx is not None:
                vectors.append(T_pca_norm[int(idx)])
        if vectors:
            user_tag_profile = np.mean(vectors, axis=0)
        else:
            user_tag_profile = global_tag_mean.copy()
    else:
        user_tag_profile = global_tag_mean.copy()

    recent_cut = now - timedelta(days=30)
    recent_play = interactions_df[
        (interactions_df['user_id']==user_id) & (interactions_df['playtime_2weeks']>0)
    ]
    recent_wish = interactions_df[
        (interactions_df['user_id']==user_id) & (interactions_df['date_added_to_wishlist']>=recent_cut)
    ]
    recency = float(len(recent_play) + len(recent_wish))

    user_meta_raw = np.concatenate([user_tag_profile, [recency]], axis=0)
    return user_tag_profile, recency, user_meta_raw

