import os
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from .utils import load_numpy, build_id_to_index_map, get_current_utc_date

_T_PCA_NORM = None
_TAG_APP_ID_TO_INDEX = None
_STRUCTURED_OK = None
_GLOBAL_TAG_MEAN = None


def compute_user_meta_raw(
    user_id: int, interactions_df: pd.DataFrame, data_dir: str | None = None
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return tag profile, recency and concatenated user metadata.

    On the first invocation the necessary tag PCA matrices and lookup maps
    are loaded from ``data_dir`` and cached in module-level variables to
    avoid repeated disk I/O. The returned tuple contains:

    * ``user_tag_profile`` – averaged tag vector for the user's owned games
    * ``recency`` – count of recently played or wishlisted games
    * ``user_meta_raw`` – concatenation of the two above
    """

    if data_dir is None:
        data_dir = os.getenv("ML_DATA_DIR", "ml/data")

    global _T_PCA_NORM, _TAG_APP_ID_TO_INDEX, _STRUCTURED_OK, _GLOBAL_TAG_MEAN
    if _T_PCA_NORM is None:
        _T_PCA_NORM = load_numpy(os.path.join(data_dir, 'T_pca_norm.npy'))
        tag_app_ids = load_numpy(os.path.join(data_dir, 'tag_pca_app_ids.npy')).tolist()
        _TAG_APP_ID_TO_INDEX = build_id_to_index_map([int(a) for a in tag_app_ids])
        try:
            with open(os.path.join(data_dir, 'appid_to_rowidx.json'), 'r', encoding='utf-8') as f:
                _STRUCTURED_OK = set(int(k) for k in json.load(f).keys())
        except Exception:
            _STRUCTURED_OK = None
        _GLOBAL_TAG_MEAN = load_numpy(os.path.join(data_dir, 'global_tag_mean.npy'))

    T_pca_norm = _T_PCA_NORM
    tag_app_id_to_index = _TAG_APP_ID_TO_INDEX
    structured_ok = _STRUCTURED_OK
    global_tag_mean = _GLOBAL_TAG_MEAN
    now = get_current_utc_date()
    owned = interactions_df[(interactions_df['user_id'] == user_id) & (
        (interactions_df['playtime_forever'] > 0)
        | (interactions_df['playtime_2weeks'] > 0)
    )]
    if not owned.empty:
        vectors: list[np.ndarray] = []
        for app in owned['app_id'].unique():
            a = int(app)
            if structured_ok is not None and a not in structured_ok:
                continue
            idx = tag_app_id_to_index.get(a)
            if idx is not None and idx < len(T_pca_norm):
                vectors.append(T_pca_norm[int(idx)])
        if vectors:
            user_tag_profile = np.mean(
                np.stack(vectors, axis=0), axis=0, dtype=np.float32
            )
        else:
            user_tag_profile = global_tag_mean.astype(np.float32).copy()
    else:
        user_tag_profile = global_tag_mean.astype(np.float32).copy()

    recent_cut = now - timedelta(days=30)
    recent_play = interactions_df[
        (interactions_df['user_id']==user_id) & (interactions_df['playtime_2weeks']>0)
    ]
    recent_wish = interactions_df[
        (interactions_df['user_id']==user_id) & (interactions_df['date_added_to_wishlist']>=recent_cut)
    ]
    recency = float(len(recent_play) + len(recent_wish))

    user_meta_raw = np.concatenate(
        [user_tag_profile, np.array([recency], dtype=np.float32)], axis=0
    )
    return user_tag_profile, recency, user_meta_raw

