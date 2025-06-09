import os
import numpy as np
import pandas as pd
from datetime import datetime

from .utils import save_numpy, load_numpy


def structured_transform(games_df: pd.DataFrame, today: datetime) -> None:
    all_structured = []
    app_ids = []

    for _, row in games_df.iterrows():
        if pd.isna(row['price_usd']):
            continue
        pr_log = np.log1p(row['positive_review_count'])
        nr_log = np.log1p(row['negative_review_count'])
        price_log = np.log1p(row['price_usd'])
        discount_val = row['discount_usd'] if not pd.isna(row['discount_usd']) else row['price_usd']
        discount_log = np.log1p(discount_val)
        if pd.isna(row['release_date']):
            days_since = 0
        else:
            days_since = (today.date() - row['release_date']).days
            days_since = max(days_since, 0)
        if days_since < 30:
            bucket = [1,0,0,0,0]
        elif days_since <= 179:
            bucket = [0,1,0,0,0]
        elif days_since <= 364:
            bucket = [0,0,1,0,0]
        elif days_since <= 729:
            bucket = [0,0,0,1,0]
        else:
            bucket = [0,0,0,0,1]
        vect = [pr_log, nr_log, price_log, discount_log, days_since] + bucket
        all_structured.append(vect)
        app_ids.append(row['app_id'])

    arr = np.array(all_structured, dtype=float)
    structured_min = arr.min(axis=0)
    structured_max = arr.max(axis=0)

    os.makedirs('data', exist_ok=True)
    save_numpy(arr, 'data/structured_raw.npy')
    save_numpy(structured_min, 'data/structured_min.npy')
    save_numpy(structured_max, 'data/structured_max.npy')
    save_numpy(np.array(app_ids), 'data/structured_app_ids.npy')
    print('✅ Chunk 2 structured features saved')

