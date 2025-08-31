import os
import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import save_numpy, save_json


CONT_COLS_ORDER = [
    "log1p_pos",
    "log1p_neg",
    "log1p_total",
    "pos_ratio",
    "wilson_score",
    # price_num_scaled and log1p_days_scaled are also scaled continuous
    # but their names already include "_scaled" as required.
]

FINAL_COL_ORDER = [
    "log1p_pos",
    "log1p_neg",
    "log1p_total",
    "pos_ratio",
    "wilson_score",
    "is_free",
    "price_num_scaled",
    "log1p_days_scaled",
    "age_0_7",
    "age_7_30",
    "age_30_90",
    "age_90_180",
    "age_180_365",
    "age_365_730",
    "age_730_1825",
    "age_1825_plus",
]


def _wilson_lower_bound(pos: np.ndarray, total: np.ndarray, z: float = 1.96) -> np.ndarray:
    p = np.divide(pos, np.maximum(total, 1), where=total>=0)
    denom = 1 + (z**2)/np.maximum(total, 1)
    center = p + (z**2)/(2*np.maximum(total, 1))
    adj = z * np.sqrt((p*(1-p) + (z**2)/(4*np.maximum(total, 1))) / np.maximum(total, 1))
    lower = (center - adj) / denom
    lower = np.clip(lower, 0.0, 1.0)
    return lower


def _age_buckets(days: np.ndarray) -> np.ndarray:
    # Buckets: [0–7), [7–30), [30–90), [90–180), [180–365), [365–730), [730–1825), [1825+)
    edges = np.array([0, 7, 30, 90, 180, 365, 730, 1825])
    out = np.zeros((len(days), 8), dtype=np.int32)
    for i, d in enumerate(days):
        if d >= 1825:
            idx = 7
        elif d >= 730:
            idx = 6
        elif d >= 365:
            idx = 5
        elif d >= 180:
            idx = 4
        elif d >= 90:
            idx = 3
        elif d >= 30:
            idx = 2
        elif d >= 7:
            idx = 1
        else:
            idx = 0
        out[i, idx] = 1
    return out


def _minmax_scale(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if not np.isfinite(x_min):
        x_min = 0.0
    if not np.isfinite(x_max) or x_max == x_min:
        x_max = x_min + 1.0
    x_scaled = (x - x_min) / (x_max - x_min)
    x_scaled = np.clip(x_scaled, 0.0, 1.0)
    return x_scaled.astype(np.float32), x_min, x_max


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    # expected/actual are 1D arrays already scaled to [0,1]
    edges = np.linspace(0.0, 1.0, bins + 1)
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = e_counts / max(e_counts.sum(), 1)
    a_pct = a_counts / max(a_counts.sum(), 1)
    eps = 1e-8
    psi = np.sum((a_pct - e_pct) * np.log((a_pct + eps) / (e_pct + eps)))
    return float(psi)


def structured_transform(games_df: pd.DataFrame, today: datetime) -> None:
    # Required fields presence; drop invalid rows
    df = games_df.copy()
    before = len(df)
    reasons = {
        'missing_pos': 0, 'missing_neg': 0, 'missing_price': 0, 'missing_release': 0
    }
    def _num(s):
        return pd.to_numeric(s, errors='coerce')

    df['pos_reviews'] = _num(df.get('pos_reviews'))
    df['neg_reviews'] = _num(df.get('neg_reviews'))
    df['base_price_usd'] = _num(df.get('base_price_usd'))
    # Ensure datetimes
    df['release_date'] = pd.to_datetime(df.get('release_date'), errors='coerce').dt.tz_localize(None)
    df['snapshot_date'] = pd.to_datetime(df.get('snapshot_date'), errors='coerce').dt.tz_localize(None)

    m_pos = df['pos_reviews'].isna()
    m_neg = df['neg_reviews'].isna()
    m_price = df['base_price_usd'].isna()
    m_rel = df['release_date'].isna()
    reasons['missing_pos'] = int(m_pos.sum())
    reasons['missing_neg'] = int(m_neg.sum())
    reasons['missing_price'] = int(m_price.sum())
    reasons['missing_release'] = int(m_rel.sum())
    keep = ~(m_pos | m_neg | m_price | m_rel)
    df = df.loc[keep].reset_index(drop=True)
    after = len(df)
    print(f"Chunk2: rows before={before}, after={after}")
    print("Drop reasons:", {k:v for k,v in reasons.items() if v})

    # Compute base features
    pos = df['pos_reviews'].astype(float).to_numpy()
    neg = df['neg_reviews'].astype(float).to_numpy()
    total = pos + neg
    log1p_pos = np.log1p(pos)
    log1p_neg = np.log1p(neg)
    log1p_total = np.log1p(total)
    pos_ratio = pos / np.maximum(total, 1.0)
    wilson = _wilson_lower_bound(pos, total)

    price = df['base_price_usd'].astype(float).to_numpy()
    is_free = (price == 0).astype(np.int32)
    log1p_price = np.log1p(price)

    # Age features
    snap = df['snapshot_date']
    # If any snapshot_date is missing, fill with max available to keep determinism
    if snap.isna().any():
        snap = snap.fillna(snap.dropna().max())
    rel = df['release_date']
    days_since = (snap.dt.normalize() - rel.dt.normalize()).dt.days.to_numpy()
    days_since = np.maximum(days_since, 0)
    log1p_days = np.log1p(days_since)
    age_onehot = _age_buckets(days_since)

    # Scale continuous columns (min-max fit on this dataset)
    scaled = {}
    scaling: Dict[str, Dict[str, float]] = {}
    for name, arr in [
        ("log1p_pos", log1p_pos),
        ("log1p_neg", log1p_neg),
        ("log1p_total", log1p_total),
        ("pos_ratio", pos_ratio),
        ("wilson_score", wilson),
        ("price_num_scaled", log1p_price),
        ("log1p_days_scaled", log1p_days),
    ]:
        vals = arr.astype(np.float32)
        vals_s, vmin, vmax = _minmax_scale(vals)
        scaled[name] = vals_s
        scaling[name] = {"min": float(vmin), "max": float(vmax)}

    # Assemble final matrix in frozen order
    cols: List[np.ndarray] = [
        scaled["log1p_pos"],
        scaled["log1p_neg"],
        scaled["log1p_total"],
        scaled["pos_ratio"],
        scaled["wilson_score"],
        is_free.astype(np.float32),
        scaled["price_num_scaled"],
        scaled["log1p_days_scaled"],
        age_onehot[:, 0].astype(np.float32),
        age_onehot[:, 1].astype(np.float32),
        age_onehot[:, 2].astype(np.float32),
        age_onehot[:, 3].astype(np.float32),
        age_onehot[:, 4].astype(np.float32),
        age_onehot[:, 5].astype(np.float32),
        age_onehot[:, 6].astype(np.float32),
        age_onehot[:, 7].astype(np.float32),
    ]
    X = np.stack(cols, axis=1).astype(np.float32)

    # Deterministic train/val split for diagnostics
    rng = np.random.default_rng(42)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    # Per-feature stats and histograms (after scaling)
    print("Per-feature min/max and 10-bin hist (scaled):")
    edges01 = np.linspace(0.0, 1.0, 11)
    for j, name in enumerate(FINAL_COL_ORDER):
        col = X[:, j]
        if name in ("is_free", "age_0_7", "age_7_30", "age_30_90", "age_90_180",
                    "age_180_365", "age_365_730", "age_730_1825", "age_1825_plus"):
            unique, counts = np.unique(col, return_counts=True)
            print(f"- {name}: binary/one-hot counts {dict(zip(unique.astype(int), counts.tolist()))}")
        else:
            mn = float(np.min(col)); mx = float(np.max(col))
            hist, _ = np.histogram(col, bins=edges01)
            print(f"- {name}: min={mn:.4f} max={mx:.4f} hist={hist.tolist()}")

    # PSI drift check (train->val) on continuous scaled columns
    print("PSI train→val (warn if > 0.2):")
    for name in FINAL_COL_ORDER:
        if name in ("is_free", "age_0_7", "age_7_30", "age_30_90", "age_90_180",
                    "age_180_365", "age_365_730", "age_730_1825", "age_1825_plus"):
            continue
        j = FINAL_COL_ORDER.index(name)
        psi = _psi(X[tr_idx, j], X[va_idx, j], bins=10)
        flag = " [WARN]" if psi > 0.2 else ""
        print(f"- {name}: PSI={psi:.3f}{flag}")

    # Popularity dominance guard: corr(log1p_total, others)
    j_total = FINAL_COL_ORDER.index("log1p_total")
    ref = X[:, j_total]
    print("Correlation |r| with log1p_total (flag if > 0.9):")
    for name in FINAL_COL_ORDER:
        if name == "log1p_total":
            continue
        j = FINAL_COL_ORDER.index(name)
        if name in ("is_free", "age_0_7", "age_7_30", "age_30_90", "age_90_180",
                    "age_180_365", "age_365_730", "age_730_1825", "age_1825_plus"):
            continue
        col = X[:, j]
        r = float(np.corrcoef(ref, col)[0, 1]) if np.std(col) > 0 and np.std(ref) > 0 else 0.0
        mark = " [FLAG]" if abs(r) > 0.9 else ""
        print(f"- {name}: r={r:.3f}{mark}")

    # Persist artifacts
    os.makedirs('ml/data', exist_ok=True)
    save_numpy(X, 'ml/data/X_structured.npy')
    app_ids = df['app_id'].astype(int).to_numpy().tolist()
    appid_to_rowidx = {int(a): int(i) for i, a in enumerate(app_ids)}
    save_json(appid_to_rowidx, 'ml/data/appid_to_rowidx.json')

    schema = {
        "columns": FINAL_COL_ORDER,
        "scaling": scaling,
        "snapshot_date": str(pd.to_datetime(df['snapshot_date'].max()).normalize().date()),
    }
    save_json(schema, 'ml/data/structured_schema.json')
    print('✅ Chunk 2 structured features saved (stationary, discount-free)')

