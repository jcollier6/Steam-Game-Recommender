import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from train_model import MLP  # your MLP definition
import logging
import time
from contextlib import asynccontextmanager
from math import log, exp
from collections import Counter
import torch
import json
import joblib
import numpy as np 

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%m-%d %H:%M:%S"
)

# ─── Constants & Globals ───────────────────────────────────────────────────────
DATA_DIR = os.getenv("ML_DATA_DIR", "/app/ml/data")
MODEL_PATH = "/app/ml/models/model_v1.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These will be populated once at container startup:
MODEL: Optional[MLP] = None
X_MATRIX: Optional[Any] = None        # scipy-csr, but we'll convert as needed
APP_ID_LIST: Optional[list[str]] = None        # List of app_ids
TAG_BINARIZER: Optional[Any] = None
TAG_MATRIX_DENSE: Optional[np.ndarray] = None  # 2D array of shape (N_games, N_tags)

# For wishlist‐agedecay:
NOW = time.time()
HALF_LIFE = 30 * 24 * 3600  # 30 days
DECAY_RATE = log(2) / HALF_LIFE


# ─── Pydantic Models ───────────────────────────────────────────────────────────
class OwnedGameRecord(BaseModel):
    app_id: str
    playtime_forever: float
    playtime_2weeks: float


class WishlistRecord(BaseModel):
    app_id: str
    priority: float   # Steam’s priority (0–6)
    date_added: float # UNIX timestamp


class RecommendationRequest(BaseModel):
    steam_id: str
    top_k: int = 20
    owned_games: List[OwnedGameRecord]
    wishlist: Optional[List[WishlistRecord]] = None


# ─── FastAPI Lifespan: load all non‐user data once ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, X_MATRIX, APP_ID_LIST, TAG_BINARIZER, TAG_MATRIX_DENSE

    logging.info("=== Loading recommendation system resources ===")

    # 1) Load feature matrix (sparse CSR) from disk
    X_MATRIX = joblib.load(os.path.join(DATA_DIR, "game_feature_matrix.joblib"))

    # 2) Load the app_id → row‐index mapping
    with open(os.path.join(DATA_DIR, "app_id_index.json"), "r") as f:
        raw_list = json.load(f)
        # Cast every ID to string so lookups match the "owned_games" strings
        APP_ID_LIST = [str(a) for a in raw_list]

    # 3) Load tag‐binarizer so we know how many “tag” columns there are
    TAG_BINARIZER = joblib.load(os.path.join(DATA_DIR, "tag_binarizer.joblib"))
    tag_feature_size = len(TAG_BINARIZER.classes_)

    # 4) Load the trained model once (in_features = X.shape[1] + 1)
    in_features = X_MATRIX.shape[1] + 1
    MODEL = MLP(in_features=in_features).to(DEVICE)
    MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    MODEL.eval()

    # 5) Precompute a dense tag‐matrix for fast vector ops: last tag_feature_size columns of X
    TAG_MATRIX_DENSE = X_MATRIX[:, -tag_feature_size :].toarray()

    logging.info("=== Resource loading complete ===")
    yield
    # (No teardown needed.)


app = FastAPI(lifespan=lifespan)


# ─── Helper: Compute “strength” (weight) of one record (owned or wishlist) ──────
def compute_strength(record: dict) -> float:
    """
    Exactly matches your old logic:
    - Owned games: strength = 1.0 (we could do log(playtime+1), but original was 1.0)
    - Wishlist: combine normalized priority [0..1] with recency decay
    """
    source = record.get("source", "owned")
    if source == "owned":
        return 1.0
    else:
        # wishlist entry:
        age = NOW - record["date_added"]               # seconds since added
        recency_weight = exp(-DECAY_RATE * age)          # decays over 30 days
        prio_norm = record.get("priority", 0.0) / 6.0    # normalize 0..6 → 0..1
        return prio_norm * recency_weight


# ─── Endpoint: /recommend ───────────────────────────────────────────────────────
@app.post("/recommend")
def recommend_games_api(req: RecommendationRequest):
    try:
        start_overall = time.perf_counter()
        logging.info("=== Recommendation process started ===")

        # 1) Build one combined “positives” list with both owned and wishlist
        #    Each entry must include: app_id, playtime_forever, playtime_2weeks (for owned),
        #    or priority & date_added (for wishlist).
        #    We’ll tag each record with a “source” field so compute_strength knows how to handle it.
        positives = []
        for og in req.owned_games:
            positives.append({
                "app_id": og.app_id,
                "playtime_forever": og.playtime_forever,
                "playtime_2weeks": og.playtime_2weeks,
                "source": "owned"
            })
        if req.wishlist:
            for wl in req.wishlist:
                positives.append({
                    "app_id": wl.app_id,
                    "priority": wl.priority,
                    "date_added": wl.date_added,
                    "source": "wishlist"
                })

        # 2) Merge duplicates (if a given app appears both in owned & wishlist),
        #    taking “owned” as higher priority.
        merged = {}
        for record in positives:
            aid = record["app_id"]
            if aid not in merged:
                merged[aid] = record
            else:
                # If already had “owned,” ignore wishlist; if existing is wishlist, but this is owned, override.
                if merged[aid]["source"] == "wishlist" and record["source"] == "owned":
                    merged[aid] = record

        merged_list = list(merged.values())
        if not merged_list:
            raise ValueError("User has no owned or wishlisted games.")

        # 3) Build a DataFrame-like structure to compute “weighted_playtime” exactly as before
        #    (Normalize by 75th quantile of owned playtime; then boost recent playtime by ×15.)
        #    To do that, gather only the “owned” items first—save their playtimes to compute quantile.
        owned_rows = [r for r in merged_list if r["source"] == "owned"]
        if owned_rows:
            forever_list = [r["playtime_forever"] for r in owned_rows]
            playtime_75th = np.quantile(forever_list, 0.75)
            if playtime_75th <= 0:
                playtime_75th = 1.0
        else:
            # If no owned games at all, skip quantile-based normalization (set to 1.0 to avoid division by 0)
            playtime_75th = 1.0

        # 4) For each record in merged_list, compute strength:
        user_tag_counts = Counter()
        app_idx_map = {str(aid): idx for idx, aid in enumerate(APP_ID_LIST)}

        for rec in merged_list:
            aid = rec["app_id"]
            if aid not in app_idx_map:
                # skip any app that isn’t in the global feature‐matrix mapping
                continue

            if rec["source"] == "owned":
                norm = rec["playtime_forever"] / playtime_75th
                weighted_playtime = norm + 15 * (rec["playtime_2weeks"] / playtime_75th)
                strength = weighted_playtime
            else:
                # wishlist
                strength = compute_strength(rec)

            # Accumulate weighted tag counts:
            idx = app_idx_map[aid]
            # TAG_MATRIX_DENSE[idx] is a 1‐D array of 0/1 for all tags; multiply by strength
            tag_vector = TAG_MATRIX_DENSE[idx]
            for j, val in enumerate(tag_vector):
                if val > 0:
                    user_tag_counts[j] += strength

        if not user_tag_counts:
            logging.warning("No weighted tags found—falling back to unweighted average of owned tags.")

            # Log any owned app_ids that weren’t found in APP_ID_LIST
            missing_owned = [
                r["app_id"]
                for r in merged_list
                if (r["source"] == "owned" and str(r["app_id"]) not in app_idx_map)
            ]
            if missing_owned:
                logging.warning(f"These owned app_ids were not found in APP_ID_LIST: {missing_owned}")

            # Build owned_indices for fallback average
            owned_indices = []
            for r in merged_list:
                if r["source"] == "owned":
                    aid_str = str(r["app_id"])
                    if aid_str in app_idx_map:
                        owned_indices.append(app_idx_map[aid_str])

            if owned_indices:
                # Take the mean of those TAG_MATRIX_DENSE rows
                fallback_vec = TAG_MATRIX_DENSE[owned_indices, :].mean(axis=0)
                user_tag_vec = np.asarray(fallback_vec, dtype=np.float32).reshape(1, -1)
            else:
                # No owned games matched → use a zero‐vector
                user_tag_vec = np.zeros((1, TAG_MATRIX_DENSE.shape[1]), dtype=np.float32)
        else:
            # 5) Build final user_tag_vec by normalizing weighted tag counts
            total_weight = sum(user_tag_counts.values())
            tag_indices = list(user_tag_counts.keys())
            tag_weights = np.array([user_tag_counts[i] for i in tag_indices], dtype=np.float32)
            user_tag_vec = np.zeros(TAG_MATRIX_DENSE.shape[1], dtype=np.float32)
            user_tag_vec[tag_indices] = tag_weights / total_weight
            user_tag_vec = user_tag_vec.reshape(1, -1)  # shape (1, N_tags)

        t1 = time.perf_counter()
        logging.info(f"User‐tag vector computation took {t1 - (start_overall):.2f} seconds")

        # 6) Compute cosine similarity between every game’s tag‐vector and the user_tag_vec
        t0 = time.perf_counter()
        similarities = cosine_similarity(TAG_MATRIX_DENSE, user_tag_vec).flatten()
        # Exclude any app the user “owns” or “wishlisted”:
        excluded_ids = {r["app_id"] for r in merged_list}
        mask_not_owned_or_wish = np.array([aid not in excluded_ids for aid in APP_ID_LIST])
        valid_mask = mask_not_owned_or_wish & (similarities > 0.1)  # only keep sims > 0.1
        candidate_indices = np.where(valid_mask)[0]
        result_ids = [APP_ID_LIST[i] for i in candidate_indices]
        t1 = time.perf_counter()
        logging.info(f"Filtering valid candidates took {t1 - t0:.2f} seconds")

        # 7) Prepare the features for those candidates in one bulk step, append the tag-sim as last column
        t0 = time.perf_counter()
        X_cand = X_MATRIX[candidate_indices].toarray()            # shape (N_cand, F_base)
        tag_sim_cand = similarities[candidate_indices].reshape(-1, 1)  # shape (N_cand, 1)
        features_np = np.hstack((X_cand, tag_sim_cand)).astype(np.float32)
        input_tensor = torch.from_numpy(features_np).to(DEVICE)    # (N_cand, F_base + 1)
        t1 = time.perf_counter()
        logging.info(f"Tensor prep took {t1 - t0:.2f} seconds")

        # 8) Batch‐infer probabilities
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = MODEL(input_tensor).squeeze().cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            scores = list(zip(probs, result_ids))
        t1 = time.perf_counter()
        logging.info(f"Batch inference took {t1 - t0:.2f} seconds")

        # 9) Find the top_k highest‐scoring app_ids
        t0 = time.perf_counter()
        scores.sort(key=lambda x: x[0], reverse=True)
        top_games = [aid for (_, aid) in scores[: req.top_k]]
        t1 = time.perf_counter()
        logging.info(f"Sorting & slicing top_k took {t1 - t0:.4f} seconds")

        total_time = time.perf_counter() - start_overall
        logging.info(f"=== Total recommendation time: {total_time:.2f} seconds ===")
        return {"recommendations": top_games}

    except ValueError as ve:
        logging.error(f"Recommendation Error (validation): {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Recommendation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
