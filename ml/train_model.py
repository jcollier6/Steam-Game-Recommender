import joblib
import json
import os
import logging
import requests
from typing import List, Dict
import numpy as np
from random import sample
import time
import math
from math import exp
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%m-%d %H:%M:%S"
)

# Adjust this path if your ML files live elsewhere
DATA_DIR = os.getenv("ML_DATA_DIR", "/app/ml/data")

# 1) Load the full feature matrix X
feature_matrix_path = os.path.join(DATA_DIR, "game_feature_matrix.joblib")
X = joblib.load(feature_matrix_path)
# X is a scipy CSR matrix of shape (N_games, F_features)

# 2) Load app_id → row index mapping
with open(os.path.join(DATA_DIR, "app_id_index.json"), "r") as f:
    app_id_list = json.load(f)
# app_id_list is a list whose i‐th element is the app_id corresponding to row i

# For quick lookup, build a dict:
app_to_idx = {app_id: idx for idx, app_id in enumerate(app_id_list)}

# --- Helper to get your Steam API key ---
def get_API_key() -> str:
    key = os.getenv("API_KEY", "")
    if not key:
        logging.error("API_KEY environment variable not set.")
    return key

# --- Fetch one user’s owned games + wishlist, merge into positives ---
def fetch_user_positives(steamid: str) -> List[Dict]:
    api_key = get_API_key()
    positives: List[Dict] = []

    def safe_get_json(url):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logging.warning(f"⚠️ Failed to fetch data for {steamid}: {e}")
            return {}

    # 1) Owned games
    owned_url = (
        f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
        f"?key={api_key}&steamid={steamid}&include_appinfo=false"
    )
    owned_resp = safe_get_json(owned_url)
    owned = owned_resp.get("response", {}).get("games", [])

    for g in owned:
        positives.append({
            "app_id": g["appid"],
            "playtime_forever": g.get("playtime_forever", "0"),
            "playtime_2weeks": g.get("playtime_2weeks", "0"),
            "priority": "0",
            "date_added": "0",
            "source": "owned"
        })

    # 2) Wishlist items
    wish_url = (
        f"https://api.steampowered.com/IWishlistService/GetWishlist/v1/"
        f"?key={api_key}&steamid={steamid}"
    )
    wish_resp = safe_get_json(wish_url)
    wish = wish_resp.get("response", {}).get("items", [])

    for w in wish:
        positives.append({
            "app_id": w["appid"],
            "playtime_forever": "0",
            "playtime_2weeks": "0",
            "priority": w.get("priority", "0"),
            "date_added": w.get("date_added", "0"),
            "source": "wishlist"
        })

    return positives



NOW = time.time()
# half‐life for recency (in seconds); e.g. 30 days
HALF_LIFE = 30 * 24 * 3600
decay_rate = math.log(2) / HALF_LIFE

def compute_wishlist_strength(record: Dict) -> float:
    """
    priority (0–6 normalized to 0–1) multiplied by recency decay.
    """
    age = NOW - record["date_added"]
    recency_weight = exp(-decay_rate * age)
    prio_norm = record.get("priority", 0) / 6
    return prio_norm * recency_weight

def sample_negatives(positives: List[Dict], num_neg: int) -> List[int]:
    all_apps = set(app_to_idx.keys())
    pos_apps = {p["app_id"] for p in positives}
    neg_candidates = list(all_apps - pos_apps)
    # simple uniform sampling; you can stratify later if you like
    return sample(neg_candidates, min(num_neg, len(neg_candidates)))

def average_ohe_vector(app_ids):
    vectors = [X[app_to_idx[aid]].toarray().ravel() for aid in app_ids if aid in app_to_idx]
    return np.mean(vectors, axis=0) if vectors else np.zeros(X.shape[1])

def build_dataset(all_users_data: Dict[str, List[Dict]]):
    """
    Build training dataset with features, labels, and weights.
    Now computes a per‐user 75th percentile on playtime_forever, and apply:
      normalized_playtime = playtime_forever / playtime_75th
      weighted_playtime  = normalized_playtime + 15 * (playtime_2weeks / playtime_75th)

    Wishlist items use compute_wishlist_strength().
    """
    X_list, y_list, w_list = [], [], []

    for steamid, positives in all_users_data.items():
        # 1) Build a DataFrame to compute the 75th percentile of playtime_forever
        df_user = pd.DataFrame(positives)
        # If no positives or missing playtime column, force a 1.0 quantile
        if df_user.empty or "playtime_forever" not in df_user.columns:
            logging.info(f"user: {steamid} is empty or missing playtime_forever column")
            playtime_75th = 1.0
        else:
            df_user["playtime_forever"] = df_user["playtime_forever"].astype(float)
            if df_user["playtime_forever"].empty:
                logging.info(f"user: {steamid} is missing playtime_forever column")
                playtime_75th = 1.0
            else:
                playtime_75th = df_user["playtime_forever"].quantile(0.75)
                if playtime_75th == 0:
                    logging.info(f"user: {steamid} has a playtime_75th of zero")
                    playtime_75th = 1.0

        # 2) Build the user_tag_vec exactly as before (uniform average of tag features)
        pos_app_ids = [p["app_id"] for p in positives if p["app_id"] in app_to_idx]
        user_tag_vec = average_ohe_vector(pos_app_ids)

        # 3) Sample negative examples (unowned games)
        negs = sample_negatives(positives, num_neg=len(positives))

        # 4) Loop through positives to create (features, label=1, weight)
        for rec in positives:
            idx = app_to_idx.get(rec["app_id"])
            if idx is None:
                continue

            fv = X[idx].toarray().ravel()
            tag_sim = cosine_similarity([user_tag_vec], [fv])[0, 0]
            fvec = np.concatenate([fv, [tag_sim]])

            # Compute the weight depending on source:
            if rec["source"] == "owned":
                # Normalize playtime_forever and incorporate playtime_2weeks
                pt_f = float(rec.get("playtime_forever", 0))
                pt_r = float(rec.get("playtime_2weeks", 0))
                normalized = pt_f / playtime_75th
                weighted_playtime = normalized + 15 * (pt_r / playtime_75th)
                weight = weighted_playtime if weighted_playtime > 0 else 0.1
            else:  # wishlist
                weight = compute_wishlist_strength(rec)

            X_list.append(fvec)
            y_list.append(1)
            w_list.append(weight)

        # 5) Loop through negatives to create (features, label=0, weight=1.0)
        for app_id in negs:
            idx = app_to_idx[app_id]
            fv = X[idx].toarray().ravel()
            tag_sim = cosine_similarity([user_tag_vec], [fv])[0, 0]
            fvec = np.concatenate([fv, [tag_sim]])
            X_list.append(fvec)
            y_list.append(0)
            w_list.append(1.0)

    return np.vstack(X_list), np.array(y_list), np.array(w_list)



class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


def load_steamids_from_file(path: str) -> List[str]:
    with open(path, "r") as f:
        # assume one SteamID per line, strip whitespace and ignore empty lines
        return [line.strip() for line in f if line.strip()]

def main():
    print("🛠 Loading feature matrix...")
    X = joblib.load("/app/ml/data/game_feature_matrix.joblib")
    print(f"✅ Loaded feature matrix: {X.shape}")

    steamids: List[str] = load_steamids_from_file("./ml/training_steamids")

    all_users_data = {sid: fetch_user_positives(sid) for sid in steamids}
    X_data, y_data, w_data = build_dataset(all_users_data)
    print(f"Built dataset with {len(y_data)} samples")
    print(f"Weight stats — min: {w_data.min():.4f}, max: {w_data.max():.4f}, mean: {w_data.mean():.4f}")

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_data, y_data, w_data, test_size=0.2, stratify=y_data, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    w_train_t = torch.tensor(w_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    w_val_t = torch.tensor(w_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t, w_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = MLP(in_features=X_train.shape[1]).to(device)  # +1 for tag similarity
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    best_val_loss = float('inf')
    patience = 1
    patience_counter = 0
    max_epochs = 20

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            weighted_loss = (loss * wb).mean()
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_loss = loss_fn(val_logits, y_val_t.to(device))
            weighted_val_loss = (val_loss * w_val_t.to(device)).mean().item()

        print(f"Epoch {epoch+1}, Train Loss: {total_loss:.4f}, Val Loss: {weighted_val_loss:.4f}")

        # Precision@K
        with torch.no_grad():
            val_logits = model(X_val_t.to(device)).cpu().numpy()
            val_probs = 1 / (1 + np.exp(-val_logits))  # sigmoid
            Ks = [10, 25, 50, 100]
            for K in Ks:
                top_k = np.argsort(val_probs)[-K:]
                precision = y_val[top_k].mean()
                print(f"Precision@{K}: {precision:.4f}")

        # Early stopping logic
        if weighted_val_loss < best_val_loss:
            best_val_loss = weighted_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "/app/ml/models/model_v1.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t.to(device)).cpu().numpy()
        val_probs = 1 / (1 + np.exp(-val_logits))

    auc = roc_auc_score(y_val, val_probs)
    print(f"Validation AUC: {auc:.4f}")

    K = 200
    top_k = np.argsort(val_probs)[-K:]
    precision_at_k = y_val[top_k].mean()
    print(f"Precision@{K}: {precision_at_k:.4f}")

    os.makedirs("/app/ml/models", exist_ok=True)
    torch.save(model.state_dict(), "/app/ml/models/model_v1.pth")
    print("✅ Model saved to /app/ml/models/model_v1.pth")

if __name__ == "__main__":
    main()
