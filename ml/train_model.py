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

# Adjust this path if your ML files live elsewhere
DATA_DIR = os.getenv("ML_DATA_DIR", "/app/ml/data")

# 1) Load the full feature matrix X
feature_matrix_path = os.path.join(DATA_DIR, "game_feature_matrix.joblib")
X = joblib.load(feature_matrix_path)  
# X is a scipy CSR matrix of shape (N_games, F_features)

# 2) Load app_id → row index mapping
with open(os.path.join(DATA_DIR, "app_id_index.json"), "r") as f:
    app_id_list = json.load(f)
# app_id_list is a list whose i-th element is the app_id corresponding to row i

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
            "playtime_forever": g.get("playtime_forever", 0),
            "playtime_2weeks": g.get("playtime_2weeks", 0),
            "priority": 0,
            "date_added": 0,
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
            "playtime_forever": 0,
            "playtime_2weeks": 0,
            "priority": w.get("priority", 0),
            "date_added": w.get("date_added", 0),
            "source": "wishlist"
        })

    return positives



NOW = time.time()
# half‐life for recency (in seconds); e.g. 30 days
HALF_LIFE = 30 * 24 * 3600
decay_rate =  math.log(2) / HALF_LIFE

def compute_strength(record: Dict) -> float:
    if record["source"] == "owned":
        # you could normalize by log(playtime_forever+1)
        return 1.0
    else:  # wishlist
        age = NOW - record["date_added"]
        recency_weight = exp(-decay_rate * age)
        # priority is Steam’s 0–6 integer: normalize to [0,1]
        prio_norm = record.get("priority", 0) / 6
        # combine so “just added” + “max priority” → ~1.0
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
    X_list, y_list, w_list = [], [], []
    for steamid, positives in all_users_data.items():
        pos_app_ids = [p["app_id"] for p in positives]
        user_tag_vec = average_ohe_vector(pos_app_ids)
        negs = sample_negatives(positives, num_neg=len(positives))

        for rec in positives:
            idx = app_to_idx.get(rec["app_id"])
            if idx is None: continue
            fv = X[idx].toarray().ravel()
            tag_sim = cosine_similarity([user_tag_vec], [fv])[0, 0]
            fv = np.concatenate([fv, [tag_sim]])
            X_list.append(fv)
            y_list.append(1)
            w_list.append(compute_strength(rec))

        for app_id in negs:
            idx = app_to_idx[app_id]
            fv = X[idx].toarray().ravel()
            tag_sim = cosine_similarity([user_tag_vec], [fv])[0, 0]
            fv = np.concatenate([fv, [tag_sim]])
            X_list.append(fv)
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


def main():
    print("🛠 Loading feature matrix...")
    X = joblib.load("/app/ml/data/game_feature_matrix.joblib")
    print(f"✅ Loaded feature matrix: {X.shape}")

    # Note to self, fill this array
    steamids: List[str] = [76561198293567287, 76561198087189722, 76561198199410747, 76561198175748806, 76561198208956405, 76561199074973018, 76561198122740474, 76561199065864867, 76561198123185319, 76561198247719246, 76561198068244088, 76561198292610490, 76561198289083517, 76561198948235914]
    all_users_data = {sid: fetch_user_positives(sid) for sid in steamids}
    X_data, y_data, w_data = build_dataset(all_users_data)
    print(f"Built dataset with {len(y_data)} samples")

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_data, y_data, w_data, test_size=0.2, stratify=y_data, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    w_train_t = torch.tensor(w_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

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
            y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
            w_val_t = torch.tensor(w_val, dtype=torch.float32).to(device)
            val_loss = loss_fn(val_logits, y_val_t)
            weighted_val_loss = (val_loss * w_val_t).mean().item()

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
