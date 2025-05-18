import os
import json
import torch
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from train_model import fetch_user_positives, MLP

DATA_DIR = os.getenv("ML_DATA_DIR", "/app/ml/data")
MODEL_PATH = "/app/ml/models/model_v1.pth"

def load_feature_matrix():
    return joblib.load(os.path.join(DATA_DIR, "game_feature_matrix.joblib"))

def load_app_ids():
    with open(os.path.join(DATA_DIR, "app_id_index.json"), "r") as f:
        return json.load(f)

def get_user_exclusions(steamid):
    positives = fetch_user_positives(steamid)
    return {rec["app_id"] for rec in positives}

def share_enough_tags(X, idx, user_tag_vec, threshold=0.1):
    game_vec = np.asarray(X[idx, -user_tag_vec.shape[1]:].toarray())
    sim = cosine_similarity(game_vec, np.asarray(user_tag_vec))[0, 0]
    return sim > threshold

def recommend_games(steamid, top_k=20):
    X = load_feature_matrix()
    app_id_list = load_app_ids()
    excluded_ids = get_user_exclusions(steamid)

    # Automatically determine tag feature size from tag_binarizer
    tag_binarizer = joblib.load(os.path.join(DATA_DIR, "tag_binarizer.joblib"))
    tag_feature_size = len(tag_binarizer.classes_)
    in_features = X.shape[1] + 1  # base + similarity score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_features=in_features).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Compute user tag vector
    user_idxs = [idx for idx, aid in enumerate(app_id_list) if aid in excluded_ids]
    tag_matrix = X[:, -tag_feature_size:]
    user_tag_vec = np.asarray(tag_matrix[user_idxs].mean(axis=0))

    scores = []
    with torch.no_grad():
        for idx, app_id in enumerate(app_id_list):
            if app_id in excluded_ids:
                continue

            if not share_enough_tags(X, idx, user_tag_vec):
                continue

            x_base = X[idx].toarray().ravel()
            tag_sim = cosine_similarity(np.asarray(tag_matrix[idx].toarray()), np.asarray(user_tag_vec.reshape(1, -1)))[0, 0]
            x_full = np.concatenate([x_base, [tag_sim]])
            x = torch.tensor(x_full, dtype=torch.float32).unsqueeze(0).to(device)
            logit = model(x).item()
            prob = 1 / (1 + np.exp(-logit))
            scores.append((prob, app_id))

    scores.sort(reverse=True)
    return [app_id for _, app_id in scores[:top_k]]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steamid", required=True, help="Steam ID of the user")
    parser.add_argument("--top_k", type=int, default=20, help="Number of games to recommend")
    args = parser.parse_args()

    recommendations = recommend_games(args.steamid, args.top_k)
    print(f"Top {args.top_k} recommendations for user {args.steamid}:")
    for app_id in recommendations:
        print(app_id)