import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from scipy import sparse
import joblib
import json

def main():
    # — Load raw features and embeddings
    df = pd.read_parquet("/app/ml/data/raw_game_features.parquet")
    embs = np.load("/app/ml/data/game_desc_embeddings.npy")

    # — 1) One‐hot encode tags, genres, and categories
    mlb_tags = MultiLabelBinarizer(sparse_output=True)
    mlb_genres = MultiLabelBinarizer(sparse_output=True)
    mlb_cats = MultiLabelBinarizer(sparse_output=True)

    tag_ohe = mlb_tags.fit_transform(df["tags"])
    genre_ohe = mlb_genres.fit_transform(df["genres"])
    category_ohe = mlb_cats.fit_transform(df["categories"])

    # — 2) Prepare numeric features
    #    is_free is already 0/1; ensure integer type
    df["is_free"] = df["is_free"].astype(int)

    #    Fill missing prices with the median price so that "no data" ≠ "free"
    price_median = df["price"].median()
    df["price"] = df["price"].fillna(price_median)

    #    Missing discount → treat as zero discount
    df["discount"] = df["discount"].fillna(0)

    #    Select and scale
    num_cols = df[["is_free", "price", "discount"]]
    scaler = StandardScaler()
    num_s = scaler.fit_transform(num_cols)

    # — 3) Stack everything into one sparse matrix
    X = sparse.hstack([
        sparse.csr_matrix(embs),       # [N × H] description embeddings
        sparse.csr_matrix(num_s),      # [N × 3] numeric block
        tag_ohe,                       # [N × T_tags]
        genre_ohe,                     # [N × T_genres]
        category_ohe                   # [N × T_categories]
    ]).tocsr()

    # — 4) Persist artifact files
    os.makedirs("/app/ml/data", exist_ok=True)
    joblib.dump(X,              "/app/ml/data/game_feature_matrix.joblib")
    joblib.dump(mlb_tags,       "/app/ml/data/tag_binarizer.joblib")
    joblib.dump(mlb_genres,     "/app/ml/data/genre_binarizer.joblib")
    joblib.dump(mlb_cats,       "/app/ml/data/category_binarizer.joblib")
    joblib.dump(scaler,         "/app/ml/data/num_scaler.joblib")

    # — 5) Persist an app_id → row_index mapping during dataset build
    app_id_list = df["app_id"].tolist()
    with open("/app/ml/data/app_id_index.json","w") as f:
        json.dump(app_id_list, f)

    print("✅ feature matrix and transformers ready")

if __name__ == "__main__":
    main()
