"""Simplified pairwise ranking training loop."""
import os
from math import ceil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from .chunk8_embeddings import EmbeddingTables
from .chunk9_model import UserMetaFC, ScoreMLP
from .chunk7_user_profile import compute_user_meta_raw
from .utils import (
    load_numpy,
    load_json,
    save_checkpoint,
    save_json,
    build_id_to_index_map,
)


def _split_by_user(df: pd.DataFrame) -> Dict[int, Dict[str, pd.DataFrame]]:
    splits = {}
    for uid, grp in df.groupby("user_id"):
        positives = grp[(grp["playtime_forever"] > 0) | (grp["playtime_2weeks"] > 0) | (grp["wishlisted"])]
        if len(positives) < 20:
            continue
        grp = grp.sample(frac=1, random_state=42)
        n = len(grp)
        idx_train_end = max(int(0.8 * n), 1)
        idx_val_end = max(int(0.9 * n), idx_train_end + 1)
        splits[uid] = {
            "train": grp.iloc[:idx_train_end],
            "val": grp.iloc[idx_train_end:idx_val_end],
            "test": grp.iloc[idx_val_end:],
        }
    return splits


def _sample_positive(train_df: pd.DataFrame) -> int:
    pos_df = train_df[(train_df["playtime_forever"] > 0) | (train_df["playtime_2weeks"] > 0) | (train_df["wishlisted"])]
    row = pos_df.sample(1).iloc[0]
    return int(row["app_id"])


def _sample_negative(user_pos: List[int], global_popular: List[int]) -> int:
    candidates = [a for a in global_popular if a not in user_pos]
    if not candidates:
        return int(np.random.choice(global_popular))
    return int(np.random.choice(candidates))


def train_model(num_epochs: int = 1, batch_size: int = 256, data_dir: str | None = None) -> None:
    if data_dir is None:
        data_dir = os.getenv("ML_DATA_DIR", "ml/data")
    interactions = pd.read_pickle(os.path.join(data_dir, "interactions_df.pkl"))
    splits = _split_by_user(interactions)
    user_ids = sorted(splits.keys())
    user_id_map = build_id_to_index_map(user_ids)
    save_json(user_id_map, os.path.join(data_dir, "user_id_to_index.json"))
    app_id_to_index = load_json(os.path.join(data_dir, "app_id_to_meta_index.json"))
    global_popular = load_json(os.path.join(data_dir, "global_popular_games.json"))

    tag_dim = load_numpy(os.path.join(data_dir, "T_pca_norm.npy")).shape[1]

    tables = EmbeddingTables(len(user_ids), len(app_id_to_index))
    user_meta_fc = UserMetaFC(tag_dim + 1)
    score_mlp = ScoreMLP()

    params = list(tables.user_emb.parameters()) + list(tables.item_emb.parameters())
    params += list(user_meta_fc.parameters()) + list(score_mlp.parameters())
    optimizer = AdamW(params, lr=1e-4, weight_decay=1e-5)
    steps_per_epoch = ceil(len(user_ids) / batch_size)
    scheduler = get_cosine_schedule_with_warmup(optimizer, steps_per_epoch, num_epochs * steps_per_epoch)

    item_meta_embs = load_numpy(os.path.join(data_dir, "item_meta_embs.npy"))

    for epoch in range(num_epochs):
        np.random.shuffle(user_ids)
        epoch_loss = 0.0
        for i in range(0, len(user_ids), batch_size):
            batch_u = user_ids[i : i + batch_size]
            batch_pairs: List[Tuple[int, int, int]] = []
            for u in batch_u:
                train_df = splits[u]["train"]
                pos = _sample_positive(train_df)
                user_pos_list = train_df["app_id"].tolist()
                neg = _sample_negative(user_pos_list, global_popular)
                batch_pairs.append((u, pos, neg))

            optimizer.zero_grad()
            losses = []
            for u, pos, neg in batch_pairs:
                u_idx = user_id_map[u]
                pos_idx = app_id_to_index.get(str(pos), 0)
                neg_idx = app_id_to_index.get(str(neg), 0)

                u_emb = tables.user_emb(torch.tensor([u_idx]))
                pos_emb = tables.item_emb(torch.tensor([pos_idx]))
                neg_emb = tables.item_emb(torch.tensor([neg_idx]))

                _, _, meta_raw = compute_user_meta_raw(u, interactions, data_dir)
                user_meta_emb = user_meta_fc(torch.tensor(meta_raw).float().unsqueeze(0))

                pos_meta = torch.from_numpy(item_meta_embs[pos_idx]).float().unsqueeze(0)
                neg_meta = torch.from_numpy(item_meta_embs[neg_idx]).float().unsqueeze(0)

                x_pos = torch.cat([u_emb, user_meta_emb, pos_emb, pos_meta], dim=1)
                x_neg = torch.cat([u_emb, user_meta_emb, neg_emb, neg_meta], dim=1)
                s_pos = score_mlp(x_pos)
                s_neg = score_mlp(x_neg)
                diff = torch.clamp(s_pos - s_neg, -30.0, 30.0)
                loss = -torch.log(torch.sigmoid(diff))
                losses.append(loss)
            loss_batch = torch.stack(losses).mean()
            loss_batch.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss_batch.item()

        print(f"epoch {epoch} loss {epoch_loss/steps_per_epoch:.4f}")

    tables.compute_means()
    tables.save(os.path.join(data_dir, "emb_tables"))
    save_checkpoint(user_meta_fc.state_dict(), os.path.join(data_dir, "user_meta_fc.pth"))
    save_checkpoint(score_mlp.state_dict(), os.path.join(data_dir, "score_mlp.pth"))


if __name__ == "__main__":
    train_model(num_epochs=1)
