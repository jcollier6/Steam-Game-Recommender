"""Simplified pairwise ranking training loop."""
import os
from math import ceil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup

from .chunk8_embeddings import EmbeddingTables
from .chunk9_model import UserMetaFC, ScoreMLP, ScoreBilinear
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
        if len(positives) < 10:
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


def train_model(
    num_epochs: int | None = None,
    batch_size: int = 256,
    data_dir: str | None = None,
    scorer: str = "mlp",
) -> None:
    """Train embedding models with pairwise ranking.

    The function precomputes item and user metadata, supports optional
    adaptive early stopping and runs on CPU or GPU depending on availability.
    """

    if data_dir is None:
        data_dir = os.getenv("ML_DATA_DIR", "ml/data")
    interactions = pd.read_pickle(os.path.join(data_dir, "interactions_df.pkl"))
    splits = _split_by_user(interactions)
    user_ids = sorted(splits.keys())
    user_id_map = build_id_to_index_map(user_ids)
    save_json(user_id_map, os.path.join(data_dir, "user_id_to_index.json"))
    app_id_to_index = load_json(os.path.join(data_dir, "app_id_to_meta_index.json"))
    global_popular = load_json(os.path.join(data_dir, "global_popular_games.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag_dim = load_numpy(os.path.join(data_dir, "tag_game_emb.npy")).shape[1]

    tables = EmbeddingTables(len(user_ids), len(app_id_to_index))
    tables.user_emb = tables.user_emb.to(device)
    tables.item_emb = tables.item_emb.to(device)
    user_meta_fc = UserMetaFC(tag_dim + 1).to(device)
    if scorer == "bilinear":
        score_model = ScoreBilinear().to(device)
    else:
        score_model = ScoreMLP().to(device)

    params = list(tables.user_emb.parameters()) + list(tables.item_emb.parameters())
    params += list(user_meta_fc.parameters()) + list(score_model.parameters())
    optimizer = AdamW(params, lr=1e-4, weight_decay=1e-5)
    steps_per_epoch = ceil(len(user_ids) / batch_size)

    item_meta_embs = torch.from_numpy(
        load_numpy(os.path.join(data_dir, "item_meta_embs.npy"))
    ).float().to(device)

    user_meta_list: List[np.ndarray] = []
    for uid in user_ids:
        _, _, meta_raw = compute_user_meta_raw(uid, interactions, data_dir)
        user_meta_list.append(meta_raw)
    user_meta_tensor = torch.tensor(
        np.vstack(user_meta_list), dtype=torch.float32, device=device
    )

    # Configure epoch schedule: fixed or adaptive and scheduler total steps
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default

    adaptive = num_epochs is None
    min_epochs = _env_int("MIN_EPOCHS", 5)
    max_epochs = _env_int("MAX_EPOCHS", 20)
    patience = _env_int("PATIENCE_EPOCHS", 2)
    rel_improve = _env_float("REL_IMPROVE", 0.005)  # 0.5% relative improvement

    # Initialize scheduler with a safe upper bound on total steps
    if num_epochs is None:
        total_steps = max(1, steps_per_epoch * max_epochs)
    else:
        total_steps = max(1, steps_per_epoch * num_epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        steps_per_epoch,  # warmup for roughly 1 epoch
        total_steps,
    )

    def validate() -> Tuple[float, float, float]:
        pair_correct = 0
        pair_total = 0
        hits = 0
        ndcg_sum = 0.0
        total_pos = 0
        all_items = [int(k) for k in app_id_to_index.keys()]
        for uid in user_ids:
            val_df = splits[uid]["val"]
            if val_df.empty:
                continue
            pos_mask = (
                (val_df["playtime_forever"] > 0)
                | (val_df["playtime_2weeks"] > 0)
                | (val_df["wishlisted"])
            )
            if pos_mask.sum() == 0 or pos_mask.sum() == len(val_df):
                continue
            u_idx = user_id_map[uid]
            user_pos_all = set(
                splits[uid]["train"]["app_id"].tolist()
                + splits[uid]["val"]["app_id"].tolist()
                + splits[uid]["test"]["app_id"].tolist()
            )
            with torch.no_grad():
                u_emb = tables.user_emb(
                    torch.tensor([u_idx], dtype=torch.long, device=device)
                )
                user_meta = user_meta_tensor[u_idx].unsqueeze(0)
                user_meta_emb = user_meta_fc(user_meta)
                scores: List[float] = []
                labels: List[int] = []
                for _, row in val_df.iterrows():
                    app = int(row["app_id"])
                    idx = app_id_to_index.get(str(app), 0)
                    item_emb = tables.item_emb(
                        torch.tensor([idx], dtype=torch.long, device=device)
                    )
                    item_meta = item_meta_embs[idx].unsqueeze(0)
                    x = torch.cat([u_emb, user_meta_emb, item_emb, item_meta], dim=1)
                    scores.append(score_model(x).item())
                    label = int(
                        (row["playtime_forever"] > 0)
                        or (row["playtime_2weeks"] > 0)
                        or row["wishlisted"]
                    )
                    labels.append(label)
                scores_np = np.array(scores)
                labels_np = np.array(labels)
                pos_scores = scores_np[labels_np == 1]
                neg_scores = scores_np[labels_np == 0]
                if len(pos_scores) == 0 or len(neg_scores) == 0:
                    continue
                pair_correct += (pos_scores[:, None] > neg_scores[None, :]).sum()
                pair_total += len(pos_scores) * len(neg_scores)

                # Per-positive ranking evaluation
                for pos_app in val_df[pos_mask]["app_id"].tolist():
                    neg_pool = [
                        a
                        for a in global_popular
                        if a not in user_pos_all and a != pos_app
                    ]
                    if len(neg_pool) < 99:
                        neg_pool = [
                            a for a in all_items if a not in user_pos_all and a != pos_app
                        ]
                    if len(neg_pool) < 99:
                        continue
                    negatives = np.random.choice(neg_pool, 99, replace=False)
                    candidates = [pos_app] + list(negatives)
                    idxs = [app_id_to_index.get(str(c), 0) for c in candidates]
                    idx_tensor = torch.tensor(idxs, dtype=torch.long, device=device)
                    item_embs = tables.item_emb(idx_tensor)
                    item_meta = item_meta_embs[idx_tensor]
                    u_rep = u_emb.repeat(len(candidates), 1)
                    um_rep = user_meta_emb.repeat(len(candidates), 1)
                    x = torch.cat([u_rep, um_rep, item_embs, item_meta], dim=1)
                    cand_scores = score_model(x).view(-1).cpu().numpy()
                    cand_labels = np.zeros(len(candidates), dtype=int)
                    cand_labels[0] = 1
                    ranking = np.argsort(-cand_scores)
                    rank_pos = np.where(ranking == 0)[0][0]
                    if rank_pos < 10:
                        hits += 1
                    topk = cand_labels[ranking][:10]
                    gains = (2 ** topk - 1) / np.log2(
                        np.arange(1, len(topk) + 1) + 1
                    )
                    ndcg_sum += gains.sum()  # idcg is 1 when only one positive
                    total_pos += 1
        pair_acc = pair_correct / pair_total if pair_total else float("nan")
        hit10 = hits / total_pos if total_pos else float("nan")
        ndcg10 = ndcg_sum / total_pos if total_pos else float("nan")
        return pair_acc, hit10, ndcg10

    if adaptive:
        print(
            f"Adaptive training: min={min_epochs}, max={max_epochs}, "
            f"patience={patience}, rel_improve={rel_improve}",
            flush=True,
        )
    else:
        print(f"Training for {num_epochs} epoch(s)", flush=True)

    best_loss = float("inf")
    no_improve = 0
    epoch = 0
    while True:
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

                u_emb = tables.user_emb(
                    torch.tensor([u_idx], dtype=torch.long, device=device)
                )
                pos_emb = tables.item_emb(
                    torch.tensor([pos_idx], dtype=torch.long, device=device)
                )
                neg_emb = tables.item_emb(
                    torch.tensor([neg_idx], dtype=torch.long, device=device)
                )

                user_meta = user_meta_tensor[u_idx].unsqueeze(0)
                user_meta_emb = user_meta_fc(user_meta)

                pos_meta = item_meta_embs[pos_idx].unsqueeze(0)
                neg_meta = item_meta_embs[neg_idx].unsqueeze(0)

                x_pos = torch.cat([u_emb, user_meta_emb, pos_emb, pos_meta], dim=1)
                x_neg = torch.cat([u_emb, user_meta_emb, neg_emb, neg_meta], dim=1)
                s_pos = score_model(x_pos)
                s_neg = score_model(x_neg)
                diff = torch.clamp(s_pos - s_neg, -30.0, 30.0)
                loss = -torch.log(torch.sigmoid(diff))
                losses.append(loss)
            loss_batch = torch.stack(losses).mean()
            loss_batch.backward()
            clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss_batch.item()

        avg_loss = epoch_loss/steps_per_epoch
        pair_acc, hit10, ndcg10 = validate()
        total = num_epochs if not adaptive else (f"≤{max_epochs}")
        print(
            f"epoch {epoch+1}/{total} loss {avg_loss:.4f} "
            f"pair_acc {pair_acc:.4f} hit10 {hit10:.4f} ndcg10 {ndcg10:.4f}",
            flush=True
        )

        # Early stopping logic
        if best_loss == float("inf") or (
            (best_loss - avg_loss) > rel_improve * max(best_loss, 1e-8)
        ):
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1

        epoch += 1
        if not adaptive:
            if epoch >= (num_epochs or 1):
                break
        else:
            if epoch < min_epochs:
                continue
            if no_improve >= patience:
                print(
                    f"Early stopping after {epoch} epochs (no improvement for {patience} epoch(s))",
                    flush=True,
                )
                break
            if epoch >= max_epochs:
                print(f"Reached max_epochs={max_epochs}", flush=True)
                break

    tables.compute_means()
    tables.save(os.path.join(data_dir, "emb_tables"))
    save_checkpoint(user_meta_fc.state_dict(), os.path.join(data_dir, "user_meta_fc.pth"))
    fname = "score_bilinear.pth" if scorer == "bilinear" else "score_mlp.pth"
    save_checkpoint(score_model.state_dict(), os.path.join(data_dir, fname))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ranking model")
    parser.add_argument("--scorer", choices=["mlp", "bilinear"], default="mlp")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    train_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        scorer=args.scorer,
    )
