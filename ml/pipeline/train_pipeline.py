import os
import time
import argparse
from .utils import get_current_utc_date
import pandas as pd
import numpy as np
import mysql.connector

from . import (
    chunk1_loader,
    chunk2_structured,
    chunk3_tags_pca,
    chunk4_text_embeddings,
    chunk5_item_meta,
    chunk6_faiss,
    chunk10_train,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help=(
            "first chunk to run. 1 runs full pipeline. Allowed values: 1–6 "
            "or 10 (10 selects the 'chunk10' step)."
        ),
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help="run only the specified chunk (1-6 or 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override NUM_EPOCHS env for chunk10 training",
    )
    parser.add_argument(
        "--svd_dim",
        type=int,
        default=None,
        help="override tag embedding dimension for chunk3",
    )
    args = parser.parse_args()

    start_time = time.time()
    step = 0

    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
    )

    epochs_env = os.getenv("NUM_EPOCHS")
    epochs = int(epochs_env) if epochs_env else None
    if args.epochs is not None:
        epochs = max(1, int(args.epochs))

    games_df_cache = {}

    def load_games_df():
        if "games_df" not in games_df_cache:
            games_df_cache["games_df"] = pd.read_pickle("ml/data/games_df.pkl")
        return games_df_cache["games_df"]

    steps = [
        ("chunk1", lambda: chunk1_loader.main(conn)),
        (
            "chunk2",
            lambda: chunk2_structured.structured_transform(
                load_games_df(), get_current_utc_date()
            ),
        ),
        (
            "chunk3",
            lambda: chunk3_tags_pca.run_tag_pca(
                load_games_df(), svd_dim=args.svd_dim or 192
            ),
        ),
        ("chunk4", lambda: chunk4_text_embeddings.run_text_embeddings(load_games_df())),
        ("chunk5", chunk5_item_meta.run_item_meta_embedding),
        ("chunk6", chunk6_faiss.build_faiss_indices),
        ("chunk10", lambda: chunk10_train.train_model(num_epochs=epochs)),
    ]

    # Determine which chunk(s) to run
    # Map chunk numbers (1-6,10) to index positions in ``steps``
    chunk_map = {
        int(name.replace("chunk", "")): idx
        for idx, (name, _) in enumerate(steps, start=1)
    }
    allowed = set(chunk_map.keys())

    if args.chunk is not None:
        run_chunk = max(1, int(args.chunk))
        if run_chunk not in allowed:
            raise SystemExit(
                f"Invalid --chunk {run_chunk}. Allowed chunks: {sorted(allowed)}"
            )
        start_idx = chunk_map[run_chunk]
        steps_to_run = [steps[start_idx - 1]]
    else:
        start_chunk = max(1, int(args.start))
        if start_chunk not in allowed:
            raise SystemExit(
                f"Invalid --start {start_chunk}. Allowed chunks: {sorted(allowed)}"
            )
        start_idx = chunk_map[start_chunk]
        steps_to_run = steps[start_idx - 1 :]

    total_steps = len(steps_to_run)

    def run_step(name, func):
        """Execute *func* and print progress with timing and ETA."""
        nonlocal step
        step += 1
        print(f"[{step}/{total_steps}] Starting {name}...", flush=True)
        s = time.time()
        result = func()
        elapsed_step = time.time() - s
        elapsed_total = time.time() - start_time
        eta = (elapsed_total / step) * (total_steps - step)
        print(
            f"[{step}/{total_steps}] Completed {name} in {elapsed_step:.1f}s "
            f"(elapsed {elapsed_total:.1f}s, ETA {eta:.1f}s)",
            flush=True,
        )
        return result

    # Pre-flight check when starting from a later chunk
    if start_idx > 1:
        data_dir = os.getenv("ML_DATA_DIR", "ml/data")
        name, _ = steps_to_run[0]
        def req(paths):
            missing = []
            print(f"Pre-flight for {name} (data_dir={data_dir}):", flush=True)
            for p in paths:
                full = p if os.path.isabs(p) else os.path.join(data_dir, p)
                ok = os.path.exists(full)
                print(f" - {'OK ' if ok else 'MISS'} {full}", flush=True)
                if not ok:
                    missing.append(full)
            if missing:
                raise SystemExit("Missing required artifacts. See 'MISS' above.")

        def print_structured_quantiles():
            qpath = os.path.join(data_dir, "structured_quantiles.json")
            if os.path.exists(qpath):
                try:
                    import json

                    with open(qpath, "r", encoding="utf-8") as f:
                        q = json.load(f)

                    metrics = [
                        k
                        for k in q.keys()
                        if k not in {"p_high_global", "p_high_by_metric", "discount_percent"}
                    ]
                    phg = q.get("p_high_global")
                    overrides = [
                        k
                        for k in q.get("p_high_by_metric", {}).keys()
                        if k != "discount_percent"
                    ]
                    print(
                        "structured_quantiles: "
                        f"metrics={metrics} "
                        f"global_p_high={phg} "
                        f"overrides={overrides}"
                    )
                except Exception as e:
                    print(f"[warn] Could not read structured_quantiles.json: {e}")

        def print_tag_pca_info():
            tpath = os.path.join(data_dir, "T_pca_norm.npy")
            ids_path = os.path.join(data_dir, "tag_pca_app_ids.npy")
            gmean_path = os.path.join(data_dir, "global_tag_mean.npy")
            try:
                t_shape = (
                    np.load(tpath, mmap_mode="r").shape if os.path.exists(tpath) else None
                )
                ids_count = (
                    np.load(ids_path, mmap_mode="r").shape[0]
                    if os.path.exists(ids_path)
                    else None
                )
                gmean_dim = (
                    np.load(gmean_path, mmap_mode="r").shape[0]
                    if os.path.exists(gmean_path)
                    else None
                )
                print(
                    "tag_pca: "
                    f"T_shape={t_shape} "
                    f"app_ids={ids_count} "
                    f"global_mean_dim={gmean_dim}"
                )
            except Exception as e:
                print(f"[warn] Could not read tag PCA artifacts: {e}")

        if name == "chunk2":
            req(["games_df.pkl"])  # from chunk1
            print_structured_quantiles()
        elif name == "chunk3":
            req(["games_df.pkl", "tag_id_map.json"])  # from chunk1
            print_structured_quantiles()
        elif name == "chunk4":
            req(["games_df.pkl"])  # from chunk1
            print_structured_quantiles()
        elif name == "chunk5":
            req([
                "structured_raw.npy",      # from chunk2
                "structured_app_ids.npy",  # from chunk2
                "structured_min.npy",      # from chunk2
                "structured_max.npy",      # from chunk2
                "tag_game_emb.npy",       # from chunk3
                "E_qwen3_short.npy",      # from chunk4
                "E_qwen3_long.npy",       # from chunk4
                "tag_app_ids.npy",        # from chunk3
            ])
            print_tag_pca_info()
            print_structured_quantiles()
        elif name == "chunk6":
            req([
                "tag_game_emb.npy",       # from chunk3
                "item_meta_embs.npy",      # from chunk5
                "structured_app_ids.npy",  # from chunk2
            ])
            print_tag_pca_info()
            print_structured_quantiles()
        elif name == "chunk10":
            req([
                "interactions_df.pkl",       # from chunk1
                "app_id_to_meta_index.json", # from chunk5
                "global_popular_games.json", # from chunk1
                "item_meta_embs.npy",        # from chunk5
                "tag_game_emb.npy",         # from chunk3
                "structured_app_ids.npy",    # from chunk2
                "tag_global_mean.npy",       # from chunk3
            ])
            print_tag_pca_info()
            print_structured_quantiles()

    for name, func in steps_to_run:
        run_step(name, func)

    conn.close()

    total_elapsed = time.time() - start_time
    print(f"Pipeline finished in {total_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
