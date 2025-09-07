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
        ("chunk3", lambda: chunk3_tags_pca.run_tag_pca(load_games_df())),
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
                    print("Structured thresholds:")
                    phg = q.get('p_high_global')
                    phm = q.get('p_high_by_metric', {})
                    if phg is not None:
                        print(f" - global_p_high={phg}")
                    # Print only metrics that use winsorization (skip discount_percent)
                    for key in [
                        'positive_review_count',
                        'negative_review_count',
                        'price_usd',
                    ]:
                        if key in q:
                            lo = q[key].get('low')
                            hi = q[key].get('high')
                            zeros = q[key].get('zeros')
                            print(f" - {key}: low={lo} high={hi} zeros={zeros}")
                    # Show per-metric p_high overrides (excluding discount_percent)
                    if phm:
                        cnt = phm.get('counts')
                        prc = phm.get('price_usd')
                        if cnt is not None:
                            print(f" - counts_p_high={cnt}")
                        if prc is not None:
                            print(f" - price_p_high={prc}")
                except Exception as e:
                    print(f"[warn] Could not read structured_quantiles.json: {e}")

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
            print_structured_quantiles()
        elif name == "chunk6":
            req([
                "tag_game_emb.npy",       # from chunk3
                "item_meta_embs.npy",      # from chunk5
                "structured_app_ids.npy",  # from chunk2
            ])
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
            print_structured_quantiles()

    for name, func in steps_to_run:
        run_step(name, func)

    conn.close()

    total_elapsed = time.time() - start_time
    print(f"Pipeline finished in {total_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
