import os
import time
import argparse
from .utils import get_current_utc_date
import pandas as pd
import mysql.connector

from . import (
    chunk1_loader,
    chunk2_structured,
    chunk3_tags_pca,
    chunk4_text_pca,
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
        help="first chunk to execute (1–6 or 10)",
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

    epochs = int(os.getenv("NUM_EPOCHS", "1"))

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
        ("chunk4", lambda: chunk4_text_pca.run_text_pca(load_games_df())),
        ("chunk5", chunk5_item_meta.run_item_meta_embedding),
        ("chunk6", chunk6_faiss.build_faiss_indices),
        ("chunk10", lambda: chunk10_train.train_model(num_epochs=epochs)),
    ]

    steps_to_run = steps[args.start - 1 :]
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

    for name, func in steps_to_run:
        run_step(name, func)

    conn.close()

    total_elapsed = time.time() - start_time
    print(f"Pipeline finished in {total_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
