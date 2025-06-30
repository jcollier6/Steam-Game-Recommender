import os
import time
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
    start_time = time.time()
    step = 0
    total_steps = 7

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

    conn = mysql.connector.connect(
        host=os.getenv('MYSQL_HOST'),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DATABASE'),
    )
    run_step("chunk1", lambda: chunk1_loader.main(conn))
    conn.close()

    games_df = pd.read_pickle('ml/data/games_df.pkl')
    run_step("chunk2", lambda: chunk2_structured.structured_transform(games_df, get_current_utc_date()))
    run_step("chunk3", lambda: chunk3_tags_pca.run_tag_pca(games_df))
    run_step("chunk4", lambda: chunk4_text_pca.run_text_pca(games_df))
    run_step("chunk5", chunk5_item_meta.run_item_meta_embedding)
    run_step("chunk6", chunk6_faiss.build_faiss_indices)

    epochs = int(os.getenv('NUM_EPOCHS', '1'))
    run_step("chunk10", lambda: chunk10_train.train_model(num_epochs=epochs))

    total_elapsed = time.time() - start_time
    print(f"Pipeline finished in {total_elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
