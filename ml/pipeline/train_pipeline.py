import os
from .utils import get_current_utc_date
import pandas as pd
import mysql.connector

from . import chunk1_loader, chunk2_structured, chunk3_tags_pca, chunk4_text_pca, chunk5_item_meta, chunk6_faiss, chunk10_train


def main():
    conn = mysql.connector.connect(
        host=os.getenv('MYSQL_HOST'),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DATABASE'),
    )
    chunk1_loader.main(conn)
    conn.close()

    games_df = pd.read_pickle('ml/data/games_df.pkl')
    chunk2_structured.structured_transform(games_df, get_current_utc_date())
    chunk3_tags_pca.run_tag_pca(games_df)
    chunk4_text_pca.run_text_pca(games_df)
    chunk5_item_meta.run_item_meta_embedding()
    chunk6_faiss.build_faiss_indices()

    epochs = int(os.getenv('NUM_EPOCHS', '1'))
    chunk10_train.train_model(num_epochs=epochs)


if __name__ == '__main__':
    main()
