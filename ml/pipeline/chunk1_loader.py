import os
import hashlib
import json
from typing import Dict, List
import pandas as pd
import requests

from .utils import save_json


def load_tags_table(connection) -> Dict[str, int]:
    df_tags = pd.read_sql(
        'SELECT tag_id FROM steam_tag_summary WHERE tag_id IS NOT NULL',
        connection,
    )
    tag_ids = df_tags['tag_id'].dropna().astype(int).unique().tolist()
    tag_id_map = {str(tag_id): idx for idx, tag_id in enumerate(tag_ids)}
    os.makedirs('ml/data', exist_ok=True)
    save_json(tag_id_map, 'ml/data/tag_id_map.json')
    return tag_id_map


def load_games_table(connection) -> pd.DataFrame:
    query = """
        SELECT 
            g.app_id,
            g.is_free AS is_free,
            r.positive AS pos_reviews,
            r.negative AS neg_reviews,
            r.total,
            r.bayesian_score,

            CASE
                WHEN g.is_free = 1 THEN 0
                ELSE JSON_UNQUOTE(JSON_EXTRACT(g.price_overview, '$.final')) / 100.0
            END AS price_usd,

            CASE
                WHEN g.is_free = 1 THEN 0
                ELSE JSON_UNQUOTE(JSON_EXTRACT(g.price_overview, '$.initial')) / 100.0
            END AS base_price_usd,

            g.release_date,
            g.fetched_at AS snapshot_date,
            g.short_description,
            g.detailed_description AS long_description,

            GROUP_CONCAT(CONCAT(ts.tag_id, ':', gt.tag_rank) ORDER BY gt.tag_rank SEPARATOR ',') AS tags_with_ranks

        FROM steam_game_details g
        LEFT JOIN steam_game_reviews r ON g.app_id = r.app_id
        LEFT JOIN steam_game_tags gt ON g.app_id = gt.app_id
        LEFT JOIN steam_tag_summary ts ON gt.tag = ts.tag
        GROUP BY g.app_id
    """
    df = pd.read_sql(query, connection)

    # Filter out games we don't want downstream
    # 1) Drop games with zero total reviews (pos + neg == 0 or both null)
    pos = pd.to_numeric(df['pos_reviews'], errors='coerce').fillna(0)
    neg = pd.to_numeric(df['neg_reviews'], errors='coerce').fillna(0)
    mask_zero_reviews = (pos + neg) <= 0

    # 2) Drop games where base_price_usd is null for non-free titles
    is_free = pd.to_numeric(df['is_free'], errors='coerce').fillna(0).astype(int)
    mask_missing_price_nonfree = (is_free == 0) & (df['base_price_usd'].isna())
    
    # 3) Drop games with missing release_date
    mask_missing_release = df['release_date'].isna()

    # 4) Drop games with no tags (NULL or empty)
    mask_no_tags = df['tags_with_ranks'].isna() | (df['tags_with_ranks'] == '')
    # 5) Drop games with too-few tags (<3); max is ~20 in this dataset
    #    Compute count as commas+1 for non-empty strings
    tag_counts = (
        df['tags_with_ranks']
        .fillna('')
        .astype(str)
        .apply(lambda s: 0 if s == '' else (s.count(',') + 1))
    )
    mask_few_tags = tag_counts < 3

    dropped_zero_reviews = int(mask_zero_reviews.sum())
    dropped_missing_price = int(mask_missing_price_nonfree.sum())
    dropped_missing_release = int(mask_missing_release.sum())
    dropped_no_tags = int(mask_no_tags.sum())
    dropped_few_tags = int((~mask_no_tags & mask_few_tags).sum())

    # Attach drop stats for printing later in main
    df.attrs['drop_stats'] = {
        'zero_reviews': dropped_zero_reviews,
        'missing_price_nonfree': dropped_missing_price,
        'missing_release_date': dropped_missing_release,
        'no_tags': dropped_no_tags,
        'few_tags': dropped_few_tags,
    }
    keep_mask = ~(mask_zero_reviews | mask_missing_price_nonfree | mask_missing_release | mask_no_tags | mask_few_tags)
    df = df.loc[keep_mask].reset_index(drop=True)

    def parse_tag_ranks(s: str) -> List[tuple[int, int]]:
        if not s:
            return []
        pairs = []
        for part in s.split(','):
            try:
                tag_id_str, rank_str = part.split(':')
                pairs.append((int(tag_id_str), int(rank_str)))
            except ValueError:
                continue
        return pairs

    df['tags'] = df['tags_with_ranks'].apply(parse_tag_ranks)
    return df


def load_interactions_table(connection) -> pd.DataFrame:
    """Load the interactions table which stores records for multiple users.

    Each row contains a user_id and their interaction with a specific game.
    """
    query = """
        SELECT user_id, app_id, playtime_forever, playtime_2weeks,
               wishlisted, wishlist_priority, date_added_to_wishlist
        FROM user_interactions
    """
    df = pd.read_sql(query, connection)
    return df


def fetch_global_popular_games(api_key: str) -> List[int]:
    """Fetch the top most played games from the Steam Charts API."""
    url = (
        "https://api.steampowered.com/ISteamChartsService/"
        "GetMostPlayedGames/v1/"
    )
    resp = requests.get(url, params={"key": api_key})
    resp.raise_for_status()
    data = resp.json()
    games = data.get("response", {}).get("ranks", [])
    app_ids = [g.get("appid") for g in games if g.get("appid")]
    save_json(app_ids, "ml/data/global_popular_games.json")
    return app_ids


def main(connection):
    tag_id_map = load_tags_table(connection)
    games_df = load_games_table(connection)
    interactions_df = load_interactions_table(connection)
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise EnvironmentError('API_KEY environment variable not set')
    fetch_global_popular_games(api_key)

    games_df.to_pickle('ml/data/games_df.pkl')
    interactions_df.to_pickle('ml/data/interactions_df.pkl')
    print('✅ Chunk 1 data saved')

    # Print drop stats after filtering
    drop_stats = getattr(games_df, 'attrs', {}).get('drop_stats', None)
    if drop_stats is not None:
        print(
            "Dropped games — "
            f"zero_reviews={drop_stats.get('zero_reviews', 0)}, "
            f"missing_price_nonfree={drop_stats.get('missing_price_nonfree', 0)}, "
            f"missing_release_date={drop_stats.get('missing_release_date', 0)}",
            flush=True,
        )

    # Quick checks: row counts, nulls, unique appids, schema hashes
    def table_row_counts(conn, tables: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in tables:
            try:
                df_cnt = pd.read_sql(f"SELECT COUNT(*) AS cnt FROM {t}", conn)
                counts[t] = int(df_cnt['cnt'].iloc[0])
            except Exception as e:
                counts[t] = -1
                print(f"[warn] Failed to count rows for {t}: {e}")
        return counts

    def df_null_percents(df: pd.DataFrame) -> Dict[str, float]:
        if df is None or df.empty:
            return {}
        return {c: float(df[c].isna().mean() * 100.0) for c in df.columns}

    def schema_hash_from_df(df: pd.DataFrame) -> str:
        signature = {
            'columns': list(df.columns),
            'dtypes': {c: str(df[c].dtype) for c in df.columns},
        }
        raw = json.dumps(signature, sort_keys=True).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()[:16]

    def schema_hash_from_table(conn, table: str) -> str:
        try:
            desc = pd.read_sql(f"DESCRIBE {table}", conn)
            sig = desc[['Field', 'Type', 'Null', 'Key', 'Default', 'Extra']].fillna('').astype(str).values.tolist()
            raw = json.dumps(sig, sort_keys=True).encode('utf-8')
            return hashlib.sha256(raw).hexdigest()[:16]
        except Exception as e:
            print(f"[warn] Failed to get schema for {table}: {e}")
            return ""

    tables = [
        'steam_tag_summary',
        'steam_game_details',
        'steam_game_reviews',
        'steam_game_tags',
        'user_interactions',
    ]

    counts = table_row_counts(connection, tables)

    print("\n=== Quick Checks (Chunk 1) ===", flush=True)
    print("Row counts per table:")
    for t in tables:
        v = counts.get(t, -1)
        print(f"- {t}: {v}")

    # Percent nulls
    print("\n% nulls in games_df:")
    for col, pct in sorted(df_null_percents(games_df).items()):
        print(f"- {col}: {pct:.2f}%")

    print("\n% nulls in interactions_df:")
    for col, pct in sorted(df_null_percents(interactions_df).items()):
        print(f"- {col}: {pct:.2f}%")

    # Unique app_id
    if 'app_id' in games_df.columns:
        n_unique = games_df['app_id'].nunique()
        n_rows = len(games_df)
        dupes = n_rows - n_unique
        print(f"\nunique(app_id) in games_df: {n_unique} (rows: {n_rows}, duplicates: {dupes})")
    else:
        print("\n[warn] games_df has no 'app_id' column")

    # Schema snapshot hashes
    print("\nSchema snapshot hashes:")
    print(f"- games_df: {schema_hash_from_df(games_df)}")
    print(f"- interactions_df: {schema_hash_from_df(interactions_df)}")
    for t in tables:
        sh = schema_hash_from_table(connection, t)
        if sh:
            print(f"- {t}: {sh}")

if __name__ == '__main__':
    import mysql.connector
    conn = mysql.connector.connect(
        host=os.getenv('MYSQL_HOST'),
        user=os.getenv('MYSQL_USER'),
        password=os.getenv('MYSQL_PASSWORD'),
        database=os.getenv('MYSQL_DATABASE')
    )
    main(conn)
    conn.close()

