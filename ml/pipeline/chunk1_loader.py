import os
from typing import Dict, List
import pandas as pd
import requests

from .utils import save_json


def load_tags_table(connection) -> Dict[str, int]:
    df_tags = pd.read_sql('SELECT tag_id FROM steam_tag_summary', connection)
    tag_ids = df_tags['tag_id'].unique().tolist()
    tag_id_map = {tag_id: idx for idx, tag_id in enumerate(tag_ids)}
    os.makedirs('data', exist_ok=True)
    save_json(tag_id_map, 'ml/data/tag_id_map.json')
    return tag_id_map


def load_games_table(connection) -> pd.DataFrame:
    query = """
        SELECT 
            g.app_id,
            r.positive AS positive_review_count,
            r.negative AS negative_review_count,
            r.total,
            r.bayesian_score,

            CASE
                WHEN g.is_free = 1 THEN 0
                ELSE JSON_UNQUOTE(JSON_EXTRACT(g.price_overview, '$.final')) / 100.0
            END AS price_usd,

            CASE
                WHEN g.is_free = 1 THEN 0
                ELSE (JSON_UNQUOTE(JSON_EXTRACT(g.price_overview, '$.initial')) - JSON_UNQUOTE(JSON_EXTRACT(g.price_overview, '$.final'))) / 100.0
            END AS discount_usd,

            g.release_date,
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
        FROM interactions
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
    save_json(app_ids, "data/global_popular_games.json")
    return app_ids


def main(connection):
    tag_id_map = load_tags_table(connection)
    games_df = load_games_table(connection)
    interactions_df = load_interactions_table(connection)
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise EnvironmentError('API_KEY environment variable not set')
    fetch_global_popular_games(api_key)

    games_df.to_pickle('data/games_df.pkl')
    interactions_df.to_pickle('data/interactions_df.pkl')
    print('✅ Chunk 1 data saved')

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

