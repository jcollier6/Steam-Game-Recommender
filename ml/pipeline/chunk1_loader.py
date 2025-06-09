import os
from typing import Dict, List
import pandas as pd
import requests

from .utils import save_json


def load_tags_table(connection) -> Dict[str, int]:
    df_tags = pd.read_sql('SELECT tag_id FROM tags', connection)
    tag_ids = df_tags['tag_id'].unique().tolist()
    tag_id_map = {tag_id: idx for idx, tag_id in enumerate(tag_ids)}
    os.makedirs('data', exist_ok=True)
    save_json(tag_id_map, 'data/tag_id_map.json')
    return tag_id_map


def load_games_table(connection) -> pd.DataFrame:
    query = """
        SELECT app_id, positive_review_count, negative_review_count,
               price_usd, discount_usd, release_date,
               tags, short_description, long_description
        FROM games
    """
    df = pd.read_sql(query, connection)
    return df


def load_interactions_table(connection) -> pd.DataFrame:
    """Load the interactions table which stores records for multiple users.

    Each row contains a user_id and their interaction with a specific game.
    """
    query = """
        SELECT user_id, app_id, playtime_forever, playtime_2weeks,
               wishlisted, wishlist_priority, date_added_to_wishlist,
               last_played
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

