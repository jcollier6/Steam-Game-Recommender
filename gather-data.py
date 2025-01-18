import argparse
import json
import time
from datetime import datetime

import requests
import mysql.connector
from fastapi import FastAPI

# ---- MySQL CONNECTION ----
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="testpassword1",
    database="mydb"
)
cursor = conn.cursor()


def gather_all_game_ids(API_KEY):
    BASE_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
    
    params = {
        "key": API_KEY,
        "include_games": True,
        "include_dlc": False,
        "include_software": False,
        "include_videos": False,
        "include_hardware": False,
        "max_results": 50000, 
    }

    all_new_games = []
    last_appid = 0

    while True:
        params["last_appid"] = last_appid

        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            games = data.get("response", {}).get("apps", [])

            if not games:
                break 

            all_new_games.extend(games)
            last_appid = games[-1]["appid"]  # steam calls them appid instead of app_id

            print(f"Fetched {len(games)} games, total so far: {len(all_new_games)}")
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            break

    current_unique_games = {game["appid"]: game for game in all_new_games}.values()

    cursor.execute("SELECT app_id FROM all_steam_game_ids")
    existing_appids = {row[0] for row in cursor.fetchall()}

    new_games = [game for game in current_unique_games if game["appid"] not in existing_appids]

    for game in new_games:
        cursor.execute(
            "INSERT INTO all_steam_game_ids (app_id, name) VALUES (%s, %s)",
            (game["appid"], game["name"])
        )
    
    conn.commit()

    print(f"{len(new_games)} new game ids added to the database.")



def store_game_details_in_db(new_ids_only):
    """
    Fetches app IDs from the 'all_steam_game_ids' database table, retrieves details for each app_id via the Steam API,
    and upserts into 'steam_game_details'. Also normalizes categories and genres into 'steam_game_categories' & 'steam_game_genres'.

    Parameters:
        new_ids_only (bool): If True, only fetch app_ids that are in 'all_steam_game_ids' but not in 'steam_game_details'.
    """
    batch_counter = 0

    try:
        if new_ids_only:
            query = """
            SELECT a.app_id
            FROM all_steam_game_ids a
            LEFT JOIN steam_game_details s ON a.app_id = s.app_id
            WHERE s.app_id IS NULL
            """
        else:
            query = "SELECT app_id FROM all_steam_game_ids"

        cursor.execute(query)
        rows = cursor.fetchall() 
    except mysql.connector.Error as err:
        print(f"Failed to fetch app IDs from the database: {err}")
        return

    app_ids = [row[0] for row in rows]

    # Preparation for Steam API's max calls of 200 calls every 5 minutes. For all games on Steam, approximate run time: 52 hours
    batch_size = 200
    duration = 300 
    start_time = time.time() 

    for i, app_id  in enumerate(app_ids, start=1):    
        if not str(app_id).isdigit():
            print(f"{app_id} appid contains non digits")
            continue 

        if i % 50 == 0:
            print(f"Processed {i} games so far...")

        if i % batch_size == 0:
            conn.commit()
            batch_counter = 0
            elapsed_time = time.time() - start_time
            remaining_time = duration - elapsed_time

            if remaining_time > 0:
                print(f"Pausing for {remaining_time:.2f} seconds...")
                time.sleep(remaining_time)

            start_time = time.time()  # Reset the timer for the next batch

        batch_counter += 1

        details_url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        try:
            response = requests.get(details_url, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Request error for app_id={app_id}: {e}")
            continue

        if response.status_code != 200:
            print(f"Failed to fetch data for app_id={app_id}. HTTP {response.status_code}")
            continue

        try:
            data = response.json()  
        except ValueError:
            print(f"Invalid JSON response for app_id={app_id}")
            continue

        app_key = str(app_id)
        app_info = data.get(app_key, {})
        # If this occurs, it is not accessible on the Steam store either because of region lock or it has been removed from the store but Steam didn't remove it from their app_id list.
        if not app_info.get("success", False):
            continue 

        details = app_info.get("data", {})

        # Extract fields
        name = details.get("name", "")
        is_free = 1 if details.get("is_free", False) else 0

        # Extract release_date info
        release_date_info = details.get("release_date", {})
        coming_soon = 1 if release_date_info.get("coming_soon", False) else 0
        
        # Parse the "Jul 9, 2013" style date
        raw_date_str = release_date_info.get("date", "")
        release_date_date = None
        if raw_date_str:
            try:
                release_date_date = datetime.strptime(raw_date_str, "%b %d, %Y").date()
            except ValueError:
                pass

        rec_data = details.get("recommendations", {})
        recommendations_count = rec_data.get("total", 0)

        raw_data_json = json.dumps(details)

        upsert_sql = """
        INSERT INTO steam_game_details (
            app_id,
            name,
            coming_soon,
            release_date_date,
            is_free,
            recommendations,
            raw_json
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            name = VALUES(name),
            coming_soon = VALUES(coming_soon),
            release_date_date = VALUES(release_date_date),
            is_free = VALUES(is_free),
            recommendations = VALUES(recommendations),
            raw_json = VALUES(raw_json),
            fetched_at = CURRENT_TIMESTAMP
        """
        vals = (
            app_id,
            name,
            coming_soon,
            release_date_date,  
            is_free,
            recommendations_count,
            raw_data_json
        )

        try:
            cursor.execute(upsert_sql, vals)
        except mysql.connector.Error as err:
            print(f"MySQL error for app_id={app_id}: {err}")
            continue

        # Delete old categories and genres for simplicity
        delete_cats_sql = "DELETE FROM steam_game_categories WHERE app_id=%s"
        delete_gens_sql = "DELETE FROM steam_game_genres WHERE app_id=%s"
        cursor.execute(delete_cats_sql, (app_id,))
        cursor.execute(delete_gens_sql, (app_id,))

        # Insert categories
        categories = details.get("categories", [])
        for cat_obj in categories:
            cat_name = cat_obj.get("description", "")
            if cat_name:
                cat_insert_sql = """
                INSERT INTO steam_game_categories (app_id, category_name)
                VALUES (%s, %s)
                """
                cursor.execute(cat_insert_sql, (app_id, cat_name))

        # Insert genres
        genres = details.get("genres", [])
        for gen_obj in genres:
            gen_name = gen_obj.get("description", "")
            if gen_name:
                gen_insert_sql = """
                INSERT INTO steam_game_genres (app_id, genre_name)
                VALUES (%s, %s)
                """
                cursor.execute(gen_insert_sql, (app_id, gen_name))

    if batch_counter > 0:
        conn.commit()
        batch_counter = 0

    print("Finished storing all game details into the database.")



def store_game_tags_in_db(new_ids_only):
    """
    Store game tags in the database. Optionally fetch only missing app_ids.
    Parameters:
        new_ids_only (bool): If True, only fetch app_ids that are in 'steam_game_details' but not in 'steam_game_tags'.
    """
    if new_ids_only:
        query = """
        SELECT app_id FROM steam_game_details 
        WHERE app_id NOT IN (SELECT DISTINCT app_id FROM steam_game_tags)
        """
    else:
        query = "SELECT app_id FROM steam_game_details"

    cursor.execute(query)
    app_ids = cursor.fetchall()

    base_url = "https://steamspy.com/api.php?request=appdetails&appid={}"

    commit_interval = 200
    processed_count = 0

    for (app_id,) in app_ids:
        try:
            response = requests.get(base_url.format(app_id))
            response.raise_for_status()
            data = response.json()
            tags = data.get("tags", {})

            for tag in tags:
                upsert_query = """
                INSERT INTO steam_game_tags (app_id, tag) 
                VALUES (%s, %s) 
                ON DUPLICATE KEY UPDATE tag=VALUES(tag)
                """
                cursor.execute(upsert_query, (app_id, tag))

            processed_count += 1

            if processed_count % commit_interval == 0:
                conn.commit()
                print(f"Processed {processed_count} games tags so far...")

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for app_id {app_id}: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for app_id {app_id}: {e}")

    conn.commit()
    


def main():
    parser = argparse.ArgumentParser(description="Gather Steam Store data.")
    parser.add_argument("--type", choices=["all-ids", "gather-all-games-info", "gather-new-games-info", "gather-games-tags", "gather-new-games-tags"], help="Gather and store game data from Steam API")
    args = parser.parse_args()

    API_KEY = ""
    try:
        with open('./environment.txt', "r") as file:
            API_KEY = file.read().strip()
    except FileNotFoundError:
        print("Environment file not found at 'environment.txt'.")

    if args.type == "all-ids":
        gather_all_game_ids(API_KEY)
    elif args.type == "gather-all-games-info":
        store_game_details_in_db(False)
    elif args.type == "gather-new-games-info":
        store_game_details_in_db(True)
    elif args.type == "gather-games-tags":
        store_game_tags_in_db(True)
    elif args.type == "gather-new-games-tags":
        store_game_tags_in_db(True)
    else:
        print("Please use --type all-ids or --type gather-new-games-info or --type gather-all-games-info or --type gather-games-tags --type gather-new-games-tags")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
