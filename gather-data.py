import argparse
import json
import time
from datetime import datetime
import requests
import mysql.connector
from requests_html import AsyncHTMLSession
import asyncio


# ---- MySQL CONNECTION ----
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="testpassword1",
    database="mydb"
)
cursor = conn.cursor()


# holds (app_id, tags_json_str) for current batch
tag_upserts: list[tuple[int, str]] = []

# holds all reviews for batch upserts
review_upserts: list[tuple[int,int,int,int]] = []

def gather_all_game_ids(API_KEY: str):
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



def store_game_details_in_db(new_ids_only: bool):
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
            LEFT JOIN steam_game_details d ON a.app_id = d.app_id
            LEFT JOIN steam_game_genres g ON a.app_id = g.app_id
            LEFT JOIN steam_game_categories c ON a.app_id = c.app_id
            WHERE d.app_id IS NULL
            OR g.app_id IS NULL
            OR c.app_id IS NULL
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

        # Extract price
        price_overview = details.get("price_overview", {})
        price_formatted = price_overview.get("final_formatted", "")

        if price_formatted and price_formatted.endswith("USD"):
            # Remove the trailing "USD" and any trailing whitespace
            price_usd = price_formatted[:-3].strip()
        else:
            price_usd = price_formatted or ""

        # Extract release_date info
        release_date_info = details.get("release_date", {})
        coming_soon = 1 if release_date_info.get("coming_soon", False) else 0
        
        # Parse the "Jul 9, 2013" style date
        raw_date_str = release_date_info.get("date", "")
        release_date = None
        if raw_date_str:
            try:
                release_date = datetime.strptime(raw_date_str, "%b %d, %Y").date()
            except ValueError:
                pass

        # Extract images
        header_image = details.get("header_image", "")

        screenshots = details.get("screenshots", [])

        # Extract up to 4 screenshot URLs (path_full). If fewer exist or "screenshots" is missing, use an empty string.
        screenshot1 = screenshots[0].get("path_full", "") if len(screenshots) > 0 else ""
        screenshot2 = screenshots[1].get("path_full", "") if len(screenshots) > 1 else ""
        screenshot3 = screenshots[2].get("path_full", "") if len(screenshots) > 2 else ""
        screenshot4 = screenshots[3].get("path_full", "") if len(screenshots) > 3 else ""


        # Extract recommendations
        rec_data = details.get("recommendations", {})
        recommendations_count = rec_data.get("total", 0)

        raw_data_json = json.dumps(details)

        upsert_sql = """
        INSERT INTO steam_game_details (
            app_id,
            name,
            coming_soon,
            release_date,
            is_free,
            price_usd,
            recommendations,
            raw_json,
            header_image,
            screenshot1,
            screenshot2,
            screenshot3,
            screenshot4
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            name = VALUES(name),
            coming_soon = VALUES(coming_soon),
            release_date = VALUES(release_date),
            is_free = VALUES(is_free),
            price_usd = VALUES(price_usd),
            recommendations = VALUES(recommendations),
            raw_json = VALUES(raw_json),
            fetched_at = CURRENT_TIMESTAMP,
            header_image = VALUES(header_image),
            screenshot1 = VALUES(screenshot1),
            screenshot2 = VALUES(screenshot2),
            screenshot3 = VALUES(screenshot3),
            screenshot4 = VALUES(screenshot4)
        """
        vals = (
            app_id,
            name,
            coming_soon,
            release_date,  
            is_free,
            price_usd,
            recommendations_count,
            raw_data_json,
            header_image,
            screenshot1,
            screenshot2,
            screenshot3,
            screenshot4
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
            else:
                print("No categories found for ", app_id)

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
            else:
                print("No genres found for ", app_id)

    if batch_counter > 0:
        conn.commit()
        batch_counter = 0

    print("Finished storing all game details into the database.")


async def process_reviews(response, app_id: int):
    positive_input = response.html.find("input#review_summary_num_positive_reviews", first=True)
    total_input    = response.html.find("input#review_summary_num_reviews", first=True)

    if positive_input is None or total_input is None:
        no_reviews_div = response.html.find("div.noReviewsYetTitle", first=True)
        if no_reviews_div:
            positive, total, negative = 0, 0, 0
        else:
            print(f"Missing review elements for app_id {app_id}. Skipping reviews.")
            return
    else:
        try:
            positive = int(positive_input.attrs.get("value", "0"))
            total    = int(total_input.attrs.get("value", "0"))
        except ValueError as ve:
            print(f"Error converting review numbers for app_id {app_id}: {ve}")
            return
        negative = total - positive

    # stash for later batch upsert
    review_upserts.append((app_id, positive, negative, total))


async def process_tags(response, app_id: int):
    tags_elements = response.html.find("a.app_tag")
    if not tags_elements:
        print(f"No tags found for app_id {app_id}. Skipping tags.")
        return
    try:
        tags = {el.text for el in tags_elements}
        payload = json.dumps({"tags": list(tags)})
        tag_upserts.append((app_id, payload))
    except Exception as e:
        print(f"Error extracting tags for app_id {app_id}: {e}")


async def process_app(asession: AsyncHTMLSession, app_id: int, semaphore):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/105.0.5195.102 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://store.steampowered.com/",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Sec-Fetch-Dest": "document"
    }
    
    # Cookies to enforce U.S. region and bypass the age gate.
    cookies = {
        "Steam_Language": "english",
        "country": "US",
        "wants_mature_content": "1",
        "birthtime": "631152000",       # Unix timestamp for Jan 1, 1990
        "lastagecheckage": "1-0-1990"
    }

    url = f"https://store.steampowered.com/app/{app_id}"
    try:
        async with semaphore:
            response = await asession.get(url, headers=headers, cookies=cookies, timeout=10)
    except Exception as e:
        print(f"Error fetching URL for app_id {app_id}: {e}")
        return

    # Use the single response to scrape both reviews and tags concurrently.
    await asyncio.gather(
        process_reviews(response, app_id),
        process_tags(response, app_id)
    )


def upsert_tags_batch(batch: list[tuple[int,str]]):
    """
    batch is a list of (app_id, tags_json_str)
    """
    if not batch:
        return
    sql = """
      INSERT INTO steam_game_tags (app_id, tags_json)
      VALUES (%s, %s)
      ON DUPLICATE KEY UPDATE
        tags_json    = VALUES(tags_json),
        last_updated = CURRENT_TIMESTAMP
    """
    try:
        cursor.executemany(sql, batch)
        conn.commit()
    except Exception as e:
        print(f"Error upserting steam_game_tags batch: {e}")
        conn.rollback()


def upsert_reviews_batch(batch: list[tuple[int,int,int,int]]):
    """
    batch is a list of (app_id, positive, negative, total)
    """
    if not batch:
        return
    sql = """
      INSERT INTO steam_game_reviews (app_id, positive, negative, total)
      VALUES (%s, %s, %s, %s)
      ON DUPLICATE KEY UPDATE
        positive = VALUES(positive),
        negative = VALUES(negative),
        total    = VALUES(total)
    """
    try:
        cursor.executemany(sql, batch)
        conn.commit()
    except Exception as e:
        print(f"Error upserting steam_game_reviews batch: {e}")
        conn.rollback()


def rebuild_tags_summary():
    rebuild_sql = """
      INSERT INTO steam_tag_summary (tag, game_count)
      SELECT tag, COUNT(DISTINCT app_id)
        FROM steam_game_tags
       GROUP BY tag
      ON DUPLICATE KEY UPDATE
        game_count = VALUES(game_count)
    """
    try:
        cursor.execute(rebuild_sql)
        conn.commit()
        print("Rebuilt steam_tag_summary.")
    except Exception as e:
        print(f"Error rebuilding summary: {e}")
        conn.rollback()


async def store_game_reviews_and_tags_in_db(new_ids_only: bool):
    max_concurrency = 10      # Limit concurrent HTTP requests
    batch_size = 200          # Commit to DB every 200 processed games
    semaphore = asyncio.Semaphore(max_concurrency)

    if new_ids_only:
        query = """
            SELECT app_id FROM steam_game_details 
            WHERE app_id NOT IN (SELECT DISTINCT app_id FROM steam_game_reviews)
            OR app_id NOT IN (SELECT DISTINCT app_id FROM steam_game_tags)
        """
    else:
        query = "SELECT app_id FROM steam_game_details"

    try:
        cursor.execute(query)
        all_app_ids = [r[0] for r in cursor.fetchall()]
        print(f"Found {len(all_app_ids)} app_ids to process.")
    except Exception as e:
        print(f"Error fetching app_ids from database: {e}")
        return

    asession = AsyncHTMLSession()
    pending = []
    tags_to_flush = []

    # scrape in batches
    for i, app_id in enumerate(all_app_ids, start=1):
        if not str(app_id).isdigit():
            print(f"{app_id} app_id is not numeric. Skipping.")
            continue

        pending.append(process_app(asession, app_id, semaphore))

        # mark for flush after scrape
        tags_to_flush.append(app_id)

        if i % batch_size == 0:
            # wait for this batch
            await asyncio.gather(*pending)
            pending.clear()

            # commit any detail updates done inside process_app
            try:
                conn.commit()
            except Exception as e:
                print(f"Commit error after batch {i}: {e}")
                conn.rollback()
            print(f"Scraped {i} games")

            # upsert and flush all tags and reviews for this batch
            upsert_tags_batch(tag_upserts)
            tag_upserts.clear()
            upsert_reviews_batch(review_upserts)
            review_upserts.clear()

    # final flush
    if pending:
        await asyncio.gather(*pending)
        try:
            conn.commit()
        except Exception as e:
            print(f"Final commit error: {e}")
            conn.rollback()

    # upsert and flush any remaining tags and reviews for this batch
    upsert_tags_batch(tag_upserts)
    tag_upserts.clear()
    upsert_reviews_batch(review_upserts)
    review_upserts.clear()

    rebuild_tags_summary()

    print("All app_ids processed and committed.")


 

def main():
    parser = argparse.ArgumentParser(description="Gather Steam Store data.")
    parser.add_argument("--type", choices=["all-ids", "gather-all-games-info", "gather-new-games-info", "gather-all-games-tags", "gather-new-games-tags", "gather-all-games-reviews-and-tags", "gather-new-games-reviews-and-tags"], help="Gather and store game data from Steam API")
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
    elif args.type == "gather-all-games-reviews-and-tags":
        asyncio.run(store_game_reviews_and_tags_in_db(False))
    elif args.type == "gather-new-games-reviews-and-tags":
        asyncio.run(store_game_reviews_and_tags_in_db(True))
    else:
        print("Please use --type all-ids or --type gather-new-games-info or --type gather-all-games-info or --type gather-all-games-tags or --type gather-new-games-tags or  --type gather-all-games-reviews-and-tags or --type gather-new-games-reviews-and-tags")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
