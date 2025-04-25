import argparse
import json
import time
from datetime import datetime
import httpx
from httpx import AsyncClient, Limits
from bs4 import BeautifulSoup
import mysql.connector
import asyncio
import logging
import sys
import re


# ---- MySQL CONNECTION ----
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="testpassword1",
    database="mydb"
)
cursor = conn.cursor()

# Regex pattern to match leading/trailing whitespace including non-breaking spaces
whitespace_pattern = re.compile(r'^[\s\u00A0]+|[\s\u00A0]+$')

# holds (app_id, tags_json_str) for current batch
tag_upserts: list[tuple[int, str]] = []

# holds all reviews for batch upserts
review_upserts: list[tuple[int,int,int,int]] = []

# 1) Create two handlers: one for INFO→stdout, one for WARNING+→stderr
stdout_handler = logging.StreamHandler(sys.stdout)
stderr_handler = logging.StreamHandler(sys.stderr)
stdout_handler.setLevel(logging.INFO)
stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)

stderr_handler.setLevel(logging.ERROR)  # ERROR and above


# 2) Use the same formatter (with level name in it)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%m-%d %H:%M:%S"
)
stdout_handler.setFormatter(formatter)
stderr_handler.setFormatter(formatter)

# 3) Configure root logger
root = logging.getLogger()
root.setLevel(logging.DEBUG)        # capture everything
root.handlers.clear()              # remove defaults
root.addHandler(stdout_handler)
root.addHandler(stderr_handler)
logging.getLogger("httpx").setLevel(logging.WARNING)


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
        response = httpx.get(BASE_URL, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            games = data.get("response", {}).get("apps", [])

            if not games:
                break 

            all_new_games.extend(games)
            last_appid = games[-1]["appid"]  # steam calls them appid instead of app_id

            logging.info(f"Fetched {len(games)} games, total so far: {len(all_new_games)}")
        else:
            logging.info(f"Failed to fetch data. Status code: {response.status_code}")
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

    logging.info(f"{len(new_games)} new game ids added to the database.")



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
        logging.error(f"Failed to fetch app IDs from the database: {err}")
        return

    app_ids = [row[0] for row in rows]

    # Preparation for Steam API's max calls of 200 calls every 5 minutes. For all games on Steam, approximate run time: 52 hours
    batch_size = 200
    duration = 300 
    start_time = time.time() 

    for i, app_id  in enumerate(app_ids, start=1):    
        if not str(app_id).isdigit():
            logging.error(f"{app_id} appid contains non digits")
            continue 

        if i % 50 == 0:
            logging.info(f"Processed {i} games so far...")

        if i % batch_size == 0:
            conn.commit()
            batch_counter = 0
            elapsed_time = time.time() - start_time
            remaining_time = duration - elapsed_time

            if remaining_time > 0:
                logging.info(f"Pausing for {remaining_time:.2f} seconds...")
                time.sleep(remaining_time)

            start_time = time.time()  # Reset the timer for the next batch

        batch_counter += 1

        details_url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
        try:
            response = httpx.get(details_url, timeout=15)
        except httpx.RequestError as e:
            logging.error(f"Request error for app_id={app_id}: {e}")
            continue

        if response.status_code != 200:
            logging.error(f"Failed to fetch data for app_id={app_id}. HTTP {response.status_code}")
            continue

        try:
            data = response.json()  
        except ValueError:
            logging.error(f"Invalid JSON response for app_id={app_id}")
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
            logging.error(f"MySQL error for app_id={app_id}: {err}")
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
                logging.info("No categories found for ", app_id)

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
                logging.info("No genres found for ", app_id)

    if batch_counter > 0:
        conn.commit()
        batch_counter = 0

    logging.info("Finished storing all game details into the database.")


async def process_reviews(html: str, app_id: int):
    soup = BeautifulSoup(html, "html.parser")
    positive_input = soup.select_one("input#review_summary_num_positive_reviews")
    total_input    = soup.select_one("input#review_summary_num_reviews")

    if positive_input is None or total_input is None:
        no_reviews_div = soup.select_one("div.noReviewsYetTitle")
        if no_reviews_div:
            positive, total, negative = 0, 0, 0
        else:
            logging.error(f"Missing review elements for app_id {app_id}. Skipping reviews.")
            return
    else:
        try:
            positive = int(positive_input.get("value", "0"))
            total    = int(total_input.get("value", "0"))
        except ValueError as ve:
            logging.error(f"Error converting review numbers for app_id {app_id}: {ve}")
            return
        negative = total - positive

    # stash for later batch upsert
    review_upserts.append((app_id, positive, negative, total))


async def process_tags(html: str, app_id: int):
    soup = BeautifulSoup(html, "html.parser")
    tags_elements = soup.select("a.app_tag")

    if not tags_elements:
        logging.error(f"No tags found for app_id {app_id}. Skipping tags.")
        return

    try:
        clean_tags = []
        for el in tags_elements:
            raw = el.get_text()
            # Remove all leading/trailing whitespace including non-breaking space (U+00A0)
            cleaned = whitespace_pattern.sub('', raw)
            if cleaned:
                clean_tags.append(cleaned)

        payload = json.dumps({"tags": clean_tags}, separators=(',', ':'))
        tag_upserts.append((app_id, payload))
    except Exception as e:
        logging.error(f"Error extracting tags for app_id {app_id}: {e}")

async def process_app(client: AsyncClient, app_id: int, semaphore):
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
            try:
                response = await asyncio.wait_for(
                    client.get(url, headers=headers, cookies=cookies, timeout=15),
                    timeout=15 
                )
            except asyncio.TimeoutError:
                logging.error(f"Timeout fetching app {app_id}")
                return

    except Exception as e:
        logging.error(f"Error fetching URL for app_id {app_id}: {e}")
        return

    # Use the single response to scrape both reviews and tags concurrently.
    html = response.text
    await asyncio.gather(
        process_reviews(html, app_id),
        process_tags(html, app_id)
    )
    await response.aclose()


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
        tags_json = VALUES(tags_json),
        last_updated = CURRENT_TIMESTAMP
    """
    try:
        cursor.executemany(sql, batch)
        conn.commit()
    except Exception as e:
        logging.error(f"Error upserting steam_game_tags batch: {e}")
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
        total = VALUES(total)
    """
    try:
        cursor.executemany(sql, batch)
        conn.commit()
    except Exception as e:
        logging.error(f"Error upserting steam_game_reviews batch: {e}")
        conn.rollback()


def refresh_steam_wide_tables():
    rebuild_tags_summary()
    update_bayesian_scores()
    rebuild_year_summary()

def rebuild_tags_summary():
    rebuild_sql = """
        INSERT INTO steam_tag_summary (tag, game_count)
        SELECT tag, COUNT(DISTINCT app_id)
        FROM steam_game_tags,
        JSON_TABLE(tags_json->'$.tags', '$[*]' COLUMNS(tag VARCHAR(255) PATH '$')) AS tag_table
        GROUP BY tag
        ON DUPLICATE KEY UPDATE game_count = VALUES(game_count);
    """
    try:
        cursor.execute(rebuild_sql)
        conn.commit()
        logging.info("Rebuilt steam_tag_summary.")
    except Exception as e:
        logging.error(f"Error rebuilding summary: {e}")
        conn.rollback()


def update_bayesian_scores():
    try:
        # 1. Get global average rating (across all reviewed games)
        cursor.execute("""
            SELECT SUM(positive) / SUM(total)
            FROM steam_game_reviews
            WHERE total > 0
        """)
        global_avg = cursor.fetchone()[0] or 0.0

        # 2. Get average number of total reviews
        cursor.execute("""
            SELECT AVG(total)
            FROM steam_game_reviews
            WHERE total > 0
        """)
        m = cursor.fetchone()[0] or 0

        logging.info(f"Global average rating: {global_avg:.6f}, Review count average: {m:.2f}")

        # 3. Run bayesian update
        update_sql = f"""
            UPDATE steam_game_reviews
            SET bayesian_score = 
                CASE 
                    WHEN total > 0 THEN
                        ((total / (total + {m})) * (positive / total)) +
                        (({m} / (total + {m})) * {global_avg})
                    ELSE
                        0.7
                END
        """
        cursor.execute(update_sql)
        conn.commit()
        logging.info("Bayesian scores updated.")
    except Exception as e:
        logging.error(f"Error updating bayesian scores: {e}")
        conn.rollback()


def rebuild_year_summary():
    """
    Populate steam_year_summary with a row per release year
    and the count of games released that year, excluding games
    scheduled more than two years in the future.
    """
    rebuild_sql = """
        INSERT INTO steam_year_summary (release_year, game_count)
        SELECT
            YEAR(release_date) AS release_year,
            COUNT(*)             AS game_count
        FROM steam_game_details
        WHERE
            release_date IS NOT NULL
            AND YEAR(release_date) BETWEEN 1997 AND (YEAR(CURRENT_DATE()) + 2)
        GROUP BY release_year
        ON DUPLICATE KEY UPDATE
            game_count   = VALUES(game_count),
            last_updated = CURRENT_TIMESTAMP;
    """
    try:
        cursor.execute(rebuild_sql)
        conn.commit()
        logging.info("Rebuilt steam_year_summary.")
    except Exception as e:
        logging.error(f"Error rebuilding year summary: {e}")
        conn.rollback()



async def store_game_reviews_and_tags_in_db(new_ids_only: bool, offset: int = None):
    max_concurrency = 10
    batch_size = 200
    semaphore = asyncio.Semaphore(max_concurrency)

    if new_ids_only:
        base_query = """
            SELECT app_id FROM steam_game_details
            WHERE app_id NOT IN (SELECT DISTINCT app_id FROM steam_game_reviews)
               OR app_id NOT IN (SELECT DISTINCT app_id FROM steam_game_tags)
        """
    else:
        base_query = "SELECT app_id FROM steam_game_details"

    if offset is not None:
        query = f"{base_query} LIMIT {batch_size} OFFSET {offset}"
    else:
        query = base_query

    try:
        cursor.execute(query)
        all_app_ids = [r[0] for r in cursor.fetchall()]
        logging.info(f"Fetched {len(all_app_ids)} app_ids"f"{' at offset '+str(offset) if offset is not None else ''}.")
    except Exception as e:
        logging.error(f"Error fetching app_ids from database: {e}")
        return

    # create HTTPX client
    client = AsyncClient(
        limits=Limits(
            max_connections=max_concurrency,
            max_keepalive_connections=max_concurrency
        )
    )
    pending = []

    for i, app_id in enumerate(all_app_ids, start=1):
        if not str(app_id).isdigit():
            logging.error(f"{app_id} app_id is not numeric. Skipping.")
            continue

        pending.append(process_app(client, app_id, semaphore))

        if i % batch_size == 0:
            t0 = time.time()
            try:
                await asyncio.wait_for(asyncio.gather(*pending), timeout=30.0)
            except asyncio.TimeoutError:
                logging.error("Gather timed out — skipping batch")
                pending.clear()

            pending.clear()

            # commit any detail updates done inside process_app
            try:
                conn.commit()
            except Exception as e:
                logging.error(f"Commit error after batch {i}: {e}")
                conn.rollback()

            logging.info(f"Gathered batch in {time.time()-t0:.1f}s and scraped {i} games so far")

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
            logging.error(f"Final commit error: {e}")
            conn.rollback()

    # upsert and flush any remaining tags and reviews for this batch
    upsert_tags_batch(tag_upserts)
    tag_upserts.clear()
    upsert_reviews_batch(review_upserts)
    review_upserts.clear()

    refresh_steam_wide_tables()

    # cleanup HTTPX client
    await client.aclose()

    logging.info("All app_ids processed and committed.")


 

def main():
    parser = argparse.ArgumentParser(description="Gather Steam Store data.")
    parser.add_argument(
        "--type",
        choices=[
            "all-ids",
            "gather-all-games-info",
            "gather-new-games-info",
            "gather-all-games-tags",
            "gather-new-games-tags",
            "gather-all-games-reviews-and-tags",
            "gather-new-games-reviews-and-tags",
            "refresh-steam-wide-tables",
            "gather-batch-games-reviews-and-tags"
        ],
        help="Gather and store game data from Steam API"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="If using gather-batch-games-reviews-and-tags, start at this OFFSET"
    )
    args = parser.parse_args()

    API_KEY = ""
    try:
        with open('./environment.txt', "r") as file:
            API_KEY = file.read().strip()
    except FileNotFoundError:
        logging.error("Environment file not found at 'environment.txt'.")

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
    elif args.type == "gather-batch-games-reviews-and-tags":
        asyncio.run(store_game_reviews_and_tags_in_db(False, offset=args.offset))
    elif args.type == "refresh-steam-wide-tables":
        refresh_steam_wide_tables()
    else:
        logging.info(
            "Please use --type then one of: all-ids, gather-new-games-info, "
            "gather-all-games-info, gather-all-games-tags, gather-new-games-tags, "
            "gather-all-games-reviews-and-tags, gather-new-games-reviews-and-tags, "
            "refresh-steam-wide-tables"
        )

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
