from fastapi import FastAPI, HTTPException
import mysql.connector
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import json
from collections import Counter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Global init here
    initialize_global_game_data()
    yield
    # Optional cleanup logic here

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%m-%d %H:%M:%S"
)

# --- Database Connection Helpers ---
def get_db_connection():
    """Return a new MySQL connection from the pool."""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="testpassword1",
        database="mydb",
        pool_name="mypool",
        pool_size=5
    )

def query_db(query, params=None, dictionary=True):
    """Run a query and return all results. Automatically opens and closes a connection."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=dictionary)
    cursor.execute(query, params or ())
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result



@app.get("/all_tags")
def get_all_tags():
    return all_unique_tags

@app.get("/recommended_games")
def get_recommended_games():
    try:
        data = get_games_additional_info(df_recommended_games)
        return JSONResponse(content=data)  
    except Exception as e:
        return JSONResponse(content={"/recommended_games error": str(e)}, status_code=500)

@app.get("/recently_played")
def get_recently_played():
    try:
        recently_played = df_user_owns[df_user_owns["playtime_2weeks"] > 0][["app_id"]]
        data = get_games_additional_info(recently_played)
        return JSONResponse(content=data)  
    except Exception as e:
        return JSONResponse(content={"/recently_played error": str(e)}, status_code=500)

@app.get("/top_tag_games")
def get_top_tag_games():
    try:
        data = get_user_top_tags_games()
        return JSONResponse(content=data)  
    except Exception as e:
        return JSONResponse(content={"/top_tag_games error": str(e)}, status_code=500)

@app.get("/steam_tag_counts")
def get_steam_tag_counts():
    try:
        data = get_steam_tag_counts()
        return JSONResponse(content=data)  
    except Exception as e:
        return JSONResponse(content={"/steam_tag_counts error": str(e)}, status_code=500)


def initialize_global_game_data():
    global df_review_data, game_details_by_app_id, app_id_to_tags, all_unique_tags

    logging.info("⚙️ Initializing global game data...")

    # Load reviews
    review_data = query_db("SELECT app_id, bayesian_score FROM steam_game_reviews;")
    df_review_data = pd.DataFrame(review_data)
    df_review_data["app_id"] = df_review_data["app_id"].astype(str)
    df_review_data["bayesian_score"] = df_review_data["bayesian_score"].astype(float)

    # Load details
    rows = query_db("""
        SELECT app_id, name, is_free, price_usd, header_image,
               screenshot1, screenshot2, screenshot3, screenshot4
        FROM steam_game_details
    """)
    game_details = {}
    for row in rows:
        app_id = str(row["app_id"])
        game_details[app_id] = {
            "name": row["name"],
            "is_free": bool(row["is_free"]),
            "price_usd": row["price_usd"] if row["price_usd"] is not None else "",
            "header_image": row["header_image"] or "",
            "screenshots": [
                row["screenshot1"] or "",
                row["screenshot2"] or "",
                row["screenshot3"] or "",
                row["screenshot4"] or ""
            ]
        }
    game_details_by_app_id = game_details

    # Load tags
    tags_raw = query_db("SELECT app_id, tags FROM steam_game_details;")
    tag_dict = {}
    for row in tags_raw:
        try:
            app_id = str(row["app_id"])
            tags_json = json.loads(row["tags"])
            tag_set = set(tags_json.get("tags", []))
            tag_dict[app_id] = tag_set
            all_unique_tags.update(tag_set)
        except (TypeError, json.JSONDecodeError):
            tag_dict[app_id] = set()
    app_id_to_tags = tag_dict

    get_API_key() 

    logging.info("✅ Global game data ready.")


# --- Steam ID Request ---
class SteamIdRequest(BaseModel):
    steamId: str

@app.post("/submit-steam-id")
async def submit_steam_id(request: SteamIdRequest):
    steam_id = request.steamId
    try:
        result = await process_steam_id(steam_id)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
async def process_steam_id(steam_id: str):
    logging.info(f"Processing Steam ID: {steam_id}")
    try: 
        user_data = is_valid_id(steam_id) 
        prepare_user_info(user_data)  
        calculate_recommended_games()  
        logging.info(f"Steam ID {steam_id} processing complete!")
        return {"status": "success", "message": f"Processed {steam_id}"}

    except ValueError as e:
        logging.info(f"Error processing Steam ID {steam_id}: {e}")
        return {"status": "error", "message": str(e)}



# --- Global Variables for Recommendations ---
user_game_scores = dict()
candidate_app_ids = set()
app_id_to_tags = dict()
game_details_by_app_id = dict()
df_recommended_games = pd.DataFrame()
df_user_owns = pd.DataFrame()
df_scores = pd.DataFrame()
API_KEY = ""
steam_id = None 
all_unique_tags = set()

# --- Helper Functions ---
def get_API_key():
    try:
        with open('./environment.txt', "r") as file:
            global API_KEY
            API_KEY = file.read().strip()
    except FileNotFoundError:
        logging.error(f"Environment file not found at {'environment.txt'}.")


def is_valid_id(steam_id):
    API_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"  

    params = {
        "key": API_KEY,
        "steamid": steam_id,
        "format": "json",
    }
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  
        user_data = response.json()
        return user_data
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 400:
            raise ValueError(f"Invalid Steam ID: {steam_id}")
        else:
            raise ValueError(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        raise ValueError(f"Request error occurred: {req_err}")


def prepare_user_info(user_data):
    owned_games = user_data.get("response", {}).get("games", [])

    user_library_data = [
        {
            "app_id": str(game["appid"]),
            "playtime_forever": game.get("playtime_forever", 0),
            "playtime_2weeks": game.get("playtime_2weeks", 0),
        }
        for game in owned_games
    ]

    global df_user_owns
    df_user_owns = pd.DataFrame(user_library_data)

    df_user_owns["app_id"] = df_user_owns["app_id"].astype(str)

    # Normalize playtime_forever using 75th quantile
    playtime_75th = df_user_owns["playtime_forever"].quantile(0.75)
    df_user_owns["normalized_playtime"] = df_user_owns["playtime_forever"] / playtime_75th

    # add a weighted playtime that incorporates playtime_2weeks with higher weight
    df_user_owns["weighted_playtime"] = (
        df_user_owns["normalized_playtime"] + 15 * (df_user_owns["playtime_2weeks"] / playtime_75th)
    )

    # collect all tags from the user's owned games and weight by weighted_playtime
    user_tag_counts = Counter()
    for _, row in df_user_owns.iterrows():
        app_id = row["app_id"]
        weighted_playtime = row["weighted_playtime"]
        if app_id in app_id_to_tags:
            for tag in app_id_to_tags[app_id]:
                user_tag_counts[tag] += weighted_playtime

    global user_game_scores
    user_game_scores = dict(user_tag_counts)

    all_app_ids = set(app_id_to_tags.keys())
    user_owned_app_ids = set(df_user_owns["app_id"].unique())
    global candidate_app_ids
    candidate_app_ids = all_app_ids - user_owned_app_ids


def calculate_recommended_games():
    results = []
    for app_id in candidate_app_ids:
        game_tags = app_id_to_tags[app_id]
        overlap_score = sum(user_game_scores.get(tag, 0) for tag in game_tags)
        results.append((app_id, overlap_score))

    global df_scores
    df_scores = pd.DataFrame(results, columns=["app_id", "overlap_score"])
    df_scores = df_scores.merge(
        df_review_data[["app_id", "bayesian_score"]],
        on="app_id",
        how="inner")
    df_scores["final_score"] = df_scores["overlap_score"] * df_scores["bayesian_score"]
    df_scores.sort_values("final_score", ascending=False, inplace=True)

    df_scores.reset_index(drop=True, inplace=True)

    global df_recommended_games
    df_recommended_games = df_scores.head(10)


def get_games_additional_info(game_list: pd.DataFrame):
    updated_games = []

    for _, game in game_list.iterrows():
        app_id = str(game['app_id'])
        details = game_details_by_app_id.get(app_id, {})
        tags = list(app_id_to_tags.get(app_id, []))

        updated_games.append({
            "app_id": app_id,
            "name": details.get("name", ""),
            "is_free": details.get("is_free", False),
            "price_usd": details.get("price_usd", ""),
            "tags": tags,
            "header_image": details.get("header_image", ""),
            "screenshots": details.get("screenshots", ["", "", "", ""])
        })

    return updated_games



def get_user_top_tags_games() :
    # get user's top five tags and remove universal tags
    universal_tags = ["Multiplayer", "Singleplayer", "Co-op", "Online Co-Op", "Action", "First-Person", "Third Person", "Third-Person Shooter"]
    filtered_user_tags = {
        tag: count for tag, count in user_game_scores.items()
        if tag not in universal_tags
    }
    top_user_tags = [tag for tag, _ in Counter(filtered_user_tags).most_common(5)]

    # Get app_ids of recommended games to exclude
    excluded_app_ids = set(df_recommended_games["app_id"].unique())

    user_top_tags_games = {}

    for tag in top_user_tags:
        # Get app_ids with this tag
        matching_app_ids = [
            app_id for app_id in df_scores["app_id"]
            if tag in app_id_to_tags.get(app_id, set())
        ]

        # Exclude already recommended games
        filtered_app_ids = [app_id for app_id in matching_app_ids if app_id not in excluded_app_ids]

        # Get score data for remaining games
        tag_scores = df_scores[df_scores["app_id"].isin(filtered_app_ids)]

        # Take top 20 by score
        tag_game_list = tag_scores.nlargest(20, "final_score")

        # Shuffle the top 20 rows to randomize the order
        tag_game_list = tag_game_list.sample(frac=1).reset_index(drop=True)

        # Add additional game info
        tag_game_list = get_games_additional_info(tag_game_list)

        user_top_tags_games[tag] = tag_game_list

    return user_top_tags_games



def get_steam_tag_counts() -> dict[str, int]:
    """
    Returns a dict mapping each tag name to its game_count,
    e.g. {"Action": 1234, "RPG": 567, …}
    """
    try:
        rows = query_db(
            "SELECT tag, game_count FROM steam_tag_summary",
            dictionary=True
        )
        return {row["tag"]: row["game_count"] for row in rows}
    except Exception as err:
        logging.error(f"Failed to fetch tag stats: {err}")
        return {}
