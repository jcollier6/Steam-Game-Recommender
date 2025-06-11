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
import os

# --- Validate required environment variables ---
required_vars = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE", "API_KEY", "ML_RECOMMEND_URL"]
for var in required_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Missing required environment variable: {var}")

API_KEY = os.getenv("API_KEY")
ML_RECOMMEND_URL = os.getenv("ML_RECOMMEND_URL")  # e.g. "http://ml:8001"

# --- FastAPI setup with lifespan for global init ---
@asynccontextmanager
async def lifespan(app: FastAPI):
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


def get_db_connection():
    """Return a new MySQL connection from the pool."""
    return mysql.connector.connect(
        host=os.environ["MYSQL_HOST"],
        user=os.environ["MYSQL_USER"],
        passwd=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
        pool_name="mypool",
        pool_size=5
    )

# ─── Helper: Steam ID → Owned Games validation ─────────────────────────────────
def is_valid_id(steam_id: str) -> (dict, str):
    steam_id = steam_id.strip()
    # If it looks like a SteamID64 (17 digits), validate directly
    if steam_id.isdigit() and len(steam_id) == 17:
        return validate_steam_id(steam_id)
    # Otherwise treat as vanity
    resolved_id = resolve_vanity(steam_id)
    return validate_steam_id(resolved_id)

def validate_steam_id(steam_id: str) -> (dict, str):
    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": API_KEY,
        "steamid": steam_id,
        "format": "json",
    }
    resp = requests.get(url, params=params)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"Steam API error: {e}")
    data = resp.json()
    return data, steam_id

def resolve_vanity(vanity: str) -> str:
    url = "https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/"
    params = {
        "key": API_KEY,
        "vanityurl": vanity,
    }
    resp = requests.get(url, params=params)
    try:
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to resolve vanity URL: {e}")
    data = resp.json()
    if data.get("response", {}).get("success") == 1:
        return data["response"]["steamid"]
    raise ValueError(f"Vanity URL not found: {vanity}")

# ─── Pydantic models ────────────────────────────────────────────────────────────
class SteamIdRequest(BaseModel):
    steam_id: str

class RecommendationRequest(BaseModel):
    steam_id: str
    top_k: int = 10
    owned_games: list[dict]   # list of {"app_id": str, "playtime_forever": float, "playtime_2weeks": float}
    wishlist: list[dict] | None = None  # optional list of {"app_id": str, "priority": float, "date_added": float}

# ─── Global variables for enrichment ────────────────────────────────────────────
df_review_data = pd.DataFrame()
game_details_by_app_id: dict[str, dict] = {}
app_id_to_tags: dict[str, set[str]] = {}
all_unique_tags: set[str] = set()

# ─── Endpoint: submit-steam-id ──────────────────────────────────────────────────
@app.post("/submit-steam-id")
async def submit_steam_id(request: SteamIdRequest):
    steam_id = request.steam_id
    logging.info(f"Steam ID submitted: {steam_id}")
    try:
        user_data, verified_steam_id = is_valid_id(steam_id)
        return {"steam_id": verified_steam_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ─── Endpoint: get_recommended_games ────────────────────────────────────────────
@app.get("/recommended_games")
def get_recommended_games(steam_id: str, top_k: int = 10):
    try:
        # 1) Validate ID & fetch owned games
        user_data, verified_steam_id = is_valid_id(steam_id)
        owned_games = user_data.get("response", {}).get("games", [])
        owned_list = [
            {
                "app_id": str(game["appid"]),
                "playtime_forever": game.get("playtime_forever", 0.0),
                "playtime_2weeks": game.get("playtime_2weeks", 0.0)
            }
            for game in owned_games
        ]

        # 2) Call ML microservice
        payload = {
            "steam_id": verified_steam_id,
            "top_k": top_k,
            "owned_games": owned_list,
            "wishlist": []
        }
        resp = requests.post(f"{ML_RECOMMEND_URL}/recommend", json=payload)
        resp.raise_for_status()
        rec_ids = resp.json().get("recommendations", [])

        # 3) Enrich recommendation IDs with metadata
        df_rec = pd.DataFrame({"app_id": [str(a) for a in rec_ids]})
        data = get_games_additional_info(df_rec)
        return JSONResponse(content=data)

    except ValueError as e:
        logging.error(f"Validation error in get_recommended_games: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        logging.error(f"Failed to get recommendations: {e}")
        return JSONResponse(content={"error": "Could not fetch recommendations"}, status_code=500)

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

@app.get("/steam_release_years")
def get_steam_release_years():
    try:
        data = get_steam_release_years()
        return JSONResponse(content=data)  
    except Exception as e:
        return JSONResponse(content={"/steam_tag_counts error": str(e)}, status_code=500)

# ─── GLOBAL VARIABLES FOR RECOMMENDATIONS ───────────────────────────────────────
user_game_scores: dict = {}
candidate_app_ids: set = set()
app_id_to_tags: dict = {}
game_details_by_app_id: dict = {}
df_recommended_games = pd.DataFrame()
df_user_owns = pd.DataFrame()
df_scores = pd.DataFrame()
API_KEY = ""
steam_id = None
all_unique_tags: set = set()


def get_API_key():
    global API_KEY
    API_KEY = os.getenv("API_KEY", "")
    if not API_KEY:
        logging.error("API_KEY environment variable not set.")

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

def query_db(query, params=None, dictionary=True):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=dictionary)
    cursor.execute(query, params or ())
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result

# ─── Helper Functions ─────────────────────────────────────
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
    excluded_app_ids = set(df_recommended_games["app_id"].unique())
    user_top_tags_games = {}
    for tag in top_user_tags:
        matching_app_ids = [
            app_id for app_id in df_scores["app_id"]
            if tag in app_id_to_tags.get(app_id, set())
        ]
        filtered_app_ids = [app_id for app_id in matching_app_ids if app_id not in excluded_app_ids]
        tag_scores = df_scores[df_scores["app_id"].isin(filtered_app_ids)]
        tag_game_list = tag_scores.nlargest(20, "final_score")
        tag_game_list = tag_game_list.sample(frac=1).reset_index(drop=True)
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


def get_steam_release_years() -> dict[int, int]:
    """
    Returns a dict mapping each release year to its game_count,
    e.g. {"2024": 1234, "1998": 567, …}
    """
    try:
        rows = query_db(
            "SELECT release_year, game_count FROM steam_year_summary ORDER BY release_year DESC",
            dictionary=True
        )
        return {row["release_year"]: row["game_count"] for row in rows}
    except Exception as err:
        logging.error(f"Failed to fetch release year stats: {err}")
        return {}