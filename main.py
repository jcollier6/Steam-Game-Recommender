from fastapi import FastAPI, HTTPException
import mysql.connector
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import pandas as pd
from collections import Counter
from fastapi.responses import JSONResponse
from pydantic import BaseModel


### database setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
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



@app.get("/get_tags")
def get_tags():
    records = query_db("SELECT * FROM steam_game_tags")
    return records

@app.get("/recommended_games")
def get_recommended_games():
    try:
        data = recommended_games_additional_info()
        return JSONResponse(content=data)  
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/recently_played")
def get_recently_played():
    try:
        data = df_user_owns[df_user_owns["playtime_2weeks"] > 0][
            ["name", "playtime_forever", "playtime_2weeks"]
        ].to_dict(orient="records")
        return JSONResponse(content=data)  
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



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
    print(f"Processing Steam ID: {steam_id}")
    try:
        get_API_key()  
        user_data = is_valid_id(steam_id) 
        prepare_user_info(user_data)  
        calculate_recommended_games()  
        print(f"Steam ID {steam_id} processing complete!")
        return {"status": "success", "message": f"Processed {steam_id}"}

    except ValueError as e:
        print(f"Error processing Steam ID {steam_id}: {e}")
        return {"status": "error", "message": str(e)}



# --- Global Variables for Recommendations ---
user_game_scores = dict()
candidate_app_ids = set()
app_id_to_tags = dict()
recommended_games = pd.DataFrame()
df_user_owns = pd.DataFrame()
API_KEY = ""
steam_id = None 

# Initialize global DataFrames using connection pooling
metadata = query_db("""
    SELECT app_id, name, coming_soon, release_date, is_free, price_usd, 
           header_image, screenshot1, screenshot2, screenshot3, screenshot4 
    FROM steam_game_details;
""")
df_game_metadata = pd.DataFrame(metadata)
df_game_metadata["app_id"] = df_game_metadata["app_id"].astype(str)
df_game_metadata["name"] = df_game_metadata["name"].astype(str)

review_data = query_db("SELECT app_id, bayesian_score FROM steam_game_reviews;")
df_review_data = pd.DataFrame(review_data)
df_review_data["app_id"] = df_review_data["app_id"].astype(str)
df_review_data["bayesian_score"] = df_review_data["bayesian_score"].astype(float)

# --- Helper Functions ---
def get_API_key():
    try:
        with open('./environment.txt', "r") as file:
            global API_KEY
            API_KEY = file.read().strip()
    except FileNotFoundError:
        print(f"Environment file not found at {'environment.txt'}.")


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

    steam_tags_csv = "tags.csv"     
    df_steam_tags = pd.read_csv(steam_tags_csv)

    df_user_owns["app_id"] = df_user_owns["app_id"].astype(str)
    df_steam_tags["app_id"] = df_steam_tags["app_id"].astype(str)

    # Filter df_user_owns to remove games not known to my database
    df_user_owns = df_user_owns[df_user_owns["app_id"].isin(df_game_metadata["app_id"])]

    df_app_tags = (
        df_steam_tags
        .groupby('app_id')['tag']
        .apply(set)
        .reset_index()
    )

    global app_id_to_tags
    app_id_to_tags = dict(zip(df_app_tags['app_id'], df_app_tags['tag']))
    user_owned_app_ids = set(df_user_owns["app_id"].unique())

    # Normalize playtime_forever using 75th quantile
    playtime_75th = df_user_owns["playtime_forever"].quantile(0.75)
    df_user_owns["normalized_playtime"] = df_user_owns["playtime_forever"] / playtime_75th

    # add a weighted playtime that incorporates playtime_2weeks with higher weight
    df_user_owns["weighted_playtime"] = (
        df_user_owns["normalized_playtime"] + 15 * (df_user_owns["playtime_2weeks"] / playtime_75th)
    )

    # add additional game information to the owned games
    df_user_owns = df_user_owns.merge(df_game_metadata, on="app_id", how="left")

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
    global candidate_app_ids
    candidate_app_ids = all_app_ids - user_owned_app_ids


def calculate_recommended_games():
    results = []
    for app_id in candidate_app_ids:
        game_tags = app_id_to_tags[app_id]
        overlap_score = sum(user_game_scores.get(tag, 0) for tag in game_tags)
        results.append((app_id, overlap_score))

    df_scores = pd.DataFrame(results, columns=["app_id", "overlap_score"])
    df_scores = df_scores.merge(
        df_review_data[["app_id", "bayesian_score"]],
        on="app_id",
        how="inner")
    df_scores["final_score"] = df_scores["overlap_score"] * df_scores["bayesian_score"]
    df_scores.sort_values("final_score", ascending=False, inplace=True)

    df_scores.reset_index(drop=True, inplace=True)

    global recommended_games
    recommended_games = df_scores.head(10).merge(df_game_metadata, on="app_id", how="left")


def recommended_games_additional_info():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    updated_games = []

    for index, game in recommended_games.iterrows():
        app_id = game['app_id']
        
        cursor.execute("""
            SELECT price_usd, header_image, screenshot1, screenshot2, screenshot3, screenshot4
            FROM steam_game_details 
            WHERE app_id = %s
        """, (app_id,))
        
        additional_info = cursor.fetchone()
        
        if additional_info:
            price_usd = additional_info['price_usd']
            header_image = additional_info['header_image']
            screenshot1 = additional_info['screenshot1']
            screenshot2 = additional_info['screenshot2']
            screenshot3 = additional_info['screenshot3']
            screenshot4 = additional_info['screenshot4']
            screenshots = [
                screenshot1 if screenshot1 is not None else "",
                screenshot2 if screenshot2 is not None else "",
                screenshot3 if screenshot3 is not None else "",
                screenshot4 if screenshot4 is not None else ""
            ]
            price_usd = price_usd if price_usd is not None else ""
        else:
            price_usd = ""
            header_image = ""
            screenshots = ["", "", "", ""]

        game_tags = app_id_to_tags.get(game['app_id'], [])

        updated_games.append({
            "app_id": game["app_id"],
            "name": game["name"],
            "is_free": game["is_free"],
            "price_usd": price_usd,
            "tags": list(game_tags),
            "header_image": game.get("header_image", ""),
            "header_image": header_image,
            "screenshots": screenshots 
        })
    cursor.close()
    conn.close()
    return updated_games
