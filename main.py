from fastapi import FastAPI
from contextlib import asynccontextmanager
import mysql.connector
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import pandas as pd
from collections import Counter
from fastapi.responses import JSONResponse



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup")
    get_API_key()
    prepare_user_info()
    calculate_recommended_games()
    yield
    print("Application shutdown")

app = FastAPI(lifespan=lifespan)


### database setup

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="testpassword1",
    database="mydb"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/get_games")
def get_games():
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM games")
    records = cursor.fetchall()
    return records

@app.get("/get_tags")
def get_tags():
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM tags")
    records = cursor.fetchall()
    return records

@app.get("/get_reviews")
def get_reviews():
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM reviews")
    records = cursor.fetchall()
    return records

@app.get("/get_users")
def get_users():
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT user_id FROM user_data")
    records = cursor.fetchall()
    return records

@app.get("/recommended_games")
def get_recommended_games():
    try:
        data = recommended_games[["app_id", "name", "overlap_score"]].to_dict(orient="records")
        return JSONResponse(content=data)  
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


### recommendation calculation

user_tag_profile = dict()
candidate_app_ids = set()
app_id_to_tags = dict()
recommended_games = pd.DataFrame()
API_KEY = ""


def get_API_key():
    try:
        with open('./environment.txt', "r") as file:
            global API_KEY
            API_KEY = file.read().strip()
    except FileNotFoundError:
        print(f"Environment file not found at {'environment.txt'}.")

def prepare_user_info():
    API_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"  
    STEAM_ID = "76561198208956405"  # variable for user id

    params = {
        "key": API_KEY,
        "steamid": STEAM_ID,
        "format": "json",
    }

    response = requests.get(API_URL, params=params)
    data = response.json()

    owned_games = data.get("response", {}).get("games", [])

    user_library_data = [
        {
            "app_id": str(game["appid"]),
            "playtime_forever": game.get("playtime_forever", 0),
            "playtime_2weeks": game.get("playtime_2weeks", 0),
        }
        for game in owned_games
    ]

    df_user_owns = pd.DataFrame(user_library_data)

    print(df_user_owns[df_user_owns["playtime_2weeks"]>0])

    steam_tags_csv = "tags.csv"     
    df_steam_tags = pd.read_csv(steam_tags_csv)

    df_user_owns["app_id"] = df_user_owns["app_id"].astype(str)
    df_steam_tags["app_id"] = df_steam_tags["app_id"].astype(str)

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

    # collect all tags from the user's owned games and weight by weighted_playtime
    user_tag_counts = Counter()
    for _, row in df_user_owns.iterrows():
        app_id = row["app_id"]
        weighted_playtime = row["weighted_playtime"]
        if app_id in app_id_to_tags:
            for tag in app_id_to_tags[app_id]:
                user_tag_counts[tag] += weighted_playtime

    global user_tag_profile
    user_tag_profile = dict(user_tag_counts) # dict {tag: weighted_score}

    all_app_ids = set(app_id_to_tags.keys())
    global candidate_app_ids
    candidate_app_ids = all_app_ids - user_owned_app_ids


def calculate_recommended_games():
    results = []
    for app_id in candidate_app_ids:
        game_tags = app_id_to_tags[app_id]
        overlap_score = sum(user_tag_profile.get(tag, 0) for tag in game_tags)
        results.append((app_id, overlap_score))

    df_scores = pd.DataFrame(results, columns=["app_id", "overlap_score"])
    df_scores.sort_values("overlap_score", ascending=False, inplace=True)
    df_scores.reset_index(drop=True, inplace=True)


    # display game name, id, and overlap score
    df_metadata = pd.read_csv("cleaned_games.csv", escapechar='\\') 
    df_metadata["app_id"] = df_metadata["app_id"].astype(str)
    df_metadata["name"] = df_metadata["name"].astype(str)
    global recommended_games
    recommended_games = df_scores.head(10).merge(df_metadata, on="app_id", how="left")
    print(recommended_games[["app_id", "name", "overlap_score"]])



