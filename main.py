from fastapi import FastAPI
import mysql.connector
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import requests
import pandas as pd
from collections import Counter


### database setup
app = FastAPI()

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

# temporary references

# @app.get("/get_tasks")
# def get_tasks():
#     cursor = conn.cursor(dictionary=True)
#     cursor.execute("SELECT * FROM todo")
#     records = cursor.fetchall()
#     return records

# @app.post("/add_task")
# def add_task(task: str = Form(...)):
#     cursor = conn.cursor()
#     cursor.execute("INSERT INTO todo (task) VALUES (%s)", (task,))
#     conn.commit()
#     return "Added Successfully"

@app.post("/delete_task")
def delete_task(id: str = Form(...)):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM todo WHERE id=%s", (id,))
    conn.commit()
    return "Deleted Successfully"




API_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
API_KEY = "8C3B145C040FC91AAF2CF63CB482EA4F"  
STEAM_ID = "76561198293567287"  # variable for user id

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

user_tag_profile = dict(user_tag_counts) # dict {tag: weighted_score}

all_app_ids = set(app_id_to_tags.keys())
candidate_app_ids = all_app_ids - user_owned_app_ids

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
recommended_games = df_scores.head(10).merge(df_metadata, on="app_id", how="left")
print(recommended_games[["app_id", "name", "overlap_score"]])
