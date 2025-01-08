from fastapi import FastAPI
import mysql.connector
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


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






cleaned_games = pd.read_csv('cleaned_games.csv', escapechar='\\')  
tags = pd.read_csv('tags.csv')                    


### Create a multi-hot encoded matrix for the tags
all_tags = tags['tag'].unique()
all_tags = sorted(all_tags)  

tag2index = {tag_name: index for index, tag_name in enumerate(all_tags)}

app_tags_dict = {}
for row in tags.itertuples(index=False):
    app_id, tag = row.app_id, row.tag
    if app_id not in app_tags_dict:
        app_tags_dict[app_id] = set()
    app_tags_dict[app_id].add(tag)

num_games = cleaned_games.shape[0]
num_tags = len(all_tags)

app_id_list = cleaned_games['app_id'].tolist()
app2index = {app: index for index, app in enumerate(app_id_list)}

tag_matrix = np.zeros((num_games, num_tags), dtype=np.float32)

# Fill the matrix with 1's where the app has that tag
for app_id, tagset in app_tags_dict.items():
    if app_id in app2index:  
        row_index = app2index[app_id]
        for t in tagset:
            col_index = tag2index[t]
            tag_matrix[row_index, col_index] = 1.0


### Combine numeric columns with the multi-hot tag matrix
numeric_cols = ["positive", "negative", "total", "recommendations", "is_free"]
numeric_data = cleaned_games[numeric_cols].values.astype(np.float32)

# log-transform numeric columns
numeric_data = np.log1p(numeric_data)

# Concatenate numeric_data with tag_matrix horizontally
# shape = [num_games, len(numeric_cols) + num_tags]
features = np.hstack([numeric_data, tag_matrix])


### standardize the entire feature set:
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# features_scaled is now [num_games, D] where D = number of columns in numeric_cols + num_tags

features_df = pd.DataFrame(features_scaled)
features_df['app_id'] = app_id_list
print(features_df.head())

# features_scaled is what we'll feed into the VAE as input data for each game
print("Final feature array shape for VAE:", features_scaled.shape)
