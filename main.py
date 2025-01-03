from fastapi import FastAPI, Form
import mysql.connector
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import requests

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

@app.post("/add_task")
def add_task(task: str = Form(...)):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO todo (task) VALUES (%s)", (task,))
    conn.commit()
    return "Added Successfully"

@app.post("/delete_task")
def delete_task(id: str = Form(...)):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM todo WHERE id=%s", (id,))
    conn.commit()
    return "Deleted Successfully"



import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import mean_absolute_error


class NeuralCF:
    def __init__(self, num_users, num_items, embedding_dim=10, hidden_layers=[64, 32], activation='relu', learning_rate=0.001):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate

    def _build_model(self):
        user_input = Input(shape=(1,))
        item_input = Input(shape=(1,))
        user_embedding = Embedding(self.num_users, self.embedding_dim)(user_input)
        user_embedding = Flatten()(user_embedding)
        item_embedding = Embedding(self.num_items, self.embedding_dim)(item_input)
        item_embedding = Flatten()(item_embedding)
        vector = Concatenate()([user_embedding, item_embedding])
        for units in self.hidden_layers:
            vector = Dense(units, activation=self.activation)(vector)
        output = Dense(1, activation='sigmoid')(vector)
        model = Model(inputs=[user_input, item_input], outputs=output)
        return model
    
    def train(self, X_train, y_train, epochs=2, batch_size=10, validation_split=0.1):
        X_train = [X_train[:, 0], X_train[:, 1]]
        y_train = np.array(y_train)

        # Compute class weights
        unique_ratings = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=unique_ratings, y=y_train)
        class_weight_dict = {rating: weight for rating, weight in zip(unique_ratings, class_weights)}
        print("Class Weights:", class_weight_dict)

        model = self._build_model()
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mean_squared_error')
        # Optional: Add callbacks for better training
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

        # Train the model and capture the history
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, lr_scheduler]
        )
        self.model = model
        self.history = history

    def predict(self, X_test):
        X_test = [X_test[:, 0], X_test[:, 1]]
        return self.model.predict(X_test)
    
# Plotting function
def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

# Hyperparameters that could be tuned
embedding_dim = 10
hidden_layers = [64, 32]
activation = 'relu'
learning_rate = 0.001
# we choose a subset of the netflix rating dataset found here : https://www.kaggle.com/datasets/rishitjavia/netflix-movie-rating-dataset?select=Netflix_Dataset_Rating.csv
df = pd.read_csv("C:/Users/justi/my-app/Netflix_Dataset_Rating.csv").iloc[:1000,:]
X = df[['User_ID','Movie_ID']].to_numpy()
y = df['Rating'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
num_users = df['User_ID'].max() + 1
num_items = df['Movie_ID'].max() + 1
# Train and predict using NeuralCF
ncf = NeuralCF(num_users, num_items, embedding_dim, hidden_layers, activation, learning_rate)
ncf.train(X_train, y_train)
y_pred = ncf.predict(X_test)
# Evaluate using Mean Squared Error
mse = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mse)

import matplotlib.pyplot as plt
plt.hist(y, bins=5)
plt.title("Distribution of Ratings")
plt.show()

plot_loss(ncf.history)


