import os
import logging
from typing import List, Optional

import mysql.connector
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline.chunk11_inference import recommend as chunk11_recommend


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)

DATA_DIR = os.getenv("ML_DATA_DIR", "ml/data")
INTERACTIONS_DF = pd.read_pickle(os.path.join(DATA_DIR, "interactions_df.pkl"))

app = FastAPI()


class OwnedGameRecord(BaseModel):
    app_id: str
    playtime_forever: float
    playtime_2weeks: float


class WishlistRecord(BaseModel):
    app_id: str
    priority: float
    date_added: float


class RecommendationRequest(BaseModel):
    steam_id: str
    top_k: int = 10
    owned_games: List[OwnedGameRecord]
    wishlist: Optional[List[WishlistRecord]] = None


def _upsert_interactions(user_id: int, library_df: pd.DataFrame) -> None:
    """Append new user interactions to the in-memory dataframe and MySQL."""
    global INTERACTIONS_DF
    INTERACTIONS_DF = pd.concat([INTERACTIONS_DF, library_df], ignore_index=True)

    required = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
    if not all(os.getenv(k) for k in required):
        return
    try:
        conn = mysql.connector.connect(
            host=os.environ["MYSQL_HOST"],
            user=os.environ["MYSQL_USER"],
            passwd=os.environ["MYSQL_PASSWORD"],
            database=os.environ["MYSQL_DATABASE"],
        )
        cursor = conn.cursor()
        insert_sql = (
            "INSERT INTO user_interactions "
            "(user_id, app_id, playtime_forever, playtime_2weeks, wishlisted, wishlist_priority, date_added_to_wishlist) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE "
            "playtime_forever=VALUES(playtime_forever), "
            "playtime_2weeks=VALUES(playtime_2weeks), "
            "wishlisted=VALUES(wishlisted), "
            "wishlist_priority=VALUES(wishlist_priority), "
            "date_added_to_wishlist=VALUES(date_added_to_wishlist)"
        )
        rows = library_df[
            [
                "user_id",
                "app_id",
                "playtime_forever",
                "playtime_2weeks",
                "wishlisted",
                "wishlist_priority",
                "date_added_to_wishlist",
            ]
        ].values.tolist()
        cursor.executemany(insert_sql, rows)
        conn.commit()
    except mysql.connector.Error as e:
        logging.error(f"Failed to upsert interactions: {e}")
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


@app.post("/recommend")
def recommend_api(req: RecommendationRequest):
    try:
        user_id = int(req.steam_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid steam_id")

    records = []
    for og in req.owned_games:
        records.append(
            {
                "user_id": user_id,
                "app_id": int(og.app_id),
                "playtime_forever": og.playtime_forever,
                "playtime_2weeks": og.playtime_2weeks,
                "wishlisted": False,
                "wishlist_priority": 0.0,
                "date_added_to_wishlist": 0.0,
            }
        )
    if req.wishlist:
        for wl in req.wishlist:
            records.append(
                {
                    "user_id": user_id,
                    "app_id": int(wl.app_id),
                    "playtime_forever": 0.0,
                    "playtime_2weeks": 0.0,
                    "wishlisted": True,
                    "wishlist_priority": wl.priority,
                    "date_added_to_wishlist": wl.date_added,
                }
            )

    user_library_df = pd.DataFrame(records)

    if INTERACTIONS_DF[INTERACTIONS_DF["user_id"] == user_id].empty and not user_library_df.empty:
        _upsert_interactions(user_id, user_library_df)

    try:
        recs = chunk11_recommend(
            user_id,
            INTERACTIONS_DF,
            K=req.top_k,
            user_library_df=user_library_df if not user_library_df.empty else None,
        )
        return {"recommendations": recs}
    except Exception as e:
        logging.error(f"Recommendation Error: {e}")
        raise HTTPException(status_code=500, detail="Could not generate recommendations")
