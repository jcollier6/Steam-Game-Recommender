import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml.pipeline.chunk11_inference import recommend

DATA_DIR = os.getenv("ML_DATA_DIR", "data")

app = FastAPI()

try:
    interactions_df = pd.read_pickle(os.path.join(DATA_DIR, "interactions_df.pkl"))
except FileNotFoundError:
    interactions_df = pd.DataFrame()

class RecommendationRequest(BaseModel):
    steamid: int
    top_k: int = 20

@app.post("/recommend")
def recommend_games_api(req: RecommendationRequest):
    if interactions_df.empty:
        raise HTTPException(status_code=500, detail="interactions_df not loaded")
    try:
        recs = recommend(req.steamid, interactions_df, K=req.top_k, data_dir=DATA_DIR)
        return {"recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
