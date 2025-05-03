import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

def embed(texts, tok, model, device):
    batch = tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**batch)
    return out.last_hidden_state[:,0].cpu().numpy()

def main():
    df = pd.read_parquet("/app/ml/data/raw_game_features.parquet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok   = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

    texts = (df.short_description.fillna("") + " " + df.detailed_description.fillna("")).tolist()
    embs  = embed(texts, tok, model, device)

    np.save("/app/ml/data/game_desc_embeddings.npy", embs)
    print("✅ embeddings saved")
    
if __name__=="__main__":
    main()
