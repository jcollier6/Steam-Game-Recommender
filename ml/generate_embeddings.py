import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

def embed_texts_in_batches(texts, tokenizer, model, device, batch_size):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embs = outputs.last_hidden_state[:, 0].cpu()  # CLS token
            all_embeddings.append(batch_embs)
        print(f"Embedded batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
    return torch.cat(all_embeddings, dim=0)

def main():
    print("Loading game descriptions...")
    df = pd.read_parquet("/app/ml/data/raw_game_features.parquet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

    # Combine short + detailed descriptions
    texts = (df["short_description"].fillna("") + " " + df["detailed_description"].fillna("")).tolist()

    print(f"Embedding {len(texts)} game descriptions in batches...")
    embeddings = embed_texts_in_batches(texts, tokenizer, model, device, batch_size=512)

    # Save as NumPy array
    np.save("/app/ml/data/game_desc_embeddings.npy", embeddings.numpy())
    print("Saved embeddings to game_desc_embeddings.npy")

if __name__ == "__main__":
    main()
