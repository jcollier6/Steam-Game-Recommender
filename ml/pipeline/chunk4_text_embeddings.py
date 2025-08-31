import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from .utils import save_numpy


def embed_texts(
    games_df: pd.DataFrame,
    device: str = "cpu",
    short_weight: float = 0.4,
    long_weight: float = 0.6,
) -> np.ndarray:
    """Compute weighted text embeddings using Qwen3."""

    model_id = "Qwen/Qwen3-Embedding-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for _, row in games_df.iterrows():
            text_short = row.get("short_description", "") or ""
            text_long = row.get("long_description", "") or ""

            toks_short = tokenizer(
                text_short,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            emb_short = model(**toks_short).last_hidden_state.mean(dim=1)

            toks_long = tokenizer(
                text_long,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)
            emb_long = model(**toks_long).last_hidden_state.mean(dim=1)

            combined = short_weight * emb_short + long_weight * emb_long
            embeddings.append(combined.squeeze().cpu().numpy())

    return np.stack(embeddings, axis=0)


def run_text_embeddings(
    games_df: pd.DataFrame,
    short_weight: float = 0.4,
    long_weight: float = 0.6,
) -> None:
    """Generate and store Qwen3 text embeddings."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    E_qwen3 = embed_texts(games_df, device, short_weight, long_weight)

    os.makedirs("ml/data", exist_ok=True)
    save_numpy(E_qwen3.astype("float32"), "ml/data/E_qwen3.npy")
    print("✅ Chunk 4 Qwen3 embeddings saved")

