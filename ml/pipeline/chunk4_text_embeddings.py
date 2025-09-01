import os
import time
from typing import List
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from .utils import save_numpy


def embed_texts(
    games_df: pd.DataFrame,
    device: str = "cpu",
    short_weight: float = 0.4,
    long_weight: float = 0.6,
    batch_size: int = 256,
    log_every: int = 20,
    short_max_length: int = 96,
    long_max_length: int = 1024,
) -> np.ndarray:
    """Compute weighted text embeddings using Qwen3 with batching and logs."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    model_id = "Qwen/Qwen3-Embedding-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            attn_implementation="flash_attention_2",
        ).to(device)
    except Exception as exc:
        raise RuntimeError(
            "FlashAttention-2 is required for efficient embedding; install the flash-attn package"
        ) from exc
    model.eval()

    short_texts: List[str] = (games_df["short_description"].fillna("").tolist())
    long_texts: List[str] = (games_df["long_description"].fillna("").tolist())

    embeddings: List[np.ndarray] = []
    total_tokens = 0
    start_total = time.time()

    with torch.no_grad():
        for i in range(0, len(games_df), batch_size):
            batch_short = short_texts[i : i + batch_size]
            batch_long = long_texts[i : i + batch_size]

            toks_short = tokenizer(
                batch_short,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=short_max_length,
            ).to(device)
            toks_long = tokenizer(
                batch_long,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=long_max_length,
            ).to(device)

            batch_start = time.time()
            emb_short = model(**toks_short).last_hidden_state.mean(dim=1)
            emb_long = model(**toks_long).last_hidden_state.mean(dim=1)
            combined = short_weight * emb_short + long_weight * emb_long
            batch_time = time.time() - batch_start

            num_tokens = int(toks_short.attention_mask.sum().item() + toks_long.attention_mask.sum().item())
            total_tokens += num_tokens
            if batch_time > 0:
                tok_per_sec = num_tokens / batch_time
            else:
                tok_per_sec = float("inf")

            embeddings.append(combined.cpu().numpy())

            batch_idx = i // batch_size + 1
            if batch_idx % log_every == 0 or batch_idx == 1:
                processed = min(i + batch_size, len(games_df))
                remaining = len(games_df) - processed
                print(
                    f"Batch {batch_idx}: processed {processed}/{len(games_df)} rows, "
                    f"{remaining} remaining at {tok_per_sec:.0f} tok/s"
                )

    total_time = time.time() - start_total
    if total_time > 0:
        print(f"Total throughput: {total_tokens / total_time:.0f} tok/s over {total_time:.1f}s")

    return np.concatenate(embeddings, axis=0)


def run_text_embeddings(
    games_df: pd.DataFrame,
    short_weight: float = 0.4,
    long_weight: float = 0.6,
    batch_size: int = 256,
    log_every: int = 20,
    short_max_length: int = 96,
    long_max_length: int = 1024,
) -> None:
    """Generate and store Qwen3 text embeddings."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    E_qwen3 = embed_texts(
        games_df,
        device,
        short_weight,
        long_weight,
        batch_size=batch_size,
        log_every=log_every,
        short_max_length=short_max_length,
        long_max_length=long_max_length,
    )

    os.makedirs("ml/data", exist_ok=True)
    save_numpy(E_qwen3.astype("float32"), "ml/data/E_qwen3.npy")
    print("✅ Chunk 4 Qwen3 embeddings saved")

