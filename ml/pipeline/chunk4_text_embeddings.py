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
    short_max_length: int = 96,
    long_max_length: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute short and long text embeddings using Qwen3 with batching and logs."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    model_id = "Qwen/Qwen3-Embedding-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    if device.startswith("cuda"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.set_float32_matmul_precision("high")

    short_texts: List[str] = (games_df["short_description"].fillna("").tolist())
    long_texts: List[str] = (games_df["long_description"].fillna("").tolist())

    embeddings_short: List[np.ndarray] = []
    embeddings_long: List[np.ndarray] = []
    total_tokens = 0
    batch_size = 64
    log_every = 80
    start_total = time.time()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for i in range(0, len(games_df), batch_size):
            batch_short = short_texts[i : i + batch_size]
            batch_long = long_texts[i : i + batch_size]

            toks_short = tokenizer(
                batch_short,
                return_tensors="pt",
                truncation=True,
                padding=True,
                pad_to_multiple_of=8,
                max_length=short_max_length,
            )
            toks_long = tokenizer(
                batch_long,
                return_tensors="pt",
                truncation=True,
                padding=True,
                pad_to_multiple_of=8,
                max_length=long_max_length,
            )
            toks_short = {k: v.contiguous().to(device, non_blocking=True) for k, v in toks_short.items()}
            toks_long = {k: v.contiguous().to(device, non_blocking=True) for k, v in toks_long.items()}

            batch_start = time.time()
            out_short = model(**toks_short)
            out_long = model(**toks_long)
            mask_s = toks_short["attention_mask"].unsqueeze(-1)
            emb_short = (out_short.last_hidden_state * mask_s).sum(dim=1) / mask_s.sum(dim=1).clamp(min=1)
            mask_l = toks_long["attention_mask"].unsqueeze(-1)
            emb_long = (out_long.last_hidden_state * mask_l).sum(dim=1) / mask_l.sum(dim=1).clamp(min=1)
            batch_time = time.time() - batch_start

            num_tokens = int(toks_short["attention_mask"].sum().item() + toks_long["attention_mask"].sum().item())
            total_tokens += num_tokens
            tok_per_sec = num_tokens / batch_time if batch_time > 0 else float("inf")

            embeddings_short.append(emb_short.float().cpu().numpy())
            embeddings_long.append(emb_long.float().cpu().numpy())

            batch_idx = i // batch_size + 1
            if batch_idx % log_every == 0 or batch_idx == 1:
                processed = min(i + batch_size, len(games_df))
                remaining = len(games_df) - processed
                print(
                    f"Batch {batch_idx}: processed {processed}/{len(games_df)} rows, "
                    f"{remaining} remaining at {tok_per_sec:.0f} tok/s", flush=True
                )

    total_time = time.time() - start_total
    if total_time > 0:
        print(f"Total throughput: {total_tokens / total_time:.0f} tok/s over {total_time:.1f}s")

    return np.concatenate(embeddings_short, axis=0), np.concatenate(embeddings_long, axis=0)


def run_text_embeddings(
    games_df: pd.DataFrame,
    short_max_length: int = 96,
    long_max_length: int = 1024,
) -> None:
    """Generate and store separate Qwen3 text embeddings."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    E_short, E_long = embed_texts(
        games_df,
        device,
        short_max_length=short_max_length,
        long_max_length=long_max_length,
    )

    os.makedirs("ml/data", exist_ok=True)
    save_numpy(E_short.astype("float32"), "ml/data/E_qwen3_short.npy")
    save_numpy(E_long.astype("float32"), "ml/data/E_qwen3_long.npy")
    print("✅ Chunk 4 Qwen3 embeddings saved")

