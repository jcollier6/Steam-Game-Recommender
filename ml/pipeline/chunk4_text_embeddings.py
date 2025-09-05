import os
import time
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
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
    if device.startswith("cuda"):
        # Reduce allocator fragmentation when repeatedly allocating large tensors
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model_dir = os.getenv("QWEN3_MODEL_DIR", "ml/models/Qwen3-Embedding-4B")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float16,

        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    if device.startswith("cuda"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.set_float32_matmul_precision("high")

    short_texts: List[str] = games_df["short_description"].fillna("").tolist()
    long_texts: List[str] = games_df["long_description"].fillna("").tolist()

    class TextDataset(Dataset):
        def __init__(self, shorts: List[str], longs: List[str]) -> None:
            self.shorts = shorts
            self.longs = longs

        def __len__(self) -> int:  # pragma: no cover - trivial
            return len(self.shorts)

        def __getitem__(self, idx: int) -> tuple[str, str]:  # pragma: no cover - trivial
            return self.shorts[idx], self.longs[idx]

    dataset = TextDataset(short_texts, long_texts)

    def collate(batch: list[tuple[str, str]]) -> tuple[dict, dict]:
        batch_short, batch_long = zip(*batch)
        toks_short = tokenizer(
            list(batch_short),
            return_tensors="pt",
            truncation=True,
            padding=True,
            pad_to_multiple_of=8,
            max_length=short_max_length,
        )
        toks_long = tokenizer(
            list(batch_long),
            return_tensors="pt",
            truncation=True,
            padding=True,
            pad_to_multiple_of=8,
            max_length=long_max_length,
        )
        return toks_short, toks_long

    batch_size = 64
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=device.startswith("cuda"),
    )

    embeddings_short: List[np.ndarray] = []
    embeddings_long: List[np.ndarray] = []
    total_tokens = 0
    log_every = 80
    start_total = time.time()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for batch_idx, (toks_short, toks_long) in enumerate(loader, 1):
            batch_start = time.time()

            toks_short = {k: v.contiguous().to(device, non_blocking=True) for k, v in toks_short.items()}
            out_short = model(**toks_short)
            mask_s = toks_short["attention_mask"].unsqueeze(-1)
            emb_short = (out_short.last_hidden_state * mask_s).sum(dim=1) / mask_s.sum(dim=1).clamp(min=1)
            embeddings_short.append(emb_short.float().cpu().numpy())
            num_tokens_short = int(toks_short["attention_mask"].sum().item())
            del out_short, emb_short, mask_s, toks_short
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            toks_long = {k: v.contiguous().to(device, non_blocking=True) for k, v in toks_long.items()}
            out_long = model(**toks_long)
            mask_l = toks_long["attention_mask"].unsqueeze(-1)
            emb_long = (out_long.last_hidden_state * mask_l).sum(dim=1) / mask_l.sum(dim=1).clamp(min=1)
            embeddings_long.append(emb_long.float().cpu().numpy())
            num_tokens_long = int(toks_long["attention_mask"].sum().item())
            del out_long, emb_long, mask_l, toks_long
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            batch_time = time.time() - batch_start
            num_tokens = num_tokens_short + num_tokens_long
            total_tokens += num_tokens
            tok_per_sec = num_tokens / batch_time if batch_time > 0 else float("inf")

            if batch_idx % log_every == 0 or batch_idx == 1:
                processed = min(batch_idx * batch_size, len(games_df))
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

