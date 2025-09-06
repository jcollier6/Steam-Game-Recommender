# Steam Game Recommender

## UI Mockup

UI Mockups made on Figma. Prototype available [here](https://www.figma.com/proto/i6GKXkI2IO2LfhrwlazxU3/Steam-Game-Recommender?)

## Build

Run `python -m uvicorn main:app --reload` in the project folder to build it before starting the server.

## Development server

Run `ng serve` for a dev server. Navigate to `http://localhost:4200/`. 

## Notes

- Currently, you can enter any user's SteamId and it will provide recommendations based on their profile only if their profile is public. I plan on adding the ability for users to sign in to Steam so it'll work even when their account is private.

- The API Key is in a file named "environment.txt" that I've hidden from github. If you want to run this use your own Steam API key. Place the file in the main folder and only put the API key in the txt file, nothing else.

## ML Pipeline

This repository now includes a new modular pipeline under `ml/pipeline` implementing the steps described in the design document. Each chunk can be executed independently:

1. `chunk1_loader` – loads database tables, user interactions, and fetches the
   top most played games from the Steam Charts API.
2. `chunk2_structured` – computes structured numeric features.
3. `chunk3_tags_pca` – builds tag vectors and PCA embeddings.
4. `chunk4_text_embeddings` – computes Qwen3 embeddings for short and long descriptions
   and stores them separately as `ml/data/E_qwen3_short.npy` and
   `ml/data/E_qwen3_long.npy`. The texts are truncated to 96 and 1,024 tokens,
   tokenization pads to multiples of 8, and embeddings are averaged with
   attention‑mask weighting. The chunk uses PyTorch SDPA in float16, batches with
   a default size of 256, and logs token‑throughput every 20 batches; both
   `batch_size` and logging frequency are configurable.
5. `chunk5_item_meta` – combines all features into 128‑dim item embeddings.
6. `chunk6_faiss` – builds FAISS indices for fast retrieval.

The remaining chunks provide utilities for user profiling, model architecture, training, and inference. See the individual files for details.

### Training in Docker

To execute the entire pipeline inside the ML container and train the model, run:

```bash
# On Linux/macOS (bash)
PYTHONUNBUFFERED=1 docker compose --profile train up --build ml-train

# On Windows Command Prompt
set PYTHONUNBUFFERED=1 && docker compose --profile train up --build ml-train

# On Windows PowerShell
$env:PYTHONUNBUFFERED=1; docker compose --profile train up --build ml-train
```

This command runs `ml/pipeline/train_pipeline.py` which sequentially executes all pipeline chunks and trains the ranking model.

To resume from a specific chunk, set the `START_CHUNK` environment variable. For example, to begin at chunk 5:

```bash
set START_CHUNK=5 && set PYTHONUNBUFFERED=1 && docker compose --profile train up --build ml-train
```

