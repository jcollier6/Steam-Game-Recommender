FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG INCLUDE_QWEN=true

WORKDIR /app

# Install Python and basic tools
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache this layer)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Download HuggingFace model early to cache it
RUN mkdir -p /root/.cache/huggingface && \
    python3 -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"

# Install fastapi to connect between backend and recommend_games
RUN pip install fastapi uvicorn scikit-learn torch joblib

# Now copy your actual code (only if needed)
COPY . .

# Optionally drop the heavy Qwen3 embedding model to speed up builds when
# starting from later chunks.
RUN if [ "$INCLUDE_QWEN" != "true" ]; then \
      rm -rf models/Qwen3-Embedding-4B; \
    fi

# Entrypoint is defined by docker-compose
