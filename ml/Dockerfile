FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python and basic tools
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache this layer)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Download HuggingFace model early to cache it
RUN mkdir -p /root/.cache/huggingface && \
    python3 -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"

# Now copy your actual code (only if needed)
COPY . .

# Entrypoint is defined by docker-compose
