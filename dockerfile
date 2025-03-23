FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV HF_HOME=/app/hf_cache


# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio

# Install other requirements
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Install flash-attention from source (with CUDA available)
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir fastapi uvicorn

# Create cache directory
RUN mkdir -p /app/hf_cache

# Copy your server script
COPY server.py /app/server.py
COPY test_model.py /app/test_model.py

# Set the working directory
WORKDIR /app

EXPOSE 8100

# Run the server when the container starts
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1", "--log-level", "debug", "--access-log"]
