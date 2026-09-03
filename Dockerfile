# Dockerfile for reproducible PI-Mamba experiments
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire codebase
COPY . /workspace

# Set the entrypoint
ENTRYPOINT ["python3"]
