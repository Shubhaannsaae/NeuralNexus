# ML models container with GPU support
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    libhdf5-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Set work directory
WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install scientific computing libraries
COPY ml_requirements.txt .
RUN pip install --no-cache-dir -r ml_requirements.txt

# Install ESMFold and AlphaFold dependencies
RUN pip install fair-esm biopython biotite

# Copy ML models code
COPY ml_models/ ./ml_models/
COPY app/services/ ./app/services/

# Create model directories
RUN mkdir -p /app/models /app/cache /app/data

# Download pre-trained models (optional)
# RUN python -c "import torch; torch.hub.load('facebookresearch/esm:main', 'esm2_t33_650M_UR50D')"

# Expose port for ML service
EXPOSE 8001

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
  CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Run ML service
CMD ["python", "-m", "uvicorn", "app.services.ml_service:app", "--host", "0.0.0.0", "--port", "8001"]
