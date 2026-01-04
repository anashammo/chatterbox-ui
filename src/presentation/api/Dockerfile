# ============================================================================
# Stage 1: Builder - Install dependencies with UV and BuildKit caching
# ============================================================================
# Using CUDA 12.8 with cuDNN 9 for RTX 5090 (Blackwell architecture, sm_120)
# CUDA 12.8+ is required for full sm_120 compute capability support
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building Python packages with BuildKit cache
# Using Python 3.11 as required by Chatterbox TTS (tested on 3.11, requires >= 3.10)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install UV package manager (pinned version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:0.9.21 /uv /uvx /bin/

# Set UV environment variables for optimal Docker builds
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300

# Set working directory
WORKDIR /build

# Copy requirements file from backend folder
COPY src/presentation/api/requirements.txt .

# Install numpy FIRST (required before pkuseg/chatterbox deps)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python python3.11 --system "numpy>=1.24.0,<1.26.0"

# Install PyTorch with CUDA 12.8 support (before other packages)
# CUDA 12.8 wheels provide full support for RTX 5090 (Blackwell, sm_120)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python python3.11 --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining Python dependencies with UV and BuildKit cache
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python python3.11 --system -r requirements.txt

# Install Chatterbox TTS dependencies that are stripped by --no-deps
# These are pinned in chatterbox's pyproject.toml but we use compatible newer versions
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python python3.11 --system \
    transformers \
    diffusers \
    safetensors \
    einops \
    omegaconf \
    conformer \
    s3tokenizer \
    resemble-perth \
    spacy-pkuseg \
    pykakasi \
    pyloudnorm \
    soundfile \
    torchcodec

# Install chatterbox-tts with --no-deps to avoid torch/torchaudio version conflict
# chatterbox-tts requires torch==2.6.0 but RTX 5090 needs PyTorch 2.7.0+
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python python3.11 --system --no-deps chatterbox-tts

# ============================================================================
# Stage 2: Runtime - Minimal runtime image with GPU support
# ============================================================================
# CUDA 12.8 runtime with cuDNN 9 for RTX 5090 (Blackwell, sm_120)
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
# Using Python 3.11 for Chatterbox TTS compatibility
# Note: Not caching /var/lib/apt to avoid stale PPA state issues
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100 \
    && update-alternatives --set python3 /usr/bin/python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && python3 --version | grep -q "3.11" || (echo "ERROR: Python 3.11 not installed correctly" && exit 1)

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/audio_outputs /app/voice_references /root/.cache/huggingface

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/usr/local/lib/python3.11/dist-packages:$PYTHONPATH" \
    PATH="/usr/local/bin:$PATH"

# Expose backend port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/api/v1/health || exit 1

# Create entrypoint script
RUN printf '#!/bin/bash\nset -e\necho "Starting Chatterbox TTS Backend Container..."\necho ""\necho "GPU Info:"\npython3 -c "import torch; print(f\"  CUDA Available: {torch.cuda.is_available()}\"); print(f\"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")" 2>/dev/null || echo "  Unable to detect GPU"\necho ""\nif [ "${PRELOAD_MODELS:-true}" = "true" ]; then\n    echo "Pre-downloading Chatterbox TTS models..."\n    python3 scripts/setup/preload_tts_models.py --models ${PRELOAD_MODEL_LIST:-turbo} || echo "Warning: Model pre-download failed"\n    echo ""\nfi\nUVICORN_CMD="python3 -m uvicorn src.presentation.api.main:app --host 0.0.0.0 --port 8002"\nif [ "${HOT_RELOAD:-false}" = "true" ]; then\n    echo "Starting FastAPI on port 8002 (hot-reload enabled)..."\n    UVICORN_CMD="$UVICORN_CMD --reload"\nelse\n    echo "Starting FastAPI on port 8002..."\nfi\nexec $UVICORN_CMD\n' > /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
