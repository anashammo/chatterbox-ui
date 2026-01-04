# Docker Deployment Guide

Complete guide for deploying the Chatterbox TTS (Text-to-Speech) system using Docker and Docker Compose.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Environment Configuration](#environment-configuration)
- [Management Scripts](#management-scripts)
- [Volume Management](#volume-management)
- [GPU Configuration](#gpu-configuration)
- [Hot-Reload Development](#hot-reload-development)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)

## Overview

The Docker deployment provides a complete containerized solution with:

- **PostgreSQL Database**: Persistent storage for synthesis history and voice references
- **FastAPI Backend**: GPU-accelerated Chatterbox TTS with CUDA 12.8
- **Angular Frontend**: Modern web interface with ng serve
- **Hot-Reload Development**: Source code changes without container rebuild
- **Volume Persistence**: All data stored in Docker volumes
- **Voice Cloning**: Zero-shot voice cloning from reference audio

## Prerequisites

### Required Software

1. **Docker Engine 20.10+**
   - Linux: `sudo apt install docker.io docker-compose`
   - Windows: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - macOS: [Docker Desktop](https://www.docker.com/products/docker-desktop/)

2. **NVIDIA Container Toolkit** (for GPU support)

   **Note**: The legacy `nvidia-docker` wrapper has been deprecated. Use NVIDIA Container Toolkit instead.

   ```bash
   # Ubuntu/Debian - Install NVIDIA Container Toolkit
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
     && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit

   # Configure Docker to use NVIDIA runtime
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

3. **Verify GPU Access**
   ```bash
   # Verify NVIDIA Container Toolkit installation
   nvidia-ctk --version

   # Test GPU access in Docker
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
   ```

### System Requirements

- **Disk Space**: 25GB+ free
  - Docker images: ~8GB (backend), ~1GB (frontend), ~200MB (postgres)
  - Chatterbox models: ~3.8GB (turbo), ~4GB (standard/multilingual) - cached in Docker volume
  - Audio outputs and voice references: Varies by usage

- **RAM**: 8GB minimum, 16GB recommended

- **GPU**: NVIDIA GPU with CUDA 12.8+ support
  - Minimum: GTX 1060 (6GB VRAM)
  - Recommended: RTX 3060+ (12GB VRAM)
  - For production: RTX 4090 or A100
  - **RTX 5090 (Blackwell)**: Verified working with CUDA 12.8 and PyTorch 2.9.1+cu128 (sm_120 architecture)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/anashammo/chatterbox-ui.git
cd chatterbox-ui
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.docker .env

# Edit .env with your preferred editor
nano .env
```

**CRITICAL**: Configure the following in your `.env`:

```bash
# Required: HuggingFace token for model downloads
# Get your free token from: https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Security: Change default password
POSTGRES_PASSWORD=your_secure_random_password_here
```

### 3. Build and Run

```bash
# Build images and start all services
python scripts/docker/run.py --build

# Wait for services to be healthy (30-60 seconds)
```

### 4. Access Application

- **Frontend**: http://localhost:4201
- **Backend API**: http://localhost:8002
- **API Documentation**: http://localhost:8002/docs
- **Health Check**: http://localhost:8002/api/v1/health

### 5. Verify GPU Access

```bash
# Open shell in backend container
python scripts/docker/shell.py backend

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Should print:
# CUDA available: True
# GPU: NVIDIA GeForce RTX 5090 (or your GPU model)
```

## Architecture

### Service Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Docker Host                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Frontend   │  │   Backend    │  │  PostgreSQL  │     │
│  │              │  │              │  │              │     │
│  │  Angular 17  │  │  FastAPI     │  │  Postgres 15 │     │
│  │  ng serve    │  │  Chatterbox  │  │              │     │
│  │  Port 4201   │  │  CUDA 12.8   │  │  Port 5432   │     │
│  │              │  │  Port 8002   │  │  (internal)  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│  ┌──────▼─────────────────▼─────────────────▼───────┐     │
│  │              Docker Volumes                      │     │
│  ├──────────────────────────────────────────────────┤     │
│  │  • chatterbox-postgres-data (database)          │     │
│  │  • chatterbox-audio-outputs (synthesized audio) │     │
│  │  • chatterbox-voice-references (voice cloning)  │     │
│  │  • chatterbox-model-cache (TTS models ~500MB)   │     │
│  │  • Source code (read-only, hot-reload)          │     │
│  └──────────────────────────────────────────────────┘     │
│                                                             │
│  ┌──────────────────────────────────────────────────┐     │
│  │           NVIDIA GPU (passed through)            │     │
│  │           Used by: backend container             │     │
│  └──────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Service Details

#### PostgreSQL Service (`postgres`)

- **Image**: `postgres:15-alpine` (lightweight PostgreSQL)
- **Port**: 5432 internal, 5433 external (to avoid conflicts)
- **Volume**: `chatterbox-postgres-data` → `/var/lib/postgresql/data`
- **Health Check**: `pg_isready -U chatterbox`
- **Purpose**: Persistent storage for synthesis history and voice references

#### Backend Service (`backend`)

- **Base Image**: `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04`
- **Port**: 8002 (exposed to host)
- **GPU**: NVIDIA GPU passed through via `nvidia` runtime
- **Volumes**:
  - `chatterbox-audio-outputs` → `/app/audio_outputs` (synthesized audio)
  - `chatterbox-voice-references` → `/app/voice_references` (voice cloning files)
  - `chatterbox-model-cache` → `/root/.cache/huggingface` (model cache)
  - `./src` → `/app/src:ro` (source code, read-only, hot-reload)
  - `./scripts` → `/app/scripts:ro` (utility scripts)
- **Environment**: PostgreSQL connection, TTS model settings
- **Health Check**: HTTP GET `http://localhost:8002/api/v1/health`
- **Dependencies**: Waits for PostgreSQL to be healthy

#### Frontend Service (`frontend`)

- **Base Image**: `node:18-alpine`
- **Port**: 4201 (exposed to host)
- **Volumes**:
  - `./src/presentation/frontend/src` → `/app/src:ro` (source code, hot-reload)
  - `./src/presentation/frontend/angular.json` → `/app/angular.json:ro`
- **Command**: `ng serve --host 0.0.0.0 --port 4201 --poll 1000`
- **Health Check**: HTTP GET `http://localhost:4201`
- **Dependencies**: Waits for backend to be healthy

### Network

All services connected via Docker bridge network:

- **Frontend** → **Backend**: `http://backend:8002/api/v1`
- **Backend** → **PostgreSQL**: `postgresql://chatterbox:password@postgres:5432/chatterbox_tts`
- **Host** → **Frontend**: `http://localhost:4201`
- **Host** → **Backend**: `http://localhost:8002`

## Environment Configuration

### .env File Structure

```bash
# ============================================================================
# PostgreSQL Database Configuration
# ============================================================================
POSTGRES_USER=chatterbox                                    # Database username
POSTGRES_PASSWORD=change_this_secure_password_in_production # Database password (CHANGE THIS!)
POSTGRES_DB=chatterbox_tts                                  # Database name
POSTGRES_PORT=5433                                          # External port (avoid conflicts)

# ============================================================================
# Chatterbox TTS Model Configuration
# ============================================================================
# Default TTS model for synthesis
# Options: turbo (350M), standard (500M), multilingual (500M)
TTS_DEFAULT_MODEL=turbo

# TTS parameters
TTS_DEFAULT_CFG_WEIGHT=0.5                                  # Accent/style transfer (0.0-1.0)
TTS_DEFAULT_EXAGGERATION=0.5                                # Expressiveness (0.0-1.0+)

# ============================================================================
# Port Configuration
# ============================================================================
BACKEND_PORT=8002                                           # Backend API port
FRONTEND_PORT=4201                                          # Frontend dev server port
```

### Port Customization

Change ports if defaults conflict with existing services:

```bash
# Use custom ports
BACKEND_PORT=8080
FRONTEND_PORT=3000
```

Then access:
- Frontend: http://localhost:3000
- Backend: http://localhost:8080

## Management Scripts

All Docker operations via Python scripts in `scripts/docker/`:

### Build Scripts

#### build.py - Build Docker Images

```bash
# Build all services
python scripts/docker/build.py

# Build specific service
python scripts/docker/build.py --backend
python scripts/docker/build.py --frontend

# Clean build (no cache)
python scripts/docker/build.py --no-cache
```

### Runtime Scripts

#### run.py - Start Services

```bash
# Start all services (use existing images)
python scripts/docker/run.py

# Build and start
python scripts/docker/run.py --build

# Run in background (detached mode)
python scripts/docker/run.py --detach
```

#### stop.py - Stop Services

```bash
# Stop containers (keep volumes)
python scripts/docker/stop.py

# Stop containers and remove volumes (WARNING: deletes all data)
python scripts/docker/stop.py --remove-volumes
```

#### rebuild.py - Rebuild and Restart

```bash
# Stop, rebuild, and restart all services
python scripts/docker/rebuild.py
```

### Debugging Scripts

#### logs.py - View Logs

```bash
# View all service logs
python scripts/docker/logs.py

# View specific service logs
python scripts/docker/logs.py backend
python scripts/docker/logs.py frontend
python scripts/docker/logs.py postgres

# Follow logs (live stream)
python scripts/docker/logs.py --follow
```

#### shell.py - Open Container Shell

```bash
# Open bash in backend container
python scripts/docker/shell.py backend

# Open sh in frontend container
python scripts/docker/shell.py frontend

# Open sh in PostgreSQL container
python scripts/docker/shell.py postgres
```

### Cleanup Scripts

#### clean.py - Remove Docker Resources

```bash
# Remove containers only
python scripts/docker/clean.py

# Remove everything (WARNING: deletes all data)
python scripts/docker/clean.py --all
```

## Volume Management

### Volume Overview

| Volume Name | Mount Point | Purpose | Size |
|------------|-------------|---------|------|
| `chatterbox-postgres-data` | `/var/lib/postgresql/data` | PostgreSQL database | ~100MB-10GB |
| `chatterbox-audio-outputs` | `/app/audio_outputs` | Synthesized audio files | Varies |
| `chatterbox-voice-references` | `/app/voice_references` | Voice cloning references | Varies |
| `chatterbox-model-cache` | `/root/.cache/huggingface` | Chatterbox TTS models | ~3.8-4GB per model |

**Note**: Models are downloaded on first use and cached in the Docker volume. First startup may take 5-10 minutes for model download.

### Volume Backup

#### Backup PostgreSQL Database

```bash
# Create backup
docker exec chatterbox-postgres pg_dump -U chatterbox chatterbox_tts > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
cat backup_20250101_120000.sql | docker exec -i chatterbox-postgres psql -U chatterbox -d chatterbox_tts
```

## GPU Configuration

### Requirements

1. **NVIDIA GPU** with CUDA Compute Capability 3.5+
2. **NVIDIA Driver** 450.80.02+ (Linux) or 452.39+ (Windows)
3. **NVIDIA Container Toolkit** installed and configured

### Verification

```bash
# Open backend shell
python scripts/docker/shell.py backend

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Expected output:
# CUDA available: True
# GPU: NVIDIA GeForce RTX 5090
```

## Hot-Reload Development

### How It Works

Source code mounted as read-only volumes:

```yaml
backend:
  volumes:
    - ./src:/app/src:ro                # Python source code
    - ./scripts:/app/scripts:ro         # Utility scripts

frontend:
  volumes:
    - ./src/presentation/frontend/src:/app/src:ro
```

**Backend**: Uvicorn `--reload` flag watches for file changes
**Frontend**: Angular `ng serve` watches for file changes

### When Rebuild Required

Hot-reload does NOT work for:

**Backend**:
- Dependency changes in `requirements.txt`
- Dockerfile changes

**Frontend**:
- Dependency changes in `package.json`
- `angular.json` or `tsconfig.json` changes

**For these changes, rebuild**:
```bash
python scripts/docker/rebuild.py
```

## Troubleshooting

### Common Issues

#### Issue: RTX 5090 CUDA Kernel Errors

**Symptom**: Synthesis fails with "CUDA kernel errors"

**Root Cause**: RTX 5090 (Blackwell architecture, sm_120) requires CUDA 12.8+

**Solution**: Ensure Dockerfile uses CUDA 12.8:
```bash
grep "nvidia/cuda" src/presentation/api/Dockerfile
# Should show: nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
```

#### Issue: Backend GPU Not Working

**Symptom**: Synthesis slow, logs show "CUDA not available"

**Solutions**:

1. Verify host GPU: `nvidia-smi`
2. Check Docker GPU runtime: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`
3. Restart Docker: `sudo systemctl restart docker`

#### Issue: Frontend Not Accessible in Browser

**Symptom**: http://localhost:4201 doesn't load or times out

**Root Cause**: Stale Docker image with wrong port configuration

**Solution**: Rebuild the frontend container:
```bash
docker compose stop frontend
docker compose rm -f frontend
docker compose build --no-cache frontend
docker compose up -d frontend
```

**Verify**: Check logs show correct port:
```bash
docker logs chatterbox-frontend --tail 20
# Should show: "Angular Live Development Server is listening on 0.0.0.0:4201"
```

#### Issue: Frontend Can't Connect to Backend

**Symptom**: Frontend shows network errors

**Solutions**:

1. Check backend health: `curl http://localhost:8002/api/v1/health`
2. Verify backend logs: `python scripts/docker/logs.py backend --tail 50`

#### Issue: SDPA Attention Error During Synthesis

**Symptom**: Synthesis fails with error:
```
TTS synthesis failed: The `output_attentions` attribute is not supported when using the `attn_implementation` set to sdpa
```

**Root Cause**: Chatterbox's T3 model creates LlamaModel/GPT2Model directly (not via `from_pretrained`) and uses `output_attentions=True`, which is incompatible with SDPA (Scaled Dot-Product Attention) in transformers >= 4.36.

**Solution**: This is automatically handled by the application. The fix involves:
1. Setting `TRANSFORMERS_ATTN_IMPLEMENTATION=eager` environment variable (configured in docker-compose.yml)
2. Monkey-patching `LlamaConfig` and `GPT2Config` `__init__` methods to force eager attention (done in `main.py`)

**Reference**: https://github.com/resemble-ai/chatterbox/issues/339

If you still encounter this error after rebuilding:
```bash
# Rebuild backend with no cache
docker compose build --no-cache backend
docker compose up -d backend
```

#### Issue: Voice Reference Upload Fails with torchcodec Error

**Symptom**: Voice reference upload fails with error:
```
Failed to read audio file: Could not load libtorchcodec
```

**Root Cause**: `torchaudio.load()` requires torchcodec/FFmpeg libraries which may not be compatible with the PyTorch version.

**Solution**: This is automatically handled. The application uses `soundfile` instead of `torchaudio` for reading audio files, which is more reliable in containerized environments.

If you encounter this error, rebuild the backend:
```bash
docker compose build --no-cache backend
docker compose up -d backend
```

#### Issue: Model Not Preloaded

**Symptom**: First synthesis takes a long time, model download starts during synthesis

**Root Cause**: Model preloading configuration may be missing or incorrect

**Solution**: Verify these environment variables in docker-compose.yml or .env:
```bash
PRELOAD_MODELS=true
PRELOAD_MODEL_LIST=turbo,standard,multilingual
```

To preload specific models only:
```bash
PRELOAD_MODEL_LIST=turbo,standard  # Only preload turbo and standard
```

Check preload logs:
```bash
docker logs chatterbox-backend | grep -i "preload\|download"
```

#### Issue: All Screens Loading During Synthesis (Fixed)

**Symptom**: When TTS synthesis is running, all other API endpoints are blocked and screens show loading spinners

**Root Cause**: GPU inference was blocking the async event loop

**Solution**: This is now fixed. The GPU inference runs in a `ThreadPoolExecutor` so other requests can be served concurrently. Health checks and other endpoints now respond in <100ms even while synthesis is running.

## Security Considerations

### Environment Variables

- **Never commit `.env` to version control**
- Add `.env` to `.gitignore`
- Use `.env.docker` as template only

### Network Security

- **Use internal networks**: PostgreSQL not exposed to host by default
- **Firewall**: Limit access to ports 4201 and 8002
- **Reverse proxy**: Use Nginx/Caddy with HTTPS

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Angular Documentation](https://angular.io/docs)

---

**Last Updated**: January 2026
**Docker Version**: 20.10+
**Docker Compose Version**: 2.0+
