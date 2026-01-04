# Chatterbox TTS - Text-to-Speech UI

A professional text-to-speech system using **Resemble AI Chatterbox**, built with clean architecture principles. Features GPU-accelerated synthesis, zero-shot voice cloning, FastAPI backend, and Angular frontend.

## Features

### Core Features
- **GPU-Accelerated TTS**: Uses CUDA for fast audio synthesis with NVIDIA RTX GPUs
- **Zero-Shot Voice Cloning**: Clone any voice from a short reference audio (~10 seconds)
- **Multiple TTS Models**: Choose from turbo (fastest), standard (best quality), or multilingual (23+ languages)
- **Paralinguistic Tags**: Add natural sounds like `[laugh]`, `[cough]`, `[chuckle]` to synthesized speech
- **Clean Architecture**: Domain-driven design with clear separation of concerns
- **REST API**: FastAPI with automatic OpenAPI documentation
- **Synthesis History**: SQLite database for storing synthesis history
- **On-Premises Deployment**: All data stored locally, no cloud dependencies

### Frontend Features
- **Text Input**: Enter text to synthesize with character count and validation
- **Voice Reference Upload**: Upload audio files for voice cloning (5-30 seconds)
- **Model Selection**: Choose between turbo, standard, and multilingual models
- **Synthesis Parameters**: Adjust CFG weight and exaggeration for fine-tuning
- **Audio Playback**: Listen to synthesized audio with play/pause controls
- **Download**: Download synthesized audio files
- **History View**: Browse and manage previous syntheses
- **Voice Library**: Manage uploaded voice references
- **Dark Mode UI**: Modern, clean interface with dark theme

## Architecture

The project follows clean architecture principles:

```
src/
├── domain/           # Core business logic (entities, repositories interfaces)
├── application/      # Use cases and business workflows
├── infrastructure/   # External implementations (Chatterbox, SQLite, file storage)
└── presentation/     # Presentation layer
    ├── api/          # FastAPI REST API
    └── frontend/     # Angular web application
```

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (required for fast synthesis)
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk**: Sufficient space for audio files and Chatterbox models (~500MB per model)

### Software
- **Python**: 3.11 (required for Docker, 3.9+ for local)
- **CUDA**: CUDA 12.8 or higher (required for RTX 5090 and newer GPUs)
- **PyTorch**: 2.9.1+cu128 (verified working with RTX 5090)
- **cuDNN**: Compatible with CUDA version
- **Node.js**: 18+ (for Angular frontend)
- **HuggingFace Token**: Required for model downloads

**Note**: RTX 5090 (Blackwell architecture, sm_120) requires CUDA 12.8+ and PyTorch 2.7.0+ with sm_120 compiled binaries.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anashammo/chatterbox-ui.git
cd chatterbox-ui
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install UV Package Manager (Optional but Recommended)

UV is 10-100x faster than pip for installing Python dependencies:

```bash
# Install UV
python scripts/setup/install_uv.py

# Verify installation
uv --version
```

### 4. Install Dependencies

**Option A: Using UV (Recommended - 10-100x faster):**
```bash
uv pip install -r src/presentation/api/requirements.txt
```

**Option B: Using pip:**
```bash
pip install -r src/presentation/api/requirements.txt
```

### 5. Install PyTorch with CUDA Support

**For RTX 5090 and newer GPUs (CUDA 12.8):**
```bash
# With UV (recommended)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Or with pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 6. Install Frontend Dependencies

```bash
cd src/presentation/frontend
npm install
cd ../../..
```

### 7. Setup Environment Variables

```bash
# Copy environment template
cp .env.example .env
```

Edit `.env` to configure:
- `HF_TOKEN`: **Required** - HuggingFace token for model downloads (get from https://huggingface.co/settings/tokens)
- `TTS_DEFAULT_MODEL`: Model (turbo/standard/multilingual, default: turbo)
- `TTS_DEVICE`: cuda or cpu (default: cuda)
- `TTS_DEFAULT_CFG_WEIGHT`: CFG weight 0.0-1.0 (default: 0.5)
- `TTS_DEFAULT_EXAGGERATION`: Exaggeration 0.0-1.0+ (default: 0.5)

**Important**: The HuggingFace token is required for downloading Chatterbox models. Get your free token from https://huggingface.co/settings/tokens

### 8. Initialize Database

```bash
python scripts/setup/init_db.py
```

## Running the Application

### Start Servers

**Option 1: Using convenience scripts (Recommended)**

Open two separate terminals:

```bash
# Terminal 1 - Start Backend (port 8002)
python scripts/server/run_backend.py

# Terminal 2 - Start Frontend (port 4201)
python scripts/server/run_frontend.py
```

**Option 2: Manual start**

```bash
# Backend (from project root)
python -m uvicorn src.presentation.api.main:app --host 0.0.0.0 --port 8002 --reload

# Frontend (from project root)
cd src/presentation/frontend
ng serve --port 4201
```

The servers will be available at:
- **Backend API**: http://localhost:8002
- **API Docs**: http://localhost:8002/docs
- **Health Check**: http://localhost:8002/api/v1/health
- **Frontend**: http://localhost:4201

### Stop Servers

```bash
# Stop all servers at once
python scripts/server/stop_all.py
```

## Docker Deployment

For production deployment or containerized development, use Docker:

### Prerequisites

- Docker Engine 20.10+ with Docker Compose
- NVIDIA Container Toolkit (for GPU support)
- 15GB+ free disk space

### Quick Start

```bash
# 1. Configure environment variables
cp .env.docker .env
# Edit .env and set secure POSTGRES_PASSWORD

# 2. Build and run all services
python scripts/docker/run.py --build

# 3. Access the application
# Frontend: http://localhost:4201
# Backend API: http://localhost:8002
# API Docs: http://localhost:8002/docs
```

### Docker Management Scripts

All Docker operations are managed via Python scripts in `scripts/docker/`:

```bash
# Build images
python scripts/docker/build.py              # Build all services
python scripts/docker/build.py --backend    # Build backend only
python scripts/docker/build.py --no-cache   # Clean build

# Run services
python scripts/docker/run.py                # Start all services
python scripts/docker/run.py --build        # Build and start
python scripts/docker/run.py --detach       # Run in background

# View logs
python scripts/docker/logs.py               # All services
python scripts/docker/logs.py backend       # Backend only
python scripts/docker/logs.py --follow      # Follow logs

# Stop services
python scripts/docker/stop.py               # Stop containers
python scripts/docker/stop.py -v            # Stop and remove volumes

# Open shell in container
python scripts/docker/shell.py backend      # Backend shell

# Clean up resources
python scripts/docker/clean.py --all        # Remove everything
```

For detailed Docker deployment guide, see [DOCKER.md](DOCKER.md).

## API Endpoints

### Synthesis Endpoints

#### Synthesize Text
```http
POST /api/v1/syntheses
Content-Type: multipart/form-data

Parameters:
- text: Text to synthesize (required)
- model: TTS model (optional, default: 'turbo', options: turbo/standard/multilingual)
- voice_reference_id: Voice reference ID for cloning (optional)
- language: Language code for multilingual model (optional)
- cfg_weight: CFG weight 0.0-1.0 (optional, default: 0.5)
- exaggeration: Exaggeration 0.0-1.0+ (optional, default: 0.5)

Response:
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "text": "Hello world",
  "status": "completed",
  "model": "turbo",
  "audio_file_path": "/audio_outputs/550e8400.wav",
  "duration_seconds": 1.5,
  "processing_time_seconds": 0.3,
  "created_at": "2026-01-04T10:30:00Z",
  "completed_at": "2026-01-04T10:30:01Z"
}
```

#### Get Synthesis History
```http
GET /api/v1/syntheses?limit=100&offset=0

Response:
{
  "items": [...],
  "total": 10,
  "limit": 100,
  "offset": 0
}
```

#### Get Specific Synthesis
```http
GET /api/v1/syntheses/{synthesis_id}
```

#### Delete Synthesis
```http
DELETE /api/v1/syntheses/{synthesis_id}

Response: 204 No Content
```

#### Download Synthesized Audio
```http
GET /api/v1/syntheses/{synthesis_id}/audio

Response: Audio file (WAV format)
```

### Voice Reference Endpoints

#### Upload Voice Reference
```http
POST /api/v1/voice-references
Content-Type: multipart/form-data

Parameters:
- file: Audio file (required, 5-30 seconds)
- name: Reference name (required)
- description: Description (optional)

Response:
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "name": "My Voice",
  "description": "Personal voice reference",
  "file_path": "/voice_references/660e8400.wav",
  "duration_seconds": 10.5,
  "created_at": "2026-01-04T10:00:00Z"
}
```

#### Get Voice References
```http
GET /api/v1/voice-references

Response:
[
  {
    "id": "660e8400-e29b-41d4-a716-446655440001",
    "name": "My Voice",
    "description": "Personal voice reference",
    "duration_seconds": 10.5,
    "created_at": "2026-01-04T10:00:00Z"
  }
]
```

#### Delete Voice Reference
```http
DELETE /api/v1/voice-references/{voice_reference_id}

Response: 204 No Content
```

### Model Endpoints

#### Get Available Models
```http
GET /api/v1/models/available

Response:
{
  "models": [
    {"code": "turbo", "name": "Turbo", "size": "~350MB", "description": "Fastest, single-step generation"},
    {"code": "standard", "name": "Standard", "size": "~500MB", "description": "Best quality"},
    {"code": "multilingual", "name": "Multilingual", "size": "~500MB", "description": "23+ languages"}
  ]
}
```

### Health Endpoints

#### Health Check
```http
GET /api/v1/health

Response:
{
  "status": "healthy",
  "message": "Chatterbox TTS API is running"
}
```

## Chatterbox TTS Models

| Model | Parameters | Cache Size | Speed | Best For |
|-------|-----------|------------|-------|----------|
| **turbo** | 350M | ~3.8GB | Fastest | Quick synthesis, real-time applications |
| **standard** | 500M | ~4.0GB | Good | Best quality, fine-tuning with CFG/exaggeration |
| **multilingual** | 500M | ~4.0GB | Good | Multiple languages, zero-shot voice cloning |

**Note**: Cache sizes include all model components (embeddings, tokenizers, etc.). Models are downloaded on first use.

**Model Selection Guide**:
- **For fast synthesis**: Use `turbo` for real-time applications
- **For best quality**: Use `standard` with CFG and exaggeration tuning
- **For multiple languages**: Use `multilingual` (supports 23+ languages)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | (required) | HuggingFace token for model downloads |
| `TTS_DEFAULT_MODEL` | turbo | Default TTS model |
| `TTS_DEVICE` | cuda | Device (cuda/cpu) |
| `TTS_DEFAULT_CFG_WEIGHT` | 0.5 | Default CFG weight |
| `TTS_DEFAULT_EXAGGERATION` | 0.5 | Default exaggeration |
| `DATABASE_URL` | sqlite:///./chatterbox_tts.db | Database connection |
| `AUDIO_OUTPUT_DIR` | ./audio_outputs | Output directory |
| `VOICE_REFERENCE_DIR` | ./voice_references | Voice reference directory |
| `API_PORT` | 8002 | API port |
| `CORS_ORIGINS` | ["http://localhost:4201"] | Allowed origins |

## Troubleshooting

### CUDA Not Available

If you see "CUDA requested but not available":
- Verify NVIDIA GPU drivers are installed
- Ensure CUDA toolkit is installed (CUDA 12.8+ for RTX 5090)
- Check PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

### RTX 5090 CUDA Kernel Errors

If you see "CUDA kernel errors" with RTX 5090:

**Root Cause**: RTX 5090 (Blackwell architecture, sm_120) requires CUDA 12.8+ and PyTorch 2.7.0+.

**Solutions**:
1. **Update CUDA toolkit to 12.8+**
2. **Reinstall PyTorch with CUDA 12.8 support**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

### Import Errors

If you encounter import errors:
- Ensure virtual environment is activated
- Verify all dependencies are installed:
  ```bash
  pip install -r src/presentation/api/requirements.txt
  ```

### Database Errors

If database errors occur:
- Delete the database file and run `python scripts/setup/init_db.py` again
- Check file permissions on the database file

### SDPA Attention Error During Synthesis

If you see:
```
TTS synthesis failed: The `output_attentions` attribute is not supported when using the `attn_implementation` set to sdpa
```

**Root Cause**: Chatterbox's T3 model uses `output_attentions=True`, which is incompatible with SDPA in transformers >= 4.36.

**Solution**: This is automatically handled by the application through:
1. Environment variable `TRANSFORMERS_ATTN_IMPLEMENTATION=eager`
2. Monkey-patching `LlamaConfig` and `GPT2Config` in `main.py`

Reference: https://github.com/resemble-ai/chatterbox/issues/339

### Voice Reference Upload Fails

If voice reference upload fails with "torchcodec" or audio reading errors:

**Solution**: The application uses `soundfile` instead of `torchaudio` for reliable audio processing. Ensure `soundfile` is installed:
```bash
pip install soundfile
```

## License

This project is for educational and internal use.

## Support

For issues or questions, please refer to the project documentation or create an issue in the repository.

## Credits

- [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) - Text-to-Speech library
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [Angular](https://angular.io/) - Frontend framework
