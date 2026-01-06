"""FastAPI application initialization for Chatterbox TTS."""

# CRITICAL: Fix SDPA attention issue with Chatterbox TTS BEFORE importing transformers
# See: https://github.com/resemble-ai/chatterbox/issues/339
#
# The issue is that Chatterbox's T3 model creates LlamaModel/GPT2Model directly
# (not via from_pretrained) and uses output_attentions=True, which is incompatible
# with SDPA (Scaled Dot-Product Attention) in transformers >= 4.36
import os
os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'eager'

# Patch LlamaConfig and GPT2Config to default to eager attention
try:
    from transformers import LlamaConfig, GPT2Config

    # Store original __init__ methods
    _original_llama_init = LlamaConfig.__init__
    _original_gpt2_init = GPT2Config.__init__

    def _patched_llama_init(self, *args, **kwargs):
        # Force eager attention implementation
        if '_attn_implementation' not in kwargs:
            kwargs['_attn_implementation'] = 'eager'
        if 'attn_implementation' not in kwargs:
            kwargs['attn_implementation'] = 'eager'
        _original_llama_init(self, *args, **kwargs)
        # Ensure it's set after init as well
        self._attn_implementation = 'eager'

    def _patched_gpt2_init(self, *args, **kwargs):
        # Force eager attention implementation
        if '_attn_implementation' not in kwargs:
            kwargs['_attn_implementation'] = 'eager'
        if 'attn_implementation' not in kwargs:
            kwargs['attn_implementation'] = 'eager'
        _original_gpt2_init(self, *args, **kwargs)
        self._attn_implementation = 'eager'

    LlamaConfig.__init__ = _patched_llama_init
    GPT2Config.__init__ = _patched_gpt2_init

except Exception as e:
    print(f"Warning: Failed to patch transformers configs for eager attention: {e}")

import json
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.presentation.api.routers import synthesis_router, voice_reference_router, health_router, model_router
from src.infrastructure.config.settings import get_settings
from src.infrastructure.persistence.database import init_db

logger = logging.getLogger(__name__)


class UTF8JSONResponse(JSONResponse):
    """
    Custom JSON response class that properly handles Unicode characters.

    By default, FastAPI/Starlette uses ensure_ascii=True which escapes
    non-ASCII characters. This class sets ensure_ascii=False to properly
    render Arabic, Chinese, and other Unicode text.
    """
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,  # Allow Unicode characters (Arabic, etc.)
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown events.
    """
    # Startup: Add ffmpeg to PATH (if available)
    project_root = Path(__file__).parent.parent.parent.parent
    ffmpeg_bin = project_root / "ffmpeg-8.0.1-essentials_build" / "bin"
    if ffmpeg_bin.exists():
        os.environ["PATH"] = str(ffmpeg_bin) + os.pathsep + os.environ.get("PATH", "")
        print(f"Added ffmpeg to PATH: {ffmpeg_bin}")

    # Startup: Ensure output directories exist
    audio_output_dir = Path(settings.audio_output_dir)
    voice_ref_dir = Path(settings.voice_reference_dir)
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    voice_ref_dir.mkdir(parents=True, exist_ok=True)
    print(f"Audio output directory: {audio_output_dir.absolute()}")
    print(f"Voice reference directory: {voice_ref_dir.absolute()}")

    # Startup: Initialize database
    print("Initializing database...")
    init_db()
    print("Database initialized successfully")

    # Startup: JIT warmup synthesis (prevents first-request delay)
    if os.environ.get("WARMUP_ON_STARTUP", "true").lower() == "true":
        print("Starting JIT warmup synthesis...")
        try:
            from src.presentation.api.dependencies import get_tts_service

            tts_service = get_tts_service()

            # Run warmup synthesis with Arabic text using multilingual model
            # This triggers JIT compilation so first real request is fast
            result = await tts_service.synthesize(
                text="مرحبا، هذا اختبار تسخين النظام",
                model="multilingual",
                language="ar",
                cfg_weight=0.5,
                exaggeration=0.5
            )

            processing_time = result.get("processing_time_seconds", 0)
            print(f"JIT warmup completed in {processing_time:.2f}s")

            # Note: tts_service.synthesize() returns audio_data bytes in memory
            # No DB record or file is created - those are handled by the use case layer
            # So no cleanup is needed

        except Exception as e:
            logger.warning(f"JIT warmup failed (non-fatal): {e}")
            print(f"Warning: JIT warmup failed: {e}")

    yield

    # Shutdown: Cleanup if needed
    print("Shutting down Chatterbox TTS...")


# Initialize FastAPI app with UTF-8 JSON response for proper Unicode handling
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Text-to-Speech API using Resemble AI Chatterbox",
    debug=settings.debug,
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,  # Proper Arabic/Unicode text in responses
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    health_router.router,
    prefix=settings.api_prefix,
    tags=["health"]
)

app.include_router(
    synthesis_router.router,
    prefix=settings.api_prefix,
    tags=["syntheses"]
)

app.include_router(
    voice_reference_router.router,
    prefix=settings.api_prefix,
    tags=["voice-references"]
)

app.include_router(
    model_router.router,
    prefix=settings.api_prefix,
    tags=["models"]
)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Chatterbox TTS API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
        "endpoints": {
            "syntheses": f"{settings.api_prefix}/syntheses",
            "voice_references": f"{settings.api_prefix}/voice-references",
            "models": f"{settings.api_prefix}/models",
        }
    }
