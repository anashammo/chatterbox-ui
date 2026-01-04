"""API routers for Chatterbox TTS."""

from . import synthesis_router
from . import voice_reference_router
from . import health_router
from . import model_router

__all__ = [
    "synthesis_router",
    "voice_reference_router",
    "health_router",
    "model_router",
]
