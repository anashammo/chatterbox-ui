"""API router for health check endpoints - Chatterbox TTS."""

from fastapi import APIRouter, Depends
from src.infrastructure.config.settings import Settings, get_settings
from src.presentation.api.dependencies import get_tts_service
from src.infrastructure.services.chatterbox_service import ChatterboxService

router = APIRouter()


@router.get(
    "/health",
    summary="Health check",
    description="Check if the API is running and healthy."
)
async def health_check():
    """
    Health check endpoint.

    Returns basic status information about the API.
    """
    return {
        "status": "healthy",
        "message": "Chatterbox TTS API is running"
    }


@router.get(
    "/info",
    summary="System information",
    description="Get information about the system and TTS configuration."
)
async def system_info(
    tts_service: ChatterboxService = Depends(get_tts_service),
    settings: Settings = Depends(get_settings)
):
    """
    Get system information.

    Returns information about the TTS service and configuration.
    """
    gpu_info = tts_service.get_gpu_info()
    models = tts_service.get_available_models()

    loaded_models = [m["name"] for m in models if m["is_loaded"]]

    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "default_model": settings.tts_default_model,
        "loaded_models": loaded_models,
        "gpu_available": gpu_info.get("available", False),
        "gpu_name": gpu_info.get("name"),
        "supported_languages_count": len(tts_service.get_supported_languages()),
        "max_text_length": settings.tts_max_text_length,
    }
