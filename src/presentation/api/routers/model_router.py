"""Model management API endpoints for Chatterbox TTS."""

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.model_schema import (
    TTSModelResponse,
    ModelListResponse,
    ModelStatusResponse,
    LanguageListResponse,
    GPUInfoResponse,
)
from ..dependencies import get_tts_service
from ....infrastructure.services.chatterbox_service import ChatterboxService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("/available", response_model=ModelListResponse)
async def get_available_models(
    tts_service: ChatterboxService = Depends(get_tts_service),
) -> ModelListResponse:
    """
    Get list of available Chatterbox TTS models with their specifications.

    Returns all available models with their capabilities and current load status.
    """
    models = tts_service.get_available_models()

    return ModelListResponse(
        models=[
            TTSModelResponse(
                name=m["name"],
                display_name=m["display_name"],
                parameters=m["parameters"],
                description=m["description"],
                supports_voice_cloning=m["supports_voice_cloning"],
                supports_multilingual=m["supports_multilingual"],
                supports_paralinguistics=m["supports_paralinguistics"],
                is_loaded=m["is_loaded"],
            )
            for m in models
        ]
    )


@router.get("/status/{model_name}", response_model=ModelStatusResponse)
async def get_model_status(
    model_name: str,
    tts_service: ChatterboxService = Depends(get_tts_service),
) -> ModelStatusResponse:
    """
    Check if a specific model is currently loaded in memory.

    Returns the load status for the specified model.
    """
    valid_models = ["turbo", "standard", "multilingual"]
    if model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name: {model_name}. Valid models: {valid_models}"
        )

    is_loaded = tts_service.is_model_loaded(model_name)

    return ModelStatusResponse(
        model_name=model_name,
        is_loaded=is_loaded,
    )


@router.post("/load/{model_name}", response_model=ModelStatusResponse)
async def preload_model(
    model_name: str,
    tts_service: ChatterboxService = Depends(get_tts_service),
) -> ModelStatusResponse:
    """
    Preload a model into GPU memory.

    This can be used to warm up the model before first synthesis.
    """
    try:
        tts_service.preload_model(model_name)
        return ModelStatusResponse(
            model_name=model_name,
            is_loaded=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post("/unload/{model_name}", response_model=ModelStatusResponse)
async def unload_model(
    model_name: str,
    tts_service: ChatterboxService = Depends(get_tts_service),
) -> ModelStatusResponse:
    """
    Unload a model from GPU memory.

    Use this to free GPU resources when a model is no longer needed.
    """
    valid_models = ["turbo", "standard", "multilingual"]
    if model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name: {model_name}. Valid models: {valid_models}"
        )

    tts_service.unload_model(model_name)

    return ModelStatusResponse(
        model_name=model_name,
        is_loaded=False,
    )


@router.get("/languages", response_model=LanguageListResponse)
async def get_supported_languages(
    tts_service: ChatterboxService = Depends(get_tts_service),
) -> LanguageListResponse:
    """
    Get list of supported language codes for the multilingual model.

    Returns ISO language codes supported by the multilingual TTS model.
    """
    languages = tts_service.get_supported_languages()

    return LanguageListResponse(
        languages=languages,
        count=len(languages),
    )


@router.get("/gpu-info", response_model=GPUInfoResponse)
async def get_gpu_info(
    tts_service: ChatterboxService = Depends(get_tts_service),
) -> GPUInfoResponse:
    """
    Get GPU information for diagnostics.

    Returns CUDA device info including memory usage.
    """
    gpu_info = tts_service.get_gpu_info()
    return GPUInfoResponse(**gpu_info)
