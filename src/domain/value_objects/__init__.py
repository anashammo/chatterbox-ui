"""Domain value objects for Chatterbox TTS."""

from .tts_model_info import (
    TTSModelInfo,
    SUPPORTED_LANGUAGES,
    is_supported_language,
    get_supported_languages,
)

__all__ = [
    "TTSModelInfo",
    "SUPPORTED_LANGUAGES",
    "is_supported_language",
    "get_supported_languages",
]
