"""
TTS Model information value object.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TTSModelInfo:
    """
    Immutable value object containing TTS model specifications.

    Provides metadata about available Chatterbox TTS models including
    their capabilities and resource requirements.
    """

    name: str
    display_name: str
    parameters: str
    description: str
    supports_voice_cloning: bool
    supports_multilingual: bool
    supports_paralinguistics: bool

    @classmethod
    def get_all_models(cls) -> List["TTSModelInfo"]:
        """
        Get information about all available Chatterbox TTS models.

        Returns:
            List of TTSModelInfo for each available model.
        """
        return [
            cls(
                name="turbo",
                display_name="Chatterbox Turbo",
                parameters="350M",
                description="Fastest model with single-step generation. "
                           "Best for real-time applications.",
                supports_voice_cloning=True,
                supports_multilingual=False,
                supports_paralinguistics=True,
            ),
            cls(
                name="standard",
                display_name="Chatterbox Standard",
                parameters="500M",
                description="Highest quality model with cfg_weight and "
                           "exaggeration tuning. Best for production quality.",
                supports_voice_cloning=True,
                supports_multilingual=False,
                supports_paralinguistics=True,
            ),
            cls(
                name="multilingual",
                display_name="Chatterbox Multilingual",
                parameters="500M",
                description="Supports 23+ languages with zero-shot voice cloning. "
                           "Best for international content.",
                supports_voice_cloning=True,
                supports_multilingual=True,
                supports_paralinguistics=True,
            ),
        ]

    @classmethod
    def get_by_name(cls, name: str) -> "TTSModelInfo | None":
        """
        Get model info by name.

        Args:
            name: Model name (turbo, standard, multilingual).

        Returns:
            TTSModelInfo if found, None otherwise.
        """
        for model in cls.get_all_models():
            if model.name == name:
                return model
        return None

    @classmethod
    def get_model_names(cls) -> List[str]:
        """Get list of available model names."""
        return [model.name for model in cls.get_all_models()]

    @classmethod
    def is_valid_model(cls, name: str) -> bool:
        """Check if a model name is valid."""
        return name in cls.get_model_names()


# Supported languages for multilingual model
SUPPORTED_LANGUAGES = [
    "en",  # English
    "es",  # Spanish
    "fr",  # French
    "de",  # German
    "it",  # Italian
    "pt",  # Portuguese
    "nl",  # Dutch
    "pl",  # Polish
    "ru",  # Russian
    "ja",  # Japanese
    "ko",  # Korean
    "zh",  # Chinese
    "ar",  # Arabic
    "hi",  # Hindi
    "tr",  # Turkish
    "vi",  # Vietnamese
    "th",  # Thai
    "id",  # Indonesian
    "sv",  # Swedish
    "da",  # Danish
    "no",  # Norwegian
    "fi",  # Finnish
    "cs",  # Czech
]


def is_supported_language(language_code: str) -> bool:
    """Check if a language code is supported."""
    return language_code.lower() in SUPPORTED_LANGUAGES


def get_supported_languages() -> List[str]:
    """Get list of supported language codes."""
    return SUPPORTED_LANGUAGES.copy()
