"""
Abstract interface for Text-to-Speech services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class TextToSpeechService(ABC):
    """
    Abstract interface for TTS services.

    This interface defines the contract for text-to-speech implementations.
    The infrastructure layer provides concrete implementations (e.g., ChatterboxService).
    """

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        model: str = "turbo",
        voice_reference_path: Optional[str] = None,
        language: Optional[str] = None,
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text.

        Args:
            text: The text to convert to speech.
            model: TTS model to use (turbo, standard, multilingual).
            voice_reference_path: Path to voice reference audio for cloning.
            language: Language code for multilingual model.
            cfg_weight: Configuration weight for accent transfer (0-1).
            exaggeration: Speech expressiveness (0-1+).

        Returns:
            Dict containing:
                - audio_data: bytes - Raw audio data
                - duration_seconds: float - Audio duration
                - sample_rate: int - Audio sample rate

        Raises:
            ValueError: If parameters are invalid.
            RuntimeError: If synthesis fails.
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available TTS models with metadata.

        Returns:
            List of dicts with model information:
                - name: str - Model identifier
                - display_name: str - Human-readable name
                - parameters: str - Model size (e.g., "350M")
                - description: str - Model description
                - supports_voice_cloning: bool
                - supports_multilingual: bool
                - supports_paralinguistics: bool
        """
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes for multilingual model.

        Returns:
            List of ISO language codes (e.g., ["en", "es", "fr", ...])
        """
        pass

    @abstractmethod
    def is_model_loaded(self, model: str) -> bool:
        """
        Check if a specific model is currently loaded in memory.

        Args:
            model: Model name to check.

        Returns:
            True if model is loaded and ready for inference.
        """
        pass
