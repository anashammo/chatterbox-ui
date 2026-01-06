"""
VoiceReference domain entity for voice cloning reference audio.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Optional, Set


@dataclass
class VoiceReference:
    """
    Domain entity for voice cloning reference audio.

    Represents an audio sample used as a reference for zero-shot voice cloning.
    Chatterbox recommends ~10 second clips for optimal voice cloning results.

    Business rules:
    - Audio duration must be between 5-30 seconds
    - Supported formats: WAV, MP3, FLAC, OGG
    - Maximum file size: 10MB
    """

    id: str
    name: str
    original_filename: str
    file_path: str
    file_size_bytes: int
    mime_type: str
    duration_seconds: float
    language: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Validation constants
    SUPPORTED_AUDIO_TYPES: ClassVar[Set[str]] = {
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/ogg",
    }

    MIN_DURATION_SECONDS: ClassVar[float] = 5.0
    MAX_DURATION_SECONDS: ClassVar[float] = 30.0
    IDEAL_DURATION_SECONDS: ClassVar[float] = 10.0
    MAX_FILE_SIZE_MB: ClassVar[int] = 10

    def __post_init__(self) -> None:
        """Validate entity on creation."""
        self._validate()

    def _validate(self) -> None:
        """Validate business rules for voice reference."""
        # Validate name
        if not self.name or not self.name.strip():
            raise ValueError("Voice name cannot be empty")

        # Validate mime type
        if self.mime_type not in self.SUPPORTED_AUDIO_TYPES:
            raise ValueError(
                f"Unsupported audio type: {self.mime_type}. "
                f"Supported types: {', '.join(self.SUPPORTED_AUDIO_TYPES)}"
            )

        # Validate duration
        if self.duration_seconds < self.MIN_DURATION_SECONDS:
            raise ValueError(
                f"Voice reference duration must be at least "
                f"{self.MIN_DURATION_SECONDS} seconds. "
                f"Got: {self.duration_seconds:.1f}s"
            )

        if self.duration_seconds > self.MAX_DURATION_SECONDS:
            raise ValueError(
                f"Voice reference duration must not exceed "
                f"{self.MAX_DURATION_SECONDS} seconds. "
                f"Got: {self.duration_seconds:.1f}s"
            )

        # Validate file size
        max_bytes = self.MAX_FILE_SIZE_MB * 1024 * 1024
        if self.file_size_bytes > max_bytes:
            raise ValueError(
                f"File size exceeds maximum of {self.MAX_FILE_SIZE_MB}MB. "
                f"Got: {self.file_size_bytes / (1024 * 1024):.1f}MB"
            )

    @property
    def is_ideal_duration(self) -> bool:
        """Check if duration is close to ideal (~10 seconds)."""
        return abs(self.duration_seconds - self.IDEAL_DURATION_SECONDS) <= 2.0

    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)

    @classmethod
    def is_supported_mime_type(cls, mime_type: str) -> bool:
        """Check if a MIME type is supported for voice references."""
        return mime_type in cls.SUPPORTED_AUDIO_TYPES

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of supported file extensions."""
        return [".wav", ".mp3", ".flac", ".ogg"]
