"""Data Transfer Object for VoiceReference entity."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ...domain.entities.voice_reference import VoiceReference


@dataclass
class VoiceReferenceDTO:
    """
    Data Transfer Object for VoiceReference entity.

    Used for transferring voice reference data between layers without exposing
    domain entity internals or business logic methods.
    """

    id: str
    name: str
    original_filename: str
    file_path: str
    file_size_bytes: int
    mime_type: str
    duration_seconds: float
    language: Optional[str]
    created_at: datetime

    @classmethod
    def from_entity(cls, entity: VoiceReference) -> "VoiceReferenceDTO":
        """
        Create DTO from domain entity.

        Args:
            entity: VoiceReference domain entity.

        Returns:
            VoiceReferenceDTO with all fields mapped from entity.
        """
        return cls(
            id=entity.id,
            name=entity.name,
            original_filename=entity.original_filename,
            file_path=entity.file_path,
            file_size_bytes=entity.file_size_bytes,
            mime_type=entity.mime_type,
            duration_seconds=entity.duration_seconds,
            language=entity.language,
            created_at=entity.created_at,
        )

    def to_dict(self) -> dict:
        """Convert DTO to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "original_filename": self.original_filename,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "file_size_mb": round(self.file_size_bytes / (1024 * 1024), 2),
            "mime_type": self.mime_type,
            "duration_seconds": self.duration_seconds,
            "language": self.language,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class VoiceReferenceCreateDTO:
    """
    DTO for creating a new voice reference.

    Contains only the input parameters needed for voice reference creation.
    """

    name: str
    original_filename: str
    file_content: bytes
    mime_type: str
    duration_seconds: float
    language: Optional[str] = None

    def validate(self) -> None:
        """
        Validate input parameters.

        Raises:
            ValueError: If validation fails.
        """
        if not self.name or not self.name.strip():
            raise ValueError("Voice name cannot be empty")

        if not self.original_filename:
            raise ValueError("Original filename is required")

        if not self.file_content:
            raise ValueError("File content cannot be empty")

        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024
        if len(self.file_content) > max_size:
            raise ValueError(
                f"File size exceeds maximum of 10MB. "
                f"Got: {len(self.file_content) / (1024 * 1024):.1f}MB"
            )

        # Check MIME type
        supported_types = {
            "audio/wav",
            "audio/x-wav",
            "audio/mpeg",
            "audio/mp3",
            "audio/flac",
            "audio/ogg",
        }
        if self.mime_type not in supported_types:
            raise ValueError(
                f"Unsupported audio type: {self.mime_type}. "
                f"Supported: {', '.join(supported_types)}"
            )

        # Check duration (5-30 seconds)
        if self.duration_seconds < 5.0:
            raise ValueError(
                f"Voice reference must be at least 5 seconds. "
                f"Got: {self.duration_seconds:.1f}s"
            )

        if self.duration_seconds > 30.0:
            raise ValueError(
                f"Voice reference must not exceed 30 seconds. "
                f"Got: {self.duration_seconds:.1f}s"
            )
