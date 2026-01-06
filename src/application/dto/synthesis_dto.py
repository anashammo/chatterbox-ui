"""Data Transfer Object for Synthesis entity."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ...domain.entities.synthesis import Synthesis, SynthesisStatus, TTSModel


@dataclass
class VoiceReferenceInfoDTO:
    """Embedded voice reference information."""

    id: str
    name: str
    language: Optional[str] = None


@dataclass
class SynthesisDTO:
    """
    Data Transfer Object for Synthesis entity.

    Used for transferring synthesis data between layers without exposing
    domain entity internals or business logic methods.
    """

    id: str
    input_text: str
    text_length: int
    model: str
    status: str
    language: Optional[str]
    voice_reference_id: Optional[str]
    voice_reference: Optional[VoiceReferenceInfoDTO] = None  # Embedded voice ref info
    cfg_weight: float = 0.5
    exaggeration: float = 0.5
    output_file_path: Optional[str] = None
    output_duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None

    @classmethod
    def from_entity(cls, entity: Synthesis) -> "SynthesisDTO":
        """
        Create DTO from domain entity.

        Args:
            entity: Synthesis domain entity.

        Returns:
            SynthesisDTO with all fields mapped from entity.
        """
        return cls(
            id=entity.id,
            input_text=entity.input_text,
            text_length=entity.text_length,
            model=entity.model.value,
            status=entity.status.value,
            language=entity.language,
            voice_reference_id=entity.voice_reference_id,
            cfg_weight=entity.cfg_weight,
            exaggeration=entity.exaggeration,
            output_file_path=entity.output_file_path,
            output_duration_seconds=entity.output_duration_seconds,
            error_message=entity.error_message,
            processing_time_seconds=entity.processing_time_seconds,
            created_at=entity.created_at,
            completed_at=entity.completed_at,
        )

    def to_dict(self) -> dict:
        """Convert DTO to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "input_text": self.input_text,
            "text_length": self.text_length,
            "model": self.model,
            "status": self.status,
            "language": self.language,
            "voice_reference_id": self.voice_reference_id,
            "voice_reference": None,
            "cfg_weight": self.cfg_weight,
            "exaggeration": self.exaggeration,
            "output_file_path": self.output_file_path,
            "output_duration_seconds": self.output_duration_seconds,
            "error_message": self.error_message,
            "processing_time_seconds": self.processing_time_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
        # Add voice reference info if available
        if self.voice_reference:
            result["voice_reference"] = {
                "id": self.voice_reference.id,
                "name": self.voice_reference.name,
                "language": self.voice_reference.language,
            }
        return result


@dataclass
class SynthesisCreateDTO:
    """
    DTO for creating a new synthesis request.

    Contains only the input parameters needed for synthesis.
    """

    text: str
    model: str = "multilingual"
    language: Optional[str] = None
    voice_reference_id: Optional[str] = None
    cfg_weight: float = 0.5
    exaggeration: float = 0.5

    def validate(self) -> None:
        """
        Validate input parameters.

        Raises:
            ValueError: If validation fails.
        """
        if not self.text or not self.text.strip():
            raise ValueError("Text cannot be empty")

        if len(self.text) > 5000:
            raise ValueError("Text exceeds maximum length of 5000 characters")

        valid_models = ["turbo", "standard", "multilingual"]
        if self.model not in valid_models:
            raise ValueError(f"Invalid model: {self.model}. Must be one of: {valid_models}")

        if self.model == "multilingual" and not self.language:
            raise ValueError("Language is required for multilingual model")

        if self.cfg_weight < 0 or self.cfg_weight > 1:
            raise ValueError("cfg_weight must be between 0 and 1")

        if self.exaggeration < 0:
            raise ValueError("exaggeration must be non-negative")
