"""
Synthesis domain entity representing a TTS synthesis request and result.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class SynthesisStatus(Enum):
    """Status of a TTS synthesis request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TTSModel(Enum):
    """Available Chatterbox TTS models."""
    TURBO = "turbo"
    STANDARD = "standard"
    MULTILINGUAL = "multilingual"


@dataclass
class Synthesis:
    """
    Domain entity representing a TTS synthesis request and result.

    Encapsulates business rules for text-to-speech generation including:
    - Text validation (length limits)
    - Model selection and language requirements
    - Status transitions (PENDING -> PROCESSING -> COMPLETED/FAILED)
    - Voice cloning configuration
    """

    id: str
    input_text: str
    text_length: int
    model: TTSModel
    status: SynthesisStatus = SynthesisStatus.PENDING

    # Optional configuration
    language: Optional[str] = None
    voice_reference_id: Optional[str] = None
    cfg_weight: float = 0.5
    exaggeration: float = 0.5

    # Output
    output_file_path: Optional[str] = None
    output_duration_seconds: Optional[float] = None

    # Processing metadata
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Business rule constants
    MAX_TEXT_LENGTH: int = 5000
    MIN_TEXT_LENGTH: int = 1

    def __post_init__(self) -> None:
        """Validate entity on creation."""
        self._validate()

    def _validate(self) -> None:
        """Validate business rules."""
        if not self.input_text or not self.input_text.strip():
            raise ValueError("Text cannot be empty")

        if len(self.input_text) > self.MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text exceeds maximum length of {self.MAX_TEXT_LENGTH} characters"
            )

        if self.model == TTSModel.MULTILINGUAL and not self.language:
            raise ValueError("Language is required for multilingual model")

        if self.cfg_weight < 0 or self.cfg_weight > 1:
            raise ValueError("cfg_weight must be between 0 and 1")

        if self.exaggeration < 0:
            raise ValueError("exaggeration must be non-negative")

    def mark_as_processing(self) -> None:
        """
        Transition status from PENDING to PROCESSING.

        Raises:
            ValueError: If current status is not PENDING.
        """
        if self.status != SynthesisStatus.PENDING:
            raise ValueError(
                f"Cannot mark as processing from {self.status.value} status"
            )
        self.status = SynthesisStatus.PROCESSING

    def complete(
        self,
        output_file_path: str,
        output_duration_seconds: float,
        processing_time_seconds: float
    ) -> None:
        """
        Mark synthesis as successfully completed.

        Args:
            output_file_path: Path to the generated audio file.
            output_duration_seconds: Duration of the generated audio.
            processing_time_seconds: Time taken to generate the audio.

        Raises:
            ValueError: If current status is not PROCESSING.
        """
        if self.status != SynthesisStatus.PROCESSING:
            raise ValueError(
                f"Cannot complete from {self.status.value} status"
            )

        self.status = SynthesisStatus.COMPLETED
        self.output_file_path = output_file_path
        self.output_duration_seconds = output_duration_seconds
        self.processing_time_seconds = processing_time_seconds
        self.completed_at = datetime.utcnow()

    def fail(self, error_message: str) -> None:
        """
        Mark synthesis as failed.

        Args:
            error_message: Description of what went wrong.

        Raises:
            ValueError: If current status is not PROCESSING.
        """
        if self.status != SynthesisStatus.PROCESSING:
            raise ValueError(
                f"Cannot fail from {self.status.value} status"
            )

        self.status = SynthesisStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()

    @property
    def is_completed(self) -> bool:
        """Check if synthesis completed successfully."""
        return self.status == SynthesisStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if synthesis failed."""
        return self.status == SynthesisStatus.FAILED

    @property
    def is_pending(self) -> bool:
        """Check if synthesis is pending."""
        return self.status == SynthesisStatus.PENDING

    @property
    def is_processing(self) -> bool:
        """Check if synthesis is currently processing."""
        return self.status == SynthesisStatus.PROCESSING

    @property
    def has_voice_cloning(self) -> bool:
        """Check if this synthesis uses voice cloning."""
        return self.voice_reference_id is not None
