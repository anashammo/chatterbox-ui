"""SQLAlchemy ORM model for Synthesis entity."""

from datetime import datetime
from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
import enum

from ..database import Base


class SynthesisStatusEnum(enum.Enum):
    """Database enum for synthesis status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TTSModelEnum(enum.Enum):
    """Database enum for TTS model selection."""
    TURBO = "turbo"
    STANDARD = "standard"
    MULTILINGUAL = "multilingual"


class SynthesisModel(Base):
    """
    SQLAlchemy ORM model for TTS synthesis records.

    Maps to the 'syntheses' table in the database.
    This is a persistence model separate from the domain entity.
    """

    __tablename__ = "syntheses"

    id = Column(String(36), primary_key=True)
    input_text = Column(Text, nullable=False)
    text_length = Column(Float, nullable=False)

    model = Column(
        SQLEnum(TTSModelEnum, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=TTSModelEnum.TURBO
    )
    status = Column(
        SQLEnum(SynthesisStatusEnum, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=SynthesisStatusEnum.PENDING
    )

    # Optional configuration
    language = Column(String(10), nullable=True)
    voice_reference_id = Column(
        String(36),
        ForeignKey("voice_references.id", ondelete="SET NULL"),
        nullable=True
    )
    cfg_weight = Column(Float, nullable=False, default=0.5)
    exaggeration = Column(Float, nullable=False, default=0.5)

    # Output
    output_file_path = Column(String(500), nullable=True)
    output_duration_seconds = Column(Float, nullable=True)

    # Processing metadata
    error_message = Column(Text, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    voice_reference = relationship("VoiceReferenceModel", back_populates="syntheses")

    def __repr__(self) -> str:
        return (
            f"<SynthesisModel(id={self.id}, "
            f"model={self.model.value if self.model else None}, "
            f"status={self.status.value if self.status else None})>"
        )
