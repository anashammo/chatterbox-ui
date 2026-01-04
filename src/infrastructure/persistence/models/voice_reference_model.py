"""SQLAlchemy ORM model for VoiceReference entity."""

from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.orm import relationship

from ..database import Base


class VoiceReferenceModel(Base):
    """
    SQLAlchemy ORM model for voice reference audio files.

    Maps to the 'voice_references' table in the database.
    This is a persistence model separate from the domain entity.
    """

    __tablename__ = "voice_references"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    duration_seconds = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships - syntheses that use this voice reference
    syntheses = relationship(
        "SynthesisModel",
        back_populates="voice_reference",
        cascade="save-update, merge",  # Don't cascade deletes
        passive_deletes=True  # Let the database handle SET NULL
    )

    def __repr__(self) -> str:
        return (
            f"<VoiceReferenceModel(id={self.id}, "
            f"name={self.name}, "
            f"duration={self.duration_seconds:.1f}s)>"
        )
