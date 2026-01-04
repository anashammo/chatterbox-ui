"""SQLite/PostgreSQL implementation of SynthesisRepository."""

from typing import List, Optional
from sqlalchemy.orm import Session

from ....domain.entities.synthesis import Synthesis, SynthesisStatus, TTSModel
from ....domain.repositories.synthesis_repository import SynthesisRepository
from ....domain.exceptions import RepositoryException
from ..models.synthesis_model import SynthesisModel, SynthesisStatusEnum, TTSModelEnum


class SQLiteSynthesisRepository(SynthesisRepository):
    """
    SQLite/PostgreSQL implementation of the SynthesisRepository interface.

    Uses SQLAlchemy ORM for database operations.
    Maps between domain entities and ORM models.
    """

    def __init__(self, db: Session):
        """
        Initialize repository with database session.

        Args:
            db: SQLAlchemy database session.
        """
        self.db = db

    def _status_to_enum(self, status: SynthesisStatus) -> SynthesisStatusEnum:
        """Convert domain status to ORM enum."""
        return SynthesisStatusEnum(status.value)

    def _enum_to_status(self, enum_val: SynthesisStatusEnum) -> SynthesisStatus:
        """Convert ORM enum to domain status."""
        return SynthesisStatus(enum_val.value)

    def _model_to_enum(self, model: TTSModel) -> TTSModelEnum:
        """Convert domain model to ORM enum."""
        return TTSModelEnum(model.value)

    def _enum_to_model(self, enum_val: TTSModelEnum) -> TTSModel:
        """Convert ORM enum to domain model."""
        return TTSModel(enum_val.value)

    def _to_entity(self, model: SynthesisModel) -> Synthesis:
        """
        Convert ORM model to domain entity.

        Args:
            model: SQLAlchemy ORM model.

        Returns:
            Domain entity with all fields mapped.
        """
        # Create entity without triggering validation
        synthesis = object.__new__(Synthesis)
        synthesis.id = model.id
        synthesis.input_text = model.input_text
        synthesis.text_length = int(model.text_length)
        synthesis.model = self._enum_to_model(model.model)
        synthesis.status = self._enum_to_status(model.status)
        synthesis.language = model.language
        synthesis.voice_reference_id = model.voice_reference_id
        synthesis.cfg_weight = model.cfg_weight
        synthesis.exaggeration = model.exaggeration
        synthesis.output_file_path = model.output_file_path
        synthesis.output_duration_seconds = model.output_duration_seconds
        synthesis.error_message = model.error_message
        synthesis.processing_time_seconds = model.processing_time_seconds
        synthesis.created_at = model.created_at
        synthesis.completed_at = model.completed_at
        return synthesis

    def _to_model(self, entity: Synthesis) -> SynthesisModel:
        """
        Convert domain entity to ORM model.

        Args:
            entity: Domain synthesis entity.

        Returns:
            SQLAlchemy ORM model with all fields mapped.
        """
        return SynthesisModel(
            id=entity.id,
            input_text=entity.input_text,
            text_length=entity.text_length,
            model=self._model_to_enum(entity.model),
            status=self._status_to_enum(entity.status),
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

    async def create(self, synthesis: Synthesis) -> Synthesis:
        """Persist a new synthesis entity."""
        try:
            model = self._to_model(synthesis)
            self.db.add(model)
            self.db.commit()
            self.db.refresh(model)
            return self._to_entity(model)
        except Exception as e:
            self.db.rollback()
            raise RepositoryException(f"Failed to create synthesis: {str(e)}")

    async def get_by_id(self, synthesis_id: str) -> Optional[Synthesis]:
        """Retrieve a synthesis by its ID."""
        try:
            model = self.db.query(SynthesisModel).filter(
                SynthesisModel.id == synthesis_id
            ).first()

            if model is None:
                return None

            return self._to_entity(model)
        except Exception as e:
            raise RepositoryException(f"Failed to get synthesis: {str(e)}")

    async def get_all(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Synthesis]:
        """Retrieve all syntheses with pagination, ordered by creation date (newest first)."""
        try:
            models = (
                self.db.query(SynthesisModel)
                .order_by(SynthesisModel.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [self._to_entity(m) for m in models]
        except Exception as e:
            raise RepositoryException(f"Failed to get syntheses: {str(e)}")

    async def update(self, synthesis: Synthesis) -> Synthesis:
        """Update an existing synthesis entity."""
        try:
            model = self.db.query(SynthesisModel).filter(
                SynthesisModel.id == synthesis.id
            ).first()

            if model is None:
                raise ValueError(f"Synthesis not found: {synthesis.id}")

            # Update all fields
            model.input_text = synthesis.input_text
            model.text_length = synthesis.text_length
            model.model = self._model_to_enum(synthesis.model)
            model.status = self._status_to_enum(synthesis.status)
            model.language = synthesis.language
            model.voice_reference_id = synthesis.voice_reference_id
            model.cfg_weight = synthesis.cfg_weight
            model.exaggeration = synthesis.exaggeration
            model.output_file_path = synthesis.output_file_path
            model.output_duration_seconds = synthesis.output_duration_seconds
            model.error_message = synthesis.error_message
            model.processing_time_seconds = synthesis.processing_time_seconds
            model.completed_at = synthesis.completed_at

            self.db.commit()
            self.db.refresh(model)
            return self._to_entity(model)
        except ValueError:
            raise
        except Exception as e:
            self.db.rollback()
            raise RepositoryException(f"Failed to update synthesis: {str(e)}")

    async def delete(self, synthesis_id: str) -> bool:
        """Delete a synthesis by its ID."""
        try:
            model = self.db.query(SynthesisModel).filter(
                SynthesisModel.id == synthesis_id
            ).first()

            if model is None:
                return False

            self.db.delete(model)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            raise RepositoryException(f"Failed to delete synthesis: {str(e)}")

    async def get_by_voice_reference_id(
        self,
        voice_reference_id: str
    ) -> List[Synthesis]:
        """Retrieve all syntheses that use a specific voice reference."""
        try:
            models = (
                self.db.query(SynthesisModel)
                .filter(SynthesisModel.voice_reference_id == voice_reference_id)
                .order_by(SynthesisModel.created_at.desc())
                .all()
            )
            return [self._to_entity(m) for m in models]
        except Exception as e:
            raise RepositoryException(
                f"Failed to get syntheses by voice reference: {str(e)}"
            )

    async def count(self) -> int:
        """Get total count of syntheses."""
        try:
            return self.db.query(SynthesisModel).count()
        except Exception as e:
            raise RepositoryException(f"Failed to count syntheses: {str(e)}")
