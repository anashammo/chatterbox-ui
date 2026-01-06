"""SQLite/PostgreSQL implementation of VoiceReferenceRepository."""

from typing import List, Optional
from sqlalchemy.orm import Session

from ....domain.entities.voice_reference import VoiceReference
from ....domain.repositories.voice_reference_repository import VoiceReferenceRepository
from ....domain.exceptions import RepositoryException
from ..models.voice_reference_model import VoiceReferenceModel


class SQLiteVoiceReferenceRepository(VoiceReferenceRepository):
    """
    SQLite/PostgreSQL implementation of the VoiceReferenceRepository interface.

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

    def _to_entity(self, model: VoiceReferenceModel) -> VoiceReference:
        """
        Convert ORM model to domain entity.

        Args:
            model: SQLAlchemy ORM model.

        Returns:
            Domain entity with all fields mapped.
        """
        # Create entity without triggering validation (data from DB is trusted)
        voice_ref = object.__new__(VoiceReference)
        voice_ref.id = model.id
        voice_ref.name = model.name
        voice_ref.original_filename = model.original_filename
        voice_ref.file_path = model.file_path
        voice_ref.file_size_bytes = model.file_size_bytes
        voice_ref.mime_type = model.mime_type
        voice_ref.duration_seconds = model.duration_seconds
        voice_ref.language = model.language
        voice_ref.created_at = model.created_at
        return voice_ref

    def _to_model(self, entity: VoiceReference) -> VoiceReferenceModel:
        """
        Convert domain entity to ORM model.

        Args:
            entity: Domain voice reference entity.

        Returns:
            SQLAlchemy ORM model with all fields mapped.
        """
        return VoiceReferenceModel(
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

    async def create(self, voice_reference: VoiceReference) -> VoiceReference:
        """Persist a new voice reference entity."""
        try:
            model = self._to_model(voice_reference)
            self.db.add(model)
            self.db.commit()
            self.db.refresh(model)
            return self._to_entity(model)
        except Exception as e:
            self.db.rollback()
            raise RepositoryException(f"Failed to create voice reference: {str(e)}")

    async def get_by_id(
        self,
        voice_reference_id: str
    ) -> Optional[VoiceReference]:
        """Retrieve a voice reference by its ID."""
        try:
            model = self.db.query(VoiceReferenceModel).filter(
                VoiceReferenceModel.id == voice_reference_id
            ).first()

            if model is None:
                return None

            return self._to_entity(model)
        except Exception as e:
            raise RepositoryException(f"Failed to get voice reference: {str(e)}")

    async def get_all(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[VoiceReference]:
        """Retrieve all voice references with pagination, ordered by creation date (newest first)."""
        try:
            models = (
                self.db.query(VoiceReferenceModel)
                .order_by(VoiceReferenceModel.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [self._to_entity(m) for m in models]
        except Exception as e:
            raise RepositoryException(f"Failed to get voice references: {str(e)}")

    async def delete(self, voice_reference_id: str) -> bool:
        """
        Delete a voice reference by its ID.

        Note: This does NOT cascade delete syntheses. Instead,
        syntheses keep a null voice_reference_id (SET NULL on FK).
        """
        try:
            model = self.db.query(VoiceReferenceModel).filter(
                VoiceReferenceModel.id == voice_reference_id
            ).first()

            if model is None:
                return False

            self.db.delete(model)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            raise RepositoryException(f"Failed to delete voice reference: {str(e)}")

    async def get_by_name(self, name: str) -> Optional[VoiceReference]:
        """Retrieve a voice reference by its name."""
        try:
            model = self.db.query(VoiceReferenceModel).filter(
                VoiceReferenceModel.name == name
            ).first()

            if model is None:
                return None

            return self._to_entity(model)
        except Exception as e:
            raise RepositoryException(f"Failed to get voice reference by name: {str(e)}")

    async def count(self) -> int:
        """Get total count of voice references."""
        try:
            return self.db.query(VoiceReferenceModel).count()
        except Exception as e:
            raise RepositoryException(f"Failed to count voice references: {str(e)}")
