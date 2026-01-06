"""
Abstract repository interface for VoiceReference entities.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities.voice_reference import VoiceReference


class VoiceReferenceRepository(ABC):
    """
    Abstract repository interface for VoiceReference entities.

    Defines the contract for persistence operations on VoiceReference entities.
    Concrete implementations (e.g., SQLiteVoiceReferenceRepository) are in
    the infrastructure layer.
    """

    @abstractmethod
    async def create(self, voice_reference: VoiceReference) -> VoiceReference:
        """
        Persist a new voice reference entity.

        Args:
            voice_reference: The voice reference entity to create.

        Returns:
            The created voice reference entity.
        """
        pass

    @abstractmethod
    async def get_by_id(
        self,
        voice_reference_id: str
    ) -> Optional[VoiceReference]:
        """
        Retrieve a voice reference by its ID.

        Args:
            voice_reference_id: The unique identifier of the voice reference.

        Returns:
            The voice reference entity if found, None otherwise.
        """
        pass

    @abstractmethod
    async def get_all(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[VoiceReference]:
        """
        Retrieve all voice references with pagination.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of voice reference entities ordered by creation date (newest first).
        """
        pass

    @abstractmethod
    async def delete(self, voice_reference_id: str) -> bool:
        """
        Delete a voice reference by its ID.

        Note: This should NOT cascade delete syntheses that used this voice.
        Instead, syntheses should keep a null reference.

        Args:
            voice_reference_id: The unique identifier of the voice reference.

        Returns:
            True if deleted successfully, False if not found.
        """
        pass

    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[VoiceReference]:
        """
        Retrieve a voice reference by its name.

        Args:
            name: The name of the voice reference.

        Returns:
            The voice reference entity if found, None otherwise.
        """
        pass

    @abstractmethod
    async def get_by_name_and_language(
        self,
        name: str,
        language: Optional[str]
    ) -> Optional[VoiceReference]:
        """
        Retrieve a voice reference by name and language combination.

        Args:
            name: The name of the voice reference.
            language: The language code (can be None).

        Returns:
            The voice reference entity if found, None otherwise.
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """
        Get total count of voice references.

        Returns:
            Total number of voice reference records.
        """
        pass

    @abstractmethod
    async def get_by_ids(self, voice_reference_ids: List[str]) -> List[VoiceReference]:
        """
        Retrieve multiple voice references by their IDs.

        Args:
            voice_reference_ids: List of voice reference IDs to fetch.

        Returns:
            List of voice reference entities found (may be fewer than requested
            if some IDs don't exist).
        """
        pass
