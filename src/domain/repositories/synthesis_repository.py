"""
Abstract repository interface for Synthesis entities.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..entities.synthesis import Synthesis


class SynthesisRepository(ABC):
    """
    Abstract repository interface for Synthesis entities.

    Defines the contract for persistence operations on Synthesis entities.
    Concrete implementations (e.g., SQLiteSynthesisRepository) are in
    the infrastructure layer.
    """

    @abstractmethod
    async def create(self, synthesis: Synthesis) -> Synthesis:
        """
        Persist a new synthesis entity.

        Args:
            synthesis: The synthesis entity to create.

        Returns:
            The created synthesis entity with any generated fields.
        """
        pass

    @abstractmethod
    async def get_by_id(self, synthesis_id: str) -> Optional[Synthesis]:
        """
        Retrieve a synthesis by its ID.

        Args:
            synthesis_id: The unique identifier of the synthesis.

        Returns:
            The synthesis entity if found, None otherwise.
        """
        pass

    @abstractmethod
    async def get_all(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Synthesis]:
        """
        Retrieve all syntheses with pagination.

        Args:
            limit: Maximum number of results to return.
            offset: Number of results to skip.

        Returns:
            List of synthesis entities ordered by creation date (newest first).
        """
        pass

    @abstractmethod
    async def update(self, synthesis: Synthesis) -> Synthesis:
        """
        Update an existing synthesis entity.

        Args:
            synthesis: The synthesis entity with updated fields.

        Returns:
            The updated synthesis entity.

        Raises:
            ValueError: If synthesis doesn't exist.
        """
        pass

    @abstractmethod
    async def delete(self, synthesis_id: str) -> bool:
        """
        Delete a synthesis by its ID.

        Args:
            synthesis_id: The unique identifier of the synthesis to delete.

        Returns:
            True if deleted successfully, False if not found.
        """
        pass

    @abstractmethod
    async def get_by_voice_reference_id(
        self,
        voice_reference_id: str
    ) -> List[Synthesis]:
        """
        Retrieve all syntheses that use a specific voice reference.

        Args:
            voice_reference_id: The voice reference ID to filter by.

        Returns:
            List of synthesis entities using the specified voice reference.
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """
        Get total count of syntheses.

        Returns:
            Total number of synthesis records.
        """
        pass
