"""Use case for retrieving a synthesis by ID."""

from typing import Optional

from ...domain.repositories.synthesis_repository import SynthesisRepository
from ..dto.synthesis_dto import SynthesisDTO


class GetSynthesisUseCase:
    """
    Use case for retrieving a single synthesis by ID.
    """

    def __init__(self, synthesis_repository: SynthesisRepository):
        """
        Initialize use case with dependencies.

        Args:
            synthesis_repository: Repository for synthesis persistence.
        """
        self.synthesis_repo = synthesis_repository

    async def execute(self, synthesis_id: str) -> Optional[SynthesisDTO]:
        """
        Execute the get synthesis use case.

        Args:
            synthesis_id: The unique identifier of the synthesis.

        Returns:
            SynthesisDTO if found, None otherwise.
        """
        synthesis = await self.synthesis_repo.get_by_id(synthesis_id)

        if synthesis is None:
            return None

        return SynthesisDTO.from_entity(synthesis)
