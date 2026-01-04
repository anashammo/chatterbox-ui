"""Use case for deleting a synthesis."""

from ...domain.repositories.synthesis_repository import SynthesisRepository
from ..interfaces.file_storage_interface import FileStorageInterface


class DeleteSynthesisUseCase:
    """
    Use case for deleting a synthesis and its output audio file.
    """

    def __init__(
        self,
        synthesis_repository: SynthesisRepository,
        audio_storage: FileStorageInterface,
    ):
        """
        Initialize use case with dependencies.

        Args:
            synthesis_repository: Repository for synthesis persistence.
            audio_storage: Storage for output audio files.
        """
        self.synthesis_repo = synthesis_repository
        self.audio_storage = audio_storage

    async def execute(self, synthesis_id: str) -> bool:
        """
        Execute the delete synthesis use case.

        Deletes both the database record and the output audio file.

        Args:
            synthesis_id: The unique identifier of the synthesis to delete.

        Returns:
            True if deleted successfully, False if not found.
        """
        # Get synthesis to find the output file path
        synthesis = await self.synthesis_repo.get_by_id(synthesis_id)

        if synthesis is None:
            return False

        # Delete output audio file if it exists
        if synthesis.output_file_path:
            await self.audio_storage.delete(synthesis.output_file_path)

        # Delete database record
        return await self.synthesis_repo.delete(synthesis_id)
