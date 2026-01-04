"""File storage interface - Application layer contract"""
from abc import ABC, abstractmethod


class FileStorageInterface(ABC):
    """
    Abstract interface for file storage operations.

    This interface defines the contract for file storage that
    infrastructure layer must implement.
    """

    @abstractmethod
    async def save(self, file_content: bytes, file_id: str, filename: str) -> str:
        """
        Save file to storage.

        Args:
            file_content: Binary content of the file
            file_id: Unique identifier for the file
            filename: Original filename (used for extension)

        Returns:
            str: Path or identifier where file was saved

        Raises:
            StorageException: If save operation fails
        """
        pass

    @abstractmethod
    async def get(self, file_path: str) -> bytes:
        """
        Retrieve file from storage.

        Args:
            file_path: Path or identifier of the file

        Returns:
            bytes: File content

        Raises:
            StorageException: If file not found or retrieval fails
        """
        pass

    @abstractmethod
    async def delete(self, file_path: str) -> bool:
        """
        Delete file from storage.

        Args:
            file_path: Path or identifier of the file

        Returns:
            bool: True if deleted, False if not found

        Raises:
            StorageException: If deletion fails
        """
        pass

    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """
        Check if file exists in storage.

        Args:
            file_path: Path or identifier of the file

        Returns:
            bool: True if file exists, False otherwise
        """
        pass
