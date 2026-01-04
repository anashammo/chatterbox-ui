"""Domain layer exceptions"""


class DomainException(Exception):
    """Base exception for domain layer errors"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ValidationException(DomainException):
    """Exception raised for validation errors in domain entities"""
    pass


class RepositoryException(DomainException):
    """Exception raised for repository operation failures"""
    pass


class SynthesisException(DomainException):
    """Exception raised for TTS synthesis-specific errors"""
    pass


class VoiceReferenceException(DomainException):
    """Exception raised for voice reference-specific errors"""
    pass


class ServiceException(DomainException):
    """Exception raised for service operation failures"""
    pass
