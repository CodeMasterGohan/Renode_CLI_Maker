"""
Custom exceptions for the Renode Peripheral Generator CLI.
"""


class RenodeGeneratorError(Exception):
    """Base exception for all Renode generator errors."""
    pass


class ConfigError(RenodeGeneratorError):
    """Raised when there's a configuration-related error."""
    pass


class LLMClientError(RenodeGeneratorError):
    """Raised when there's an error with the LLM client."""
    pass


class MilvusClientError(RenodeGeneratorError):
    """Raised when there's an error with the Milvus client."""
    pass


class GenerationError(RenodeGeneratorError):
    """Raised when there's an error during peripheral generation."""
    pass


class ValidationError(RenodeGeneratorError):
    """Raised when validation fails."""
    pass 