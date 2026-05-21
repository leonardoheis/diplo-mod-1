"""Shared base for all domain value objects."""

from pydantic import BaseModel, ConfigDict


class BaseDomainModel(BaseModel):
    """Immutable Pydantic base for all domain entities.

    ``frozen=True`` enforces value-object semantics: instances are hashable
    and cannot be mutated after construction.
    """

    model_config = ConfigDict(frozen=True)
