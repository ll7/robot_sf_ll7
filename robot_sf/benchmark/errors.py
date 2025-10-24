"""Custom exception types for the benchmark package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


class AggregationMetadataError(ValueError):
    """Raised when algorithm metadata required for aggregation is missing or invalid."""

    def __init__(
        self,
        message: str,
        *,
        episode_id: str | None = None,
        missing_fields: Iterable[str] | None = None,
        advice: str | None = None,
    ) -> None:
        super().__init__(message)
        self.episode_id = episode_id
        self.missing_fields: tuple[str, ...] = tuple(missing_fields or ())
        self.advice = advice

    def to_dict(self) -> dict[str, object]:
        """Structured representation suitable for logging or JSON responses."""

        payload: dict[str, object] = {
            "message": str(self),
        }
        if self.episode_id is not None:
            payload["episode_id"] = self.episode_id
        if self.missing_fields:
            payload["missing_fields"] = list(self.missing_fields)
        if self.advice is not None:
            payload["advice"] = self.advice
        return payload


__all__ = ["AggregationMetadataError"]
