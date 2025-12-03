"""Metrics-related type definitions (T042)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MetricsBundle:
    """View/accessor for episode metrics.

    This provides a typed interface for accessing computed metrics
    from episode records, with optional validation and defaults.
    """

    values: dict[str, float]

    def get(self, name: str, default: float | None = None) -> float | None:
        """Get metric value by name with optional default.

        Returns:
            Metric value if present, otherwise the default.
        """
        return self.values.get(name, default)

    def require(self, name: str) -> float:
        """Get required metric value, raise KeyError if missing.

        Returns:
            The metric value.
        """
        if name not in self.values:
            raise KeyError(f"Required metric '{name}' not found")
        return self.values[name]

    def has(self, name: str) -> bool:
        """Check if metric is present.

        Returns:
            True if the metric exists in values, False otherwise.
        """
        return name in self.values

    def to_dict(self) -> dict[str, float]:
        """Convert to plain dict for serialization.

        Returns:
            Dictionary copy of metric values.
        """
        return dict(self.values)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> MetricsBundle:
        """Create from plain dict.

        Returns:
            New MetricsBundle instance with the provided values.
        """
        return cls(values=data)


__all__ = ["MetricsBundle"]
