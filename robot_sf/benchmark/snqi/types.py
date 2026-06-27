"""SNQI-related type definitions (T043)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Provenance metadata fields that must be present (and string-typed) for a
# weights config to be considered well-formed. These mirror the dataclass
# fields written by ``SNQIWeights.to_dict`` / the recompute CLI.
_REQUIRED_META_FIELDS: tuple[str, ...] = (
    "weights_version",
    "created_at",
    "git_sha",
    "baseline_stats_path",
    "baseline_stats_hash",
    "normalization_strategy",
)


def _validate_weight_values(weights: object, *, source: str) -> dict[str, float]:
    """Validate a weights mapping holds finite, non-negative numeric values.

    The ``SNQIWeights.weights`` container is a generic ``component -> weight``
    mapping, so this helper deliberately does not require a fixed key set (the
    canonical ``WEIGHT_NAMES`` completeness check lives in
    ``robot_sf.benchmark.snqi.weights_validation`` for the recompute path). It
    only enforces that every supplied value is a usable weight, casting it to
    ``float`` so downstream consumers never receive a malformed value.

    Args:
        weights: Candidate weights mapping loaded from untrusted JSON/config.
        source: Human-readable origin (file path or ``"<dict>"``) used in error
            messages so failures are diagnosable.

    Returns:
        A new ``dict[str, float]`` with all values validated and cast to float.

    Raises:
        ValueError: If ``weights`` is not a mapping or holds a value that is not
            a finite, non-negative number.
    """
    if not isinstance(weights, Mapping):
        raise ValueError(
            f"SNQIWeights 'weights' must be a mapping, got {type(weights).__name__} "
            f"(source: {source})"
        )
    validated: dict[str, float] = {}
    for name, value in weights.items():
        # Reject bools explicitly: bool is an int subclass and float(True) == 1.0
        # would silently coerce a clearly malformed value.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"SNQIWeights weight '{name}' is not numeric: {value!r} (source: {source})"
            )
        fv = float(value)
        if not math.isfinite(fv) or fv < 0:
            raise ValueError(
                f"SNQIWeights weight '{name}' must be finite and non-negative, "
                f"got {fv} (source: {source})"
            )
        validated[name] = fv
    return validated


@dataclass(slots=True)
class SNQIWeights:
    """SNQI weights configuration.

    Contains the weights for computing the Social Navigation Quality Index
    along with provenance metadata for reproducibility.
    """

    weights_version: str
    created_at: str
    git_sha: str
    baseline_stats_path: str
    baseline_stats_hash: str
    normalization_strategy: str
    bootstrap_params: dict[str, Any] = field(default_factory=dict)
    components: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)

    def get_weight(self, component: str, default: float = 0.0) -> float:
        """Get weight for a component with default.

        Returns:
            Weight value for the component, or default if not found.
        """
        return self.weights.get(component, default)

    def has_component(self, component: str) -> bool:
        """Check if component has a weight defined.

        Returns:
            True if component exists in weights dictionary.
        """
        return component in self.weights

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization.

        Returns:
            Dictionary representation of the SNQIWeights object.
        """
        return {
            "weights_version": self.weights_version,
            "created_at": self.created_at,
            "git_sha": self.git_sha,
            "baseline_stats_path": self.baseline_stats_path,
            "baseline_stats_hash": self.baseline_stats_hash,
            "normalization_strategy": self.normalization_strategy,
            "bootstrap_params": self.bootstrap_params,
            "components": self.components,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: str = "<dict>") -> SNQIWeights:
        """Create from dict (e.g., loaded from JSON), failing closed on bad input.

        Required provenance metadata must be present and string-typed, and every
        weight value must be a finite, non-negative number. Malformed input
        raises a descriptive ``ValueError`` instead of a bare ``KeyError`` or a
        silently-coerced value, so callers can surface an actionable diagnostic.

        Args:
            data: Parsed mapping (typically from JSON).
            source: Human-readable origin (file path or ``"<dict>"``) included in
                error messages for diagnosability.

        Returns:
            SNQIWeights instance reconstructed from dictionary data.

        Raises:
            ValueError: If ``data`` is not a mapping, a required metadata field is
                missing or not a string, or any optional field has the wrong type
                or an invalid weight value.
        """
        if not isinstance(data, Mapping):
            raise ValueError(
                f"SNQIWeights config must be a JSON object/mapping, "
                f"got {type(data).__name__} (source: {source})"
            )

        missing = [key for key in _REQUIRED_META_FIELDS if key not in data]
        if missing:
            raise ValueError(
                f"SNQIWeights config is missing required field(s): {missing} (source: {source})"
            )
        for key in _REQUIRED_META_FIELDS:
            value = data[key]
            if not isinstance(value, str):
                raise ValueError(
                    f"SNQIWeights field '{key}' must be a string, "
                    f"got {type(value).__name__} (source: {source})"
                )

        bootstrap_params = data.get("bootstrap_params", {})
        if not isinstance(bootstrap_params, Mapping):
            raise ValueError(
                f"SNQIWeights 'bootstrap_params' must be a mapping, "
                f"got {type(bootstrap_params).__name__} (source: {source})"
            )

        components = data.get("components", [])
        if not isinstance(components, list):
            raise ValueError(
                f"SNQIWeights 'components' must be a list, "
                f"got {type(components).__name__} (source: {source})"
            )

        weights = _validate_weight_values(data.get("weights", {}), source=source)

        return cls(
            weights_version=data["weights_version"],
            created_at=data["created_at"],
            git_sha=data["git_sha"],
            baseline_stats_path=data["baseline_stats_path"],
            baseline_stats_hash=data["baseline_stats_hash"],
            normalization_strategy=data["normalization_strategy"],
            bootstrap_params=dict(bootstrap_params),
            components=list(components),
            weights=weights,
        )

    def save(self, path: Path | str) -> None:
        """Save weights to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> SNQIWeights:
        """Load weights from JSON file, failing closed on malformed content.

        Returns:
            SNQIWeights instance loaded from the JSON file.

        Raises:
            ValueError: If the file is not valid JSON or the parsed config fails
                the :meth:`from_dict` validation contract. The originating path is
                included in the error message for diagnosability.
        """
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"SNQIWeights config is not valid JSON (source: {path}): {e}"
                ) from e
        return cls.from_dict(data, source=str(path))


__all__ = ["SNQIWeights"]
