"""SNQI-related type definitions (T043)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


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
    bootstrap_params: Dict[str, Any] = field(default_factory=dict)
    components: list[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)

    def get_weight(self, component: str, default: float = 0.0) -> float:
        """Get weight for a component with default."""
        return self.weights.get(component, default)

    def has_component(self, component: str) -> bool:
        """Check if component has a weight defined."""
        return component in self.weights

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
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
    def from_dict(cls, data: Dict[str, Any]) -> SNQIWeights:
        """Create from dict (e.g., loaded from JSON)."""
        return cls(
            weights_version=data["weights_version"],
            created_at=data["created_at"],
            git_sha=data["git_sha"],
            baseline_stats_path=data["baseline_stats_path"],
            baseline_stats_hash=data["baseline_stats_hash"],
            normalization_strategy=data["normalization_strategy"],
            bootstrap_params=data.get("bootstrap_params", {}),
            components=data.get("components", []),
            weights=data.get("weights", {}),
        )

    def save(self, path: Path | str) -> None:
        """Save weights to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> SNQIWeights:
        """Load weights from JSON file."""
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


__all__ = ["SNQIWeights"]
