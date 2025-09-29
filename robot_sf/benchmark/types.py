"""Core benchmark data structures (Phase 3.4 tasks T040, T041, T044).

These dataclasses provide typed containers for scenario specifications,
episode records, and resume manifests. They are deliberately minimal and
avoid introducing runtime dependencies (pure typing + stdlib) so they can
be imported in lightweight tooling (schema generation, hashing, etc.).

Serialization: writing to JSONL will typically convert instances to
``dict`` via ``dataclasses.asdict`` or explicit ``to_dict`` helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import (
    UTC,  # type: ignore[attr-defined]
    datetime,
)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(slots=True)
class ScenarioSpec:
    """Scenario specification (single row from scenario matrix).

    Required fields align with `scenario-matrix.schema.v1.json`.
    Additional algorithm-specific configuration can be passed via
    the optional `algo_config_path` or embedded metadata dict.
    """

    id: str
    algo: str
    map: str
    episodes: int
    seed: int
    notes: str | None = None
    algo_config_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:  # stable conversion
        return asdict(self)


@dataclass(slots=True)
class MetricsBundle:
    """Container for computed metric values.

    Internally just wraps a mapping but gives a semantic type for future
    validation or access helpers (e.g., enforcing presence of required keys).
    """

    values: dict[str, float]

    def get(self, name: str, default: float | None = None) -> float | None:
        return self.values.get(name, default)

    def to_dict(self) -> dict[str, float]:
        return dict(self.values)


@dataclass(slots=True)
class EpisodeRecord:
    """High-level episode record suitable for JSONL persistence.

    The `raw` field can contain implementation-specific extras (timing, identity
    materials, debug traces) that are not part of the stable metrics payload.
    """

    version: str
    episode_id: str
    scenario_id: str
    seed: int
    metrics: MetricsBundle
    algo: str | None = None
    horizon: int | None = None
    timing: dict[str, float] | None = None
    tags: list[str] | None = None
    identity: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # flatten metrics bundle for JSON writing
        d["metrics"] = self.metrics.to_dict()
        return d


@dataclass(slots=True)
class SNQIWeights:
    """Weight file content for SNQI computation (subset for early phases)."""

    version: str
    weights: Mapping[str, float]
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "weights": dict(self.weights), "meta": self.meta or {}}


@dataclass(slots=True)
class ResumeManifest:
    """Resume manifest describing completed episode ids (Phase 3.6/3.3 link)."""

    version: str
    episodes: list[str]
    meta: dict[str, Any] | None = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC)
        .astimezone(UTC)
        .replace(microsecond=0)
        .isoformat(),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "episodes": list(self.episodes),
            "meta": self.meta or {},
            "generated_at": self.generated_at,
        }


__all__ = [
    "EpisodeRecord",
    "MetricsBundle",
    "ResumeManifest",
    "SNQIWeights",
    "ScenarioSpec",
]
