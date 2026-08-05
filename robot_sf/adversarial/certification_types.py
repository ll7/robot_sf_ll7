"""Dependency-free certification result types (issue #6455 cycle break).

``CertificationStatus`` lives here so modules such as
``robot_sf.adversarial.config`` can reference the certification result type
without importing ``robot_sf.adversarial.certification``, which itself depends
on ``config`` for ``CandidateSpec``. ``robot_sf.adversarial.certification``
re-exports the class, so every existing public import path keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CertificationStatus:
    """Certification outcome for a generated candidate."""

    schema_version: str
    status: str
    reason: str
    details: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Return True when the candidate is certified valid."""
        return self.status == "passed"

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "reason": self.reason,
            "details": dict(self.details),
        }
