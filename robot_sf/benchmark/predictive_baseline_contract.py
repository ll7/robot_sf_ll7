"""Typed provenance contract for predictive planner diagnostic comparisons.

The contract keeps method identity, configuration, action/observation
interfaces, and evidence status together. It is intentionally suitable for a
small deterministic smoke; it does not make a benchmark, safety, or source-
paper reproduction claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

PREDICTIVE_BASELINE_DIAGNOSTIC_SCHEMA = "predictive_baseline_diagnostic.v1"
PREDICTIVE_BASELINE_EVIDENCE_TIER = "diagnostic-only"
PREDICTIVE_BASELINE_CLAIM_BOUNDARY = (
    "diagnostic-only same-seed planner smoke; no simulator, benchmark, safety, "
    "source-paper reproduction, or transfer claim"
)

EvidenceStatus = Literal["smoke_pass", "unavailable", "failed"]


def canonical_sha256(payload: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PlannerMethodCard:
    """Reproducibility and evidence card for one comparator lane."""

    method_id: str
    display_name: str
    planner_family: str
    adapter_name: str
    observation_contract: str
    action_contract: str
    source_reference: str
    license_status: str
    implementation_mode: Literal["native", "adapter", "unavailable"]
    benchmark_status: Literal["diagnostic_only", "not_available"]
    fallback_policy: str
    claim_boundary: str
    config: dict[str, Any]
    formula: str = ""
    input_visibility: str = ""
    missing_input_policy: str = ""

    def __post_init__(self) -> None:
        """Reject incomplete method identity or unsupported evidence labels."""

        for name in (
            "method_id",
            "display_name",
            "planner_family",
            "adapter_name",
            "observation_contract",
            "action_contract",
            "source_reference",
            "license_status",
            "fallback_policy",
            "claim_boundary",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"method card {name} must be a non-empty string")
        if not isinstance(self.config, dict):
            raise ValueError("method card config must be a mapping")

    @property
    def config_digest(self) -> str:
        """Return the stable digest of the resolved method configuration."""

        return canonical_sha256(self.config)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable method card with its digest."""

        payload = asdict(self)
        payload["config_digest"] = self.config_digest
        return payload


@dataclass(frozen=True, slots=True)
class PlannerSmokeRecord:
    """One deterministic smoke observation for a method card."""

    method_id: str
    status: EvidenceStatus
    command: tuple[float, float] | None
    repeat_command: tuple[float, float] | None
    deterministic: bool
    diagnostics: dict[str, Any]
    unavailable_metrics: dict[str, str]
    runtime_ms: float | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate smoke status, finite actions, and explicit metric gaps."""

        if not self.method_id.strip():
            raise ValueError("smoke method_id must be non-empty")
        if self.status == "smoke_pass" and self.command is None:
            raise ValueError("smoke_pass records require a command")
        for command_name in ("command", "repeat_command"):
            command = getattr(self, command_name)
            if command is not None:
                if len(command) != 2 or not all(map(_is_finite_float, command)):
                    raise ValueError(f"{command_name} must contain two finite values")
        if not isinstance(self.deterministic, bool):
            raise ValueError("smoke deterministic must be boolean")
        if not isinstance(self.diagnostics, dict) or not isinstance(self.unavailable_metrics, dict):
            raise ValueError("smoke diagnostics and unavailable_metrics must be mappings")
        if self.runtime_ms is not None and (
            not _is_finite_float(self.runtime_ms) or float(self.runtime_ms) < 0.0
        ):
            raise ValueError("smoke runtime_ms must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable smoke record."""

        payload = asdict(self)
        for key in ("command", "repeat_command"):
            if payload[key] is not None:
                payload[key] = list(payload[key])
        return payload


def _is_finite_float(value: Any) -> bool:
    """Return whether a value can be represented as a finite float."""

    try:
        return bool(__import__("math").isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def build_predictive_baseline_report(
    *,
    config: dict[str, Any],
    seed: int,
    scenario_id: str,
    method_cards: tuple[PlannerMethodCard, ...],
    smoke_records: tuple[PlannerSmokeRecord, ...],
) -> dict[str, Any]:
    """Build the fail-closed diagnostic report envelope.

    Returns:
        Schema-ready diagnostic report mapping.
    """

    if not method_cards:
        raise ValueError("predictive baseline report requires method cards")
    method_ids = {card.method_id for card in method_cards}
    if len(method_ids) != len(method_cards):
        raise ValueError("predictive baseline method IDs must be unique")
    if any(record.method_id not in method_ids for record in smoke_records):
        raise ValueError("smoke record references an unknown method ID")
    unavailable_metrics = {
        "success": "no simulator rollout",
        "collision": "no simulator rollout",
        "near_miss": "no simulator rollout",
        "timeout": "no simulator rollout",
        "path_efficiency": "no route rollout",
        "pedestrian_disruption": "no simulator truth",
        "minimum_distance": "no simulator truth",
        "action_smoothness": "one-step smoke only",
    }
    return {
        "schema_version": PREDICTIVE_BASELINE_DIAGNOSTIC_SCHEMA,
        "evidence_tier": PREDICTIVE_BASELINE_EVIDENCE_TIER,
        "claim_boundary": PREDICTIVE_BASELINE_CLAIM_BOUNDARY,
        "config": config,
        "config_digest": canonical_sha256(config),
        "seed_manifest": {"scenario_id": scenario_id, "seed": int(seed)},
        "pairing": {
            "same_scenario_and_seed": True,
            "method_ids": sorted(method_ids),
            "reference_method_id": "mppi_social_reference_v1",
        },
        "simulator_executed": False,
        "benchmark_evidence": False,
        "campaign_approval_required": True,
        "source_transfer_claim": False,
        "methods": [card.to_dict() for card in method_cards],
        "smoke_records": [record.to_dict() for record in smoke_records],
        "unavailable_metrics": unavailable_metrics,
        "limitations": [
            "The PGIF-style cost is an explicit Robot SF adaptation, not an exact source reproduction.",
            "The constrained-MPC lane composes existing Robot SF NMPC and CBF primitives; it does not copy external code.",
            "One-step fixture commands do not establish navigation, safety, runtime, or ranking claims.",
        ],
    }


__all__ = [
    "PREDICTIVE_BASELINE_CLAIM_BOUNDARY",
    "PREDICTIVE_BASELINE_DIAGNOSTIC_SCHEMA",
    "PREDICTIVE_BASELINE_EVIDENCE_TIER",
    "PlannerMethodCard",
    "PlannerSmokeRecord",
    "build_predictive_baseline_report",
    "canonical_sha256",
]
