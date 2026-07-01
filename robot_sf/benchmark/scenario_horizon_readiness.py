"""Fail-closed readiness classification for scenario-horizon Results evidence.

The classifier answers one narrow question for re-exported scenario-horizon campaign artifacts:
can this artifact support benchmark/Results wording, or must it remain diagnostic/blocked?
It does not rerun campaigns or promote evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.benchmark.fallback_policy import classify_planner_row_status

VALID = "valid"
DIAGNOSTIC_ONLY = "diagnostic_only"
BLOCKED = "blocked"

_ROW_STATUS_KEYS = (
    "status",
    "row_status",
    "availability_status",
    "benchmark_status",
)
_SNQI_CONTRACT_KEYS = (
    "snqi_contract_status",
    "snqi_status",
    "snqi_contract",
)
_SNQI_PASS_STATUSES = {"pass", "passed", "ok"}


@dataclass(frozen=True)
class ScenarioHorizonReadiness:
    """Scenario-horizon evidence readiness verdict."""

    status: str
    artifact: str
    planner_rows: int = 0
    ppo_status: str | None = None
    snqi_contract_status: str | None = None
    blockers: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return whether the artifact is valid benchmark evidence."""
        return self.status == VALID

    @property
    def is_blocked(self) -> bool:
        """Return whether the artifact is blocked by missing or unreadable input."""
        return self.status == BLOCKED

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable verdict payload."""
        return {
            "schema_version": "scenario_horizon_readiness.v1",
            "status": self.status,
            "artifact": self.artifact,
            "planner_rows": self.planner_rows,
            "ppo_status": self.ppo_status,
            "snqi_contract_status": self.snqi_contract_status,
            "blockers": list(self.blockers),
        }


def _parse_markdown_table(text: str) -> list[dict[str, str]]:
    """Parse the first pipe-delimited Markdown table in ``text``.

    Returns:
        Parsed row dictionaries keyed by lowercased column name.
    """
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if header is not None:
                break
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if header is None:
            header = [cell.lower() for cell in cells]
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        rows.append(dict(zip(header, cells, strict=False)))
    return rows


def _rows_from_json(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Return planner rows plus campaign-level SNQI status from JSON payload."""
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact root is not an object")

    campaign = payload.get("campaign")
    campaign_snqi = _snqi_contract_status_from_mapping(
        campaign if isinstance(campaign, dict) else {}
    )
    top_level_snqi = _snqi_contract_status_from_mapping(payload)
    snqi_status = campaign_snqi or top_level_snqi

    rows = payload.get("planner_rows")
    if rows is None:
        rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("JSON artifact does not expose planner_rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSON artifact planner_rows must be objects")
    return rows, snqi_status


def _load_artifact(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Load planner rows and campaign-level SNQI status from an artifact path.

    Returns:
        Planner rows and optional campaign-level SNQI status.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return _rows_from_json(json.loads(text))

    rows = _parse_markdown_table(text)
    if not rows:
        raise ValueError("no Markdown campaign table found in artifact")
    return rows, None


def _row_status(row: dict[str, Any]) -> str:
    """Return the most specific per-row execution status available."""
    for key in _ROW_STATUS_KEYS:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _planner_label(row: dict[str, Any]) -> str:
    """Return a stable planner identifier for one row."""
    for key in ("planner_key", "algo", "planner", "planner_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return "<unknown>"


def _snqi_contract_status_from_mapping(mapping: dict[str, Any]) -> str | None:
    """Extract an explicitly asserted SNQI contract status from one mapping.

    Returns:
        Lowercase SNQI status, or ``None`` when absent.
    """
    for key in _SNQI_CONTRACT_KEYS:
        value = mapping.get(key)
        if isinstance(value, dict):
            value = value.get("status")
        status = str(value or "").strip().lower()
        if status:
            return status
    return None


def _snqi_contract_status(rows: list[dict[str, Any]], fallback: str | None) -> str | None:
    """Extract SNQI contract status from campaign metadata or row fields.

    Returns:
        Lowercase SNQI status, or ``None`` when absent.
    """
    if fallback:
        return fallback
    for row in rows:
        status = _snqi_contract_status_from_mapping(row)
        if status:
            return status
    return None


def classify_scenario_horizon_readiness(artifact: str | Path) -> ScenarioHorizonReadiness:
    """Classify whether a scenario-horizon artifact is benchmark-valid.

    Missing or unparseable artifacts are ``blocked``. Any non-success planner row or unresolved
    SNQI contract caveat caps the verdict at ``diagnostic_only``.

    Returns:
        Scenario-horizon readiness verdict.
    """
    path = Path(artifact)
    artifact_str = str(artifact)
    if not path.exists():
        return ScenarioHorizonReadiness(
            status=BLOCKED,
            artifact=artifact_str,
            blockers=[f"Artifact not found: {artifact_str}"],
        )

    try:
        rows, campaign_snqi_status = _load_artifact(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return ScenarioHorizonReadiness(
            status=BLOCKED,
            artifact=artifact_str,
            blockers=[f"Could not parse planner rows from campaign table: {exc}"],
        )

    if not rows:
        return ScenarioHorizonReadiness(
            status=BLOCKED,
            artifact=artifact_str,
            blockers=["Artifact contains no planner rows"],
        )

    blockers: list[str] = []
    ppo_status: str | None = None
    saw_ppo = False
    for row in rows:
        planner = _planner_label(row)
        normalized_status = classify_planner_row_status(_row_status(row))
        if planner == "ppo":
            saw_ppo = True
            ppo_status = normalized_status
        if normalized_status != "ok":
            reason = str(
                row.get("most_likely_failure_reason") or row.get("failure_reason") or ""
            ).strip()
            detail = f": {reason}" if reason else ""
            blockers.append(
                f"Planner '{planner}' row status is {normalized_status}, "
                f"not benchmark success{detail}"
            )

    if not saw_ppo:
        blockers.append("PPO row missing from scenario-horizon artifact")

    snqi_contract_status = _snqi_contract_status(rows, campaign_snqi_status)
    if snqi_contract_status is None:
        blockers.append(
            "SNQI contract status not asserted by artifact; caveat unresolved "
            "(re-exported tables carry no SNQI pass/warn/fail status)"
        )
    elif snqi_contract_status not in _SNQI_PASS_STATUSES:
        blockers.append(f"SNQI contract status '{snqi_contract_status}', not pass")

    return ScenarioHorizonReadiness(
        status=VALID if not blockers else DIAGNOSTIC_ONLY,
        artifact=artifact_str,
        planner_rows=len(rows),
        ppo_status=ppo_status,
        snqi_contract_status=snqi_contract_status,
        blockers=blockers,
    )


__all__ = [
    "BLOCKED",
    "DIAGNOSTIC_ONLY",
    "VALID",
    "ScenarioHorizonReadiness",
    "classify_scenario_horizon_readiness",
]
