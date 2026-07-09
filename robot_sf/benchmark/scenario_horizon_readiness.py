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
from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256

VALID = "valid"
TABLE_REEXPORT_READY = "table_reexport_ready"
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
    claim_boundary: str = "benchmark_results"
    readiness_packet: str | None = None
    planner_rows: int = 0
    ppo_status: str | None = None
    snqi_contract_status: str | None = None
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

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
            "claim_boundary": self.claim_boundary,
            "readiness_packet": self.readiness_packet,
            "planner_rows": self.planner_rows,
            "ppo_status": self.ppo_status,
            "snqi_contract_status": self.snqi_contract_status,
            "blockers": list(self.blockers),
            "notes": list(self.notes),
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


def _load_summary_artifact(path: Path) -> dict[str, Any]:
    """Load structured campaign summary JSON required by narrow table-readiness packets.

    Returns:
        Parsed campaign summary object.
    """
    if path.suffix.lower() != ".json":
        raise ValueError("narrow table-readiness packets require campaign_summary.json input")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("campaign summary root is not object")
    return payload


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


def _load_packet(path: Path) -> dict[str, Any]:
    """Load a predeclared JSON readiness packet.

    Returns:
        Parsed readiness packet object.
    """
    packet = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        raise ValueError("readiness packet root is not object")
    return packet


def _matches_artifact_path(packet_path: Any, artifact_path: Path, artifact_str: str) -> bool:
    """Return whether packet path identifies the same artifact.

    Returns:
        ``True`` when packet path matches absolute or current-worktree-relative artifact path.
    """
    if packet_path == artifact_str:
        return True
    source_path = Path(str(packet_path))
    if source_path.is_absolute():
        return source_path == artifact_path
    try:
        return source_path == artifact_path.relative_to(Path.cwd())
    except ValueError:
        return False


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


def classify_scenario_horizon_table_reexport_readiness(  # noqa: C901, PLR0912, PLR0915
    artifact: str | Path,
    readiness_packet: str | Path,
) -> ScenarioHorizonReadiness:
    """Classify a predeclared narrow table re-export claim boundary.

    This intentionally does not promote the artifact to benchmark/Results evidence. It only
    proves the packet predeclared that SNQI is excluded from the claim and that the structured
    July 2026 re-export summary still satisfies row/provenance readiness.

    Returns:
        Narrow table re-export readiness verdict.
    """
    base = classify_scenario_horizon_readiness(artifact)
    artifact_path = Path(artifact)
    packet_path = Path(readiness_packet)
    artifact_str = str(artifact)
    packet_str = str(readiness_packet)
    claim_boundary = "scenario_horizon_table_reexport_only"

    if base.is_blocked:
        return ScenarioHorizonReadiness(
            status=BLOCKED,
            artifact=artifact_str,
            claim_boundary=claim_boundary,
            readiness_packet=packet_str,
            blockers=list(base.blockers),
        )
    if not packet_path.exists():
        return ScenarioHorizonReadiness(
            status=BLOCKED,
            artifact=artifact_str,
            claim_boundary=claim_boundary,
            readiness_packet=packet_str,
            planner_rows=base.planner_rows,
            ppo_status=base.ppo_status,
            snqi_contract_status=base.snqi_contract_status,
            blockers=[f"Readiness packet not found: {packet_str}"],
        )

    blockers: list[str] = []
    notes: list[str] = []
    try:
        packet = _load_packet(packet_path)
        summary = _load_summary_artifact(artifact_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return ScenarioHorizonReadiness(
            status=BLOCKED,
            artifact=artifact_str,
            claim_boundary=claim_boundary,
            readiness_packet=packet_str,
            planner_rows=base.planner_rows,
            ppo_status=base.ppo_status,
            snqi_contract_status=base.snqi_contract_status,
            blockers=[f"Could not load narrow readiness packet inputs: {exc}"],
        )

    if packet.get("claim_boundary") != claim_boundary:
        blockers.append(f"readiness packet claim_boundary must be {claim_boundary!r}")
    if packet.get("paper_facing") is not False:
        blockers.append("readiness packet must declare paper_facing=false")
    if packet.get("benchmark_results_claim") is not False:
        blockers.append("readiness packet must declare benchmark_results_claim=false")
    if packet.get("snqi_validity_claim") is not False:
        blockers.append("readiness packet must declare snqi_validity_claim=false")

    source = packet.get("source_artifact", {})
    if not isinstance(source, dict):
        blockers.append("readiness packet source_artifact must be object")
        source = {}
    if not _matches_artifact_path(source.get("path"), artifact_path, artifact_str):
        blockers.append(
            f"readiness packet source_artifact.path {source.get('path')!r} "
            f"does not match artifact {artifact_str!r}"
        )
    expected_sha = str(source.get("sha256", "")).strip()
    if not expected_sha:
        blockers.append("readiness packet source_artifact.sha256 missing")
    elif expected_sha != _sha256(artifact_path):
        blockers.append("readiness packet source_artifact.sha256 is stale")

    campaign = summary.get("campaign", {})
    rows = summary.get("planner_rows", [])
    if not isinstance(campaign, dict):
        blockers.append("campaign_summary.json campaign must be object")
        campaign = {}
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        blockers.append("campaign_summary.json planner_rows must be objects")
        rows = []

    required = packet.get("required_observations", {})
    if not isinstance(required, dict):
        blockers.append("readiness packet required_observations must be object")
        required = {}
    for key, expected in required.items():
        if key in {"ppo_status", "ppo_execution_mode", "ppo_learned_policy_contract_status"}:
            continue
        observed = campaign.get(key)
        if observed != expected:
            blockers.append(f"campaign.{key}={observed!r}, expected {expected!r}")

    ppo_row = next((row for row in rows if _planner_label(row) == "ppo"), None)
    if ppo_row is None:
        blockers.append("PPO row missing scenario-horizon artifact")
    else:
        ppo_status = classify_planner_row_status(_row_status(ppo_row))
        expected_status = required.get("ppo_status", "ok")
        if ppo_status != expected_status:
            blockers.append(f"PPO row status {ppo_status!r}, expected {expected_status!r}")
        expected_mode = required.get("ppo_execution_mode", "native")
        if ppo_row.get("execution_mode") != expected_mode:
            blockers.append(
                f"PPO execution_mode {ppo_row.get('execution_mode')!r}, expected {expected_mode!r}"
            )
        expected_contract = required.get("ppo_learned_policy_contract_status", "pass")
        if ppo_row.get("learned_policy_contract_status") != expected_contract:
            blockers.append(
                "PPO learned_policy_contract_status "
                f"{ppo_row.get('learned_policy_contract_status')!r}, "
                f"expected {expected_contract!r}"
            )

    for row in rows:
        normalized_status = classify_planner_row_status(_row_status(row))
        if normalized_status != "ok":
            blockers.append(
                f"Planner '{_planner_label(row)}' row status {normalized_status}, not table-ready"
            )

    snqi_exclusion = packet.get("snqi_exclusion", {})
    if "snqi_exclusion" not in packet:
        blockers.append("readiness packet snqi_exclusion missing")
    if not isinstance(snqi_exclusion, dict):
        blockers.append("readiness packet snqi_exclusion must be object")
        snqi_exclusion = {}
    if snqi_exclusion.get("predeclared") is not True:
        blockers.append("readiness packet must predeclare SNQI exclusion")
    if snqi_exclusion.get("excluded_from_claim") is not True:
        blockers.append("readiness packet must exclude SNQI from the narrow claim")
    if snqi_exclusion.get("observed_status") != base.snqi_contract_status:
        blockers.append(
            "readiness packet SNQI observed_status "
            f"{snqi_exclusion.get('observed_status')!r} does not match "
            f"{base.snqi_contract_status!r}"
        )
    if not str(snqi_exclusion.get("reason", "")).strip():
        blockers.append("readiness packet SNQI exclusion reason missing")
    else:
        notes.append(
            "SNQI remains excluded from this narrow table-readiness claim; "
            f"observed status is {base.snqi_contract_status!r}."
        )

    return ScenarioHorizonReadiness(
        status=TABLE_REEXPORT_READY if not blockers else DIAGNOSTIC_ONLY,
        artifact=artifact_str,
        claim_boundary=claim_boundary,
        readiness_packet=packet_str,
        planner_rows=base.planner_rows,
        ppo_status=base.ppo_status,
        snqi_contract_status=base.snqi_contract_status,
        blockers=blockers,
        notes=notes,
    )


__all__ = [
    "BLOCKED",
    "DIAGNOSTIC_ONLY",
    "TABLE_REEXPORT_READY",
    "VALID",
    "ScenarioHorizonReadiness",
    "classify_scenario_horizon_readiness",
    "classify_scenario_horizon_table_reexport_readiness",
]
