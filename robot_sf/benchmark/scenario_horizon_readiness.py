"""Fail-closed readiness classification for scenario-horizon Results evidence.

This module answers one diagnostic-only question for the scenario-horizon
campaign tables (the dissertation tables re-exported by PR #3263 / issue #3203):

    *Is the re-exported scenario-horizon campaign evidence valid benchmark
    evidence, diagnostic-only, or blocked by a missing artifact?*

It does **not** rerun any campaign and does **not** promote evidence. It reads a
campaign-table artifact (the re-exported Markdown ``campaign_table.md`` or an
equivalent campaign-summary JSON with ``planner_rows``) and classifies whether
the evidence is allowed to support benchmark/Results wording.

The classification is fail-closed (see ``docs/context/issue_691_benchmark_fallback_policy.md``):

- a missing or unparseable artifact is ``blocked`` (never silently "valid");
- any planner row that did not run as benchmark-success -- in particular the PPO
  ``partial-failure`` recorded for issue #3266 -- caps the evidence at
  ``diagnostic_only``;
- the SNQI contract status must be explicitly asserted as ``pass`` in the
  artifact. When the artifact carries no SNQI contract status (the re-exported
  Markdown tables do not), the SNQI caveat is treated as unresolved and the
  evidence is capped at ``diagnostic_only`` rather than assumed valid.

Per-row status normalization reuses the canonical
``robot_sf.benchmark.fallback_policy.classify_planner_row_status`` owner instead
of re-implementing the status taxonomy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.benchmark.fallback_policy import classify_planner_row_status

#: Overall readiness verdicts, fail-closed from strongest to weakest.
VALID = "valid"
DIAGNOSTIC_ONLY = "diagnostic_only"
BLOCKED = "blocked"

#: Row-status column names that may carry the per-planner execution status.
_ROW_STATUS_KEYS = ("status", "readiness_status", "availability_status")
#: Column names that may carry the SNQI contract pass/warn/fail status.
_SNQI_CONTRACT_KEYS = ("snqi_contract_status", "snqi_status", "snqi_contract")
#: SNQI contract statuses other than these block a ``valid`` verdict.
_SNQI_PASS_STATUSES = {"pass", "ok", "passed"}


@dataclass(frozen=True)
class ScenarioHorizonReadiness:
    """Diagnostic-only readiness verdict for a scenario-horizon evidence artifact.

    Attributes:
        status: One of ``valid``, ``diagnostic_only``, or ``blocked``.
        artifact: Repository-relative or absolute path to the inspected artifact.
        planner_rows: Number of planner rows parsed from the artifact.
        ppo_status: Normalized PPO row outcome, or ``None`` when no PPO row exists.
        snqi_contract_status: SNQI contract status asserted by the artifact, or
            ``None`` when the artifact does not assert one.
        blockers: Human-readable reasons the evidence is not ``valid``.
    """

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
        """Return whether the artifact is blocked by a missing/unreadable source."""
        return self.status == BLOCKED

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the readiness verdict.

        Returns:
            Mapping with the verdict, inspected artifact, and blocker reasons.
        """
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
    """Parse a single pipe-delimited Markdown table into row dictionaries.

    Only the first table found in the text is parsed. The header row supplies the
    column keys; the separator row (cells of only ``-`` and ``:``) is skipped.

    Returns:
        One dictionary per data row, keyed by lowercased header column name.
        Returns an empty list when no pipe table is present.
    """
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            # End the table once a non-table line follows the header.
            if header is not None:
                break
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if header is None:
            header = [cell.lower() for cell in cells]
            continue
        if all(set(cell) <= {"-", ":"} and cell for cell in cells):
            continue
        # Tolerate ragged rows: zip stops at the shorter of header/cells.
        rows.append(dict(zip(header, cells, strict=False)))
    return rows


def _rows_from_artifact(path: Path) -> list[dict[str, Any]]:
    """Read planner rows from a Markdown table or a campaign-summary JSON.

    JSON artifacts are expected to expose a ``planner_rows`` list. Markdown
    artifacts are parsed as a single pipe-delimited table.

    Returns:
        Planner-row dictionaries.

    Raises:
        ValueError: When the artifact cannot be parsed into planner rows.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        rows = payload.get("planner_rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("campaign-summary JSON must contain a 'planner_rows' list")
        return [row for row in rows if isinstance(row, dict)]
    rows = _parse_markdown_table(text)
    if not rows:
        raise ValueError("no Markdown campaign table found in artifact")
    return rows


def _row_status(row: dict[str, Any]) -> str:
    """Return the most specific per-row execution status available.

    Returns:
        The first non-empty value among the known status columns, else "".
    """
    for key in _ROW_STATUS_KEYS:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _planner_label(row: dict[str, Any]) -> str:
    """Return a stable planner identifier for a row.

    Returns:
        The planner key/algo label, or "<unknown>" when none is present.
    """
    for key in ("planner_key", "algo", "planner", "planner_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return "<unknown>"


def _snqi_contract_status(rows: list[dict[str, Any]]) -> str | None:
    """Extract the SNQI contract status asserted anywhere in the rows.

    Returns:
        The first non-empty SNQI contract status found, else ``None``.
    """
    for row in rows:
        for key in _SNQI_CONTRACT_KEYS:
            value = str(row.get(key, "")).strip()
            if value:
                return value.lower()
    return None


def classify_scenario_horizon_readiness(artifact: str | Path) -> ScenarioHorizonReadiness:
    """Classify whether a scenario-horizon evidence artifact is benchmark-valid.

    The verdict is fail-closed: a missing/unparseable artifact is ``blocked``;
    any non-benchmark-success planner row or an unresolved SNQI contract caveat
    caps the verdict at ``diagnostic_only``.

    Args:
        artifact: Path to a re-exported ``campaign_table.md`` or a campaign-summary
            JSON exposing ``planner_rows``.

    Returns:
        A :class:`ScenarioHorizonReadiness` verdict.
    """
    path = Path(artifact)
    artifact_str = str(artifact)

    if not path.exists():
        return ScenarioHorizonReadiness(
            status=BLOCKED,
            artifact=artifact_str,
            blockers=[f"artifact not found: {artifact_str}"],
        )
    if not path.is_file():
        return ScenarioHorizonReadiness(
            status=BLOCKED,
            artifact=artifact_str,
            blockers=[f"artifact is not a file: {artifact_str}"],
        )

    try:
        rows = _rows_from_artifact(path)
    except (ValueError, OSError) as exc:
        return ScenarioHorizonReadiness(
            status=BLOCKED,
            artifact=artifact_str,
            blockers=[f"could not parse artifact: {exc}"],
        )
    if not rows:
        return ScenarioHorizonReadiness(
            status=BLOCKED,
            artifact=artifact_str,
            blockers=["artifact contains no planner rows"],
        )

    blockers: list[str] = []
    ppo_status: str | None = None
    for row in rows:
        label = _planner_label(row)
        outcome = classify_planner_row_status(_row_status(row))
        if label.lower() == "ppo":
            ppo_status = outcome
        if outcome == "unexpected_failure":
            reason = str(row.get("most_likely_failure_reason", "")).strip()
            detail = f" ({reason})" if reason else ""
            blockers.append(f"planner '{label}' did not run as benchmark-success{detail}")

    snqi_contract_status = _snqi_contract_status(rows)
    if snqi_contract_status is None:
        blockers.append(
            "SNQI contract status not asserted by artifact; caveat unresolved "
            "(re-exported tables carry no SNQI pass/warn/fail status)"
        )
    elif snqi_contract_status not in _SNQI_PASS_STATUSES:
        blockers.append(f"SNQI contract status is '{snqi_contract_status}', not pass")

    status = DIAGNOSTIC_ONLY if blockers else VALID
    return ScenarioHorizonReadiness(
        status=status,
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
