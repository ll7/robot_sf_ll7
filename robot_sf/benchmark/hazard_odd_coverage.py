"""Campaign-level hazard and ODD coverage rollup utilities."""

# ruff: noqa: DOC201

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from matplotlib.figure import Figure

from robot_sf.benchmark.artifact_catalog import sha256_file
from robot_sf.benchmark.hazard_traceability import (
    HazardTraceability,
    load_hazard_traceability,
)
from robot_sf.benchmark.odd_contract import OddContract, load_odd_contracts
from robot_sf.benchmark.scenario_contract import ScenarioContract, load_scenario_contracts
from robot_sf.benchmark.stress_uncertainty_coverage import (
    StressUncertaintyCoverageError,
    load_stress_uncertainty_coverage_payload,
)

SCHEMA_VERSION = "hazard_odd_coverage_rollup.v1"
CLAIM_BOUNDARY = (
    "Coverage rollup separates metadata-only contract surfaces from executed benchmark evidence. "
    "Fallback, degraded, failed, and not_available rows remain caveats and are not success evidence."
)

_CAVEATED_EXECUTION_STATUSES = {
    "degraded",
    "fallback",
    "failed",
    "failure",
    "not_available",
    "partial-failure",
    "unavailable",
    "false",
}
_EXCLUDED_CERT_STATUSES = {
    "invalid",
    "geometrically_infeasible",
    "kinodynamically_infeasible",
    "dynamically_overconstrained",
    "excluded",
    "ineligible",
}
_SCENARIO_ID_FIELDS = ("scenario_id", "scenario", "scenario_name", "scenario_key")
_SCENARIO_FAMILY_FIELDS = ("scenario_family", "family", "scenario_set", "map_family")
_CAMPAIGN_TABLES = (
    "reports/campaign_table.csv",
    "reports/seed_episode_rows.csv",
    "reports/episode_summary.csv",
)


@dataclass(frozen=True, slots=True)
class CampaignRow:
    """Normalized campaign row with scenario identity and execution caveats."""

    source: str
    index: int
    scenario_id: str
    scenario_family: str
    execution_labels: tuple[str, ...]
    caveated: bool
    metadata: Mapping[str, str]


def build_hazard_odd_coverage_rollup(  # noqa: PLR0913
    *,
    campaign_root: Path,
    output: Path,
    report_id: str = "hazard_odd_coverage",
    hazard_traceability_path: Path | None = None,
    odd_contract_paths: Sequence[Path] = (),
    scenario_contract_paths: Sequence[Path] = (),
    scenario_cert_paths: Sequence[Path] = (),
    stress_uncertainty_coverage_path: Path | None = None,
    command: str = "",
) -> dict[str, Any]:
    """Build and write a campaign-level hazard/ODD coverage report.

    Returns:
        JSON-safe payload containing output paths and the generated summary.
    """

    campaign_root = campaign_root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows, row_sources = _load_campaign_rows(campaign_root)
    hazard_mapping, hazard_input = _load_hazard_mapping(hazard_traceability_path)
    odd_contracts, odd_inputs = _load_odd_contracts(odd_contract_paths)
    scenario_contracts, scenario_inputs = _load_scenario_contracts(scenario_contract_paths)
    cert_records, cert_inputs = _load_scenario_certificates(scenario_cert_paths)
    stress_summary = _load_stress_summary(stress_uncertainty_coverage_path)

    excluded_scenarios = _excluded_scenarios(cert_records)
    hazard_rows = _hazard_rows(hazard_mapping, rows, excluded_scenarios)
    odd_rows = _odd_rows(odd_contracts, scenario_contracts, rows)
    scenario_contract_rows = _scenario_contract_rows(scenario_contracts, rows, excluded_scenarios)

    table_paths = _write_tables(
        output=output,
        hazard_rows=hazard_rows,
        odd_rows=odd_rows,
        scenario_contract_rows=scenario_contract_rows,
    )
    figure_path = _write_status_figure(output, hazard_rows, odd_rows)
    markdown_path = _write_markdown_summary(
        output=output,
        hazard_rows=hazard_rows,
        odd_rows=odd_rows,
        scenario_contract_rows=scenario_contract_rows,
        row_count=len(rows),
    )

    generated_artifacts = _generated_artifacts(
        {
            "hazard_coverage_table_csv": table_paths["hazards"],
            "odd_boundary_table_csv": table_paths["odd"],
            "scenario_contract_table_csv": table_paths["scenario_contracts"],
            "hazard_odd_coverage_summary_md": markdown_path,
            "coverage_status_figure_png": figure_path,
        }
    )
    checksum_path = _write_checksums(output, [Path(item["path"]) for item in generated_artifacts])

    summary = {
        "schema_version": SCHEMA_VERSION,
        "report_id": report_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "campaign_root": campaign_root.as_posix(),
        "claim_boundary": CLAIM_BOUNDARY,
        "executed_evidence": _executed_evidence_summary(rows),
        "metadata_inputs": [
            *row_sources,
            hazard_input,
            *odd_inputs,
            *scenario_inputs,
            *cert_inputs,
            stress_summary["input"],
        ],
        "stress_uncertainty_coverage": stress_summary["summary"],
        "hazards": hazard_rows,
        "odd_boundaries": odd_rows,
        "scenario_contracts": scenario_contract_rows,
        "provenance": {
            "source_files": _source_refs(
                row_sources,
                [
                    hazard_input,
                    *odd_inputs,
                    *scenario_inputs,
                    *cert_inputs,
                    stress_summary["input"],
                ],
            ),
            "generated_artifacts": generated_artifacts,
            "checksums": {
                "path": checksum_path.as_posix(),
                "sha256": sha256_file(checksum_path),
            },
            "generation_command": command,
            "generation_commit": _git_commit(),
        },
    }
    summary_path = output / "hazard_odd_coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "summary": summary,
        "outputs": {
            "hazard_odd_coverage_summary_json": summary_path.as_posix(),
            "hazard_odd_coverage_summary_md": markdown_path.as_posix(),
            "hazard_coverage_table_csv": table_paths["hazards"].as_posix(),
            "odd_boundary_table_csv": table_paths["odd"].as_posix(),
            "scenario_contract_table_csv": table_paths["scenario_contracts"].as_posix(),
            "coverage_status_figure_png": figure_path.as_posix(),
            "checksums": checksum_path.as_posix(),
        },
    }


def _load_campaign_rows(campaign_root: Path) -> tuple[list[CampaignRow], list[dict[str, Any]]]:
    """Load supported campaign row tables from a campaign root."""

    rows: list[CampaignRow] = []
    inputs: list[dict[str, Any]] = []
    for relative in _CAMPAIGN_TABLES:
        path = campaign_root / relative
        if not path.exists():
            inputs.append(_input_ref(path, "unavailable", "optional campaign table missing"))
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for index, raw_row in enumerate(reader):
                row = {field: _cell(raw_row.get(field)) for field in (reader.fieldnames or [])}
                rows.append(_campaign_row(path, index, row))
        inputs.append(_input_ref(path, "available", f"loaded {relative}"))
    # Deduplicate across tables so overlapping execution rows cannot inflate
    # evidence_rows or caveated_rows.
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    deduplicated: list[CampaignRow] = []
    for row in rows:
        key = (row.scenario_id, row.scenario_family, row.execution_labels)
        if key not in seen:
            seen.add(key)
            deduplicated.append(row)
    return deduplicated, inputs


def _campaign_row(path: Path, index: int, row: Mapping[str, str]) -> CampaignRow:
    """Normalize one row from a campaign CSV."""

    labels = _execution_labels(row)
    caveated = bool(set(labels).intersection(_CAVEATED_EXECUTION_STATUSES))
    return CampaignRow(
        source=path.as_posix(),
        index=index,
        scenario_id=_first_present(row, _SCENARIO_ID_FIELDS),
        scenario_family=_first_present(row, _SCENARIO_FAMILY_FIELDS),
        execution_labels=tuple(labels),
        caveated=caveated,
        metadata=row,
    )


def _load_hazard_mapping(path: Path | None) -> tuple[HazardTraceability | None, dict[str, Any]]:
    """Load optional hazard traceability metadata."""

    if path is None:
        return None, _input_ref(None, "unavailable", "hazard traceability file not supplied")
    try:
        return load_hazard_traceability(path), _input_ref(
            path, "available", "loaded hazard mapping"
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        return None, _input_ref(path, "unavailable", f"could not load hazard mapping: {exc}")


def _load_odd_contracts(paths: Sequence[Path]) -> tuple[list[OddContract], list[dict[str, Any]]]:
    """Load optional ODD contract metadata."""

    contracts: list[OddContract] = []
    inputs: list[dict[str, Any]] = []
    if not paths:
        return [], [_input_ref(None, "unavailable", "ODD contract file not supplied")]
    for path in paths:
        try:
            loaded = load_odd_contracts(path)
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            inputs.append(_input_ref(path, "unavailable", f"could not load ODD contracts: {exc}"))
            continue
        contracts.extend(loaded)
        inputs.append(_input_ref(path, "available", f"loaded {len(loaded)} ODD contract(s)"))
    return contracts, inputs


def _load_scenario_contracts(
    paths: Sequence[Path],
) -> tuple[list[ScenarioContract], list[dict[str, Any]]]:
    """Load optional scenario contract metadata."""

    contracts: list[ScenarioContract] = []
    inputs: list[dict[str, Any]] = []
    if not paths:
        return [], [_input_ref(None, "unavailable", "scenario contract file not supplied")]
    for path in paths:
        try:
            loaded = load_scenario_contracts(path)
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            inputs.append(
                _input_ref(path, "unavailable", f"could not load scenario contracts: {exc}")
            )
            continue
        contracts.extend(loaded)
        inputs.append(_input_ref(path, "available", f"loaded {len(loaded)} scenario contract(s)"))
    return contracts, inputs


def _load_scenario_certificates(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load optional scenario certificate dictionaries."""

    records: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    if not paths:
        return [], [_input_ref(None, "unavailable", "scenario_cert.v1 file not supplied")]
    for path in paths:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
            inputs.append(_input_ref(path, "unavailable", f"could not load scenario certs: {exc}"))
            continue
        loaded = _as_record_list(raw)
        records.extend(loaded)
        inputs.append(_input_ref(path, "available", f"loaded {len(loaded)} scenario cert(s)"))
    return records, inputs


def _load_stress_summary(path: Path | None) -> dict[str, Any]:
    """Load optional stress/uncertainty coverage status."""

    if path is None:
        return {
            "input": _input_ref(
                None, "unavailable", "stress/uncertainty coverage file not supplied"
            ),
            "summary": {
                "status": "unavailable",
                "schema_version": None,
                "claim_boundary": "stress/uncertainty metadata absent from this rollup",
            },
        }
    try:
        payload = load_stress_uncertainty_coverage_payload(path)
    except (OSError, json.JSONDecodeError, StressUncertaintyCoverageError) as exc:
        return {
            "input": _input_ref(path, "unavailable", f"could not load coverage report: {exc}"),
            "summary": {"status": "unavailable", "error": str(exc)},
        }
    return {
        "input": _input_ref(path, "available", "loaded stress/uncertainty coverage report"),
        "summary": {
            "status": payload.get("availability_status", "available"),
            "schema_version": payload.get("schema_version"),
            "schema_mode": payload.get("schema_mode"),
            "missing_fields": payload.get("missing_fields", []),
        },
    }


def _hazard_rows(
    mapping: HazardTraceability | None,
    rows: Sequence[CampaignRow],
    excluded_scenarios: set[str],
) -> list[dict[str, Any]]:
    """Build the hazard coverage table rows."""

    if mapping is None:
        return [
            {
                "hazard_id": "hazard_traceability",
                "status": "unavailable",
                "evidence_rows": 0,
                "caveated_rows": 0,
                "matched_scenarios": "",
                "reason": "hazard traceability metadata unavailable",
            }
        ]

    mappings_by_hazard: dict[str, list[Any]] = defaultdict(list)
    for item in mapping.scenario_mappings:
        for hazard_id in item.hazards:
            mappings_by_hazard[hazard_id].append(item)

    output_rows: list[dict[str, Any]] = []
    for hazard in mapping.hazards:
        linked = mappings_by_hazard.get(hazard.id, [])
        matched = _matching_rows(rows, linked)
        good = [row for row in matched if not row.caveated]
        caveated = [row for row in matched if row.caveated]
        mapped_scenarios = sorted(
            {
                scenario
                for item in linked
                for scenario in [*item.scenario_ids, *item.scenario_families]
            }
        )
        status = _coverage_status(
            matched=matched,
            good=good,
            caveated=caveated,
            mapped_scenarios=mapped_scenarios,
            excluded_scenarios=excluded_scenarios,
        )
        output_rows.append(
            {
                "hazard_id": hazard.id,
                "status": status,
                "severity": hazard.severity,
                "evidence_rows": len(good),
                "caveated_rows": len(caveated),
                "matched_scenarios": ", ".join(_row_scenarios(matched)),
                "supporting_metrics": ", ".join(hazard.supporting_metrics),
                "reason": _coverage_reason(status, matched, caveated, mapped_scenarios),
            }
        )
    return output_rows


def _odd_rows(
    odd_contracts: Sequence[OddContract],
    scenario_contracts: Sequence[ScenarioContract],
    rows: Sequence[CampaignRow],
) -> list[dict[str, Any]]:
    """Build ODD boundary table rows."""

    if not odd_contracts:
        return [
            {
                "contract_id": "odd_contract",
                "boundary_type": "input",
                "boundary_id": "odd_contract.v1",
                "status": "unavailable",
                "evidence_rows": 0,
                "caveated_rows": 0,
                "reason": "ODD contract metadata unavailable",
            }
        ]

    contracts_by_id = {contract.id: contract for contract in odd_contracts}
    refs_by_contract: dict[str, list[ScenarioContract]] = defaultdict(list)
    for contract in scenario_contracts:
        if contract.odd_contract_ref is not None:
            refs_by_contract[contract.odd_contract_ref.contract_id].append(contract)

    output_rows: list[dict[str, Any]] = []
    for contract_id, contract in sorted(contracts_by_id.items()):
        linked = refs_by_contract.get(contract_id, [])
        matched = _rows_for_scenario_contracts(rows, linked)
        good = [row for row in matched if not row.caveated]
        caveated = [row for row in matched if row.caveated]
        for claim_id in contract.claim_boundaries.supported_claims:
            status = _odd_supported_status(linked, matched, good, caveated)
            output_rows.append(
                {
                    "contract_id": contract_id,
                    "boundary_type": "supported_claim",
                    "boundary_id": claim_id,
                    "status": status,
                    "evidence_rows": len(good),
                    "caveated_rows": len(caveated),
                    "reason": _odd_supported_reason(status),
                }
            )
        for claim_id in [*contract.claim_boundaries.non_claims, *contract.exclusions]:
            output_rows.append(
                {
                    "contract_id": contract_id,
                    "boundary_type": "excluded_claim",
                    "boundary_id": claim_id,
                    "status": "excluded",
                    "evidence_rows": 0,
                    "caveated_rows": 0,
                    "reason": "ODD contract explicitly excludes this claim or operating condition",
                }
            )
    return output_rows


def _scenario_contract_rows(
    contracts: Sequence[ScenarioContract],
    rows: Sequence[CampaignRow],
    excluded_scenarios: set[str],
) -> list[dict[str, Any]]:
    """Build metadata-only scenario contract table rows."""

    if not contracts:
        return [
            {
                "contract_id": "scenario_contract",
                "scenario": "",
                "scenario_family": "",
                "status": "unavailable",
                "evidence_rows": 0,
                "caveated_rows": 0,
                "reason": "scenario_contract.v1 metadata unavailable",
            }
        ]

    output_rows: list[dict[str, Any]] = []
    for contract in contracts:
        matched = _rows_for_scenario_contracts(rows, [contract])
        good = [row for row in matched if not row.caveated]
        caveated = [row for row in matched if row.caveated]
        scenario_name = contract.scenario_ref.scenario_name
        mapped = [scenario_name, contract.scenario_ref.scenario_family]
        status = _coverage_status(
            matched=matched,
            good=good,
            caveated=caveated,
            mapped_scenarios=mapped,
            excluded_scenarios=excluded_scenarios,
        )
        output_rows.append(
            {
                "contract_id": contract.id,
                "scenario": scenario_name,
                "scenario_family": contract.scenario_ref.scenario_family,
                "status": status,
                "evidence_rows": len(good),
                "caveated_rows": len(caveated),
                "reason": _coverage_reason(status, matched, caveated, mapped),
            }
        )
    return output_rows


def _write_tables(
    *,
    output: Path,
    hazard_rows: Sequence[Mapping[str, Any]],
    odd_rows: Sequence[Mapping[str, Any]],
    scenario_contract_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Write all CSV tables."""

    hazards = output / "hazard_coverage_table.csv"
    odd = output / "odd_boundary_table.csv"
    scenario_contracts = output / "scenario_contract_table.csv"
    _write_csv(hazards, hazard_rows)
    _write_csv(odd, odd_rows)
    _write_csv(scenario_contracts, scenario_contract_rows)
    return {"hazards": hazards, "odd": odd, "scenario_contracts": scenario_contracts}


def _write_markdown_summary(
    *,
    output: Path,
    hazard_rows: Sequence[Mapping[str, Any]],
    odd_rows: Sequence[Mapping[str, Any]],
    scenario_contract_rows: Sequence[Mapping[str, Any]],
    row_count: int,
) -> Path:
    """Write a compact Markdown summary."""

    path = output / "hazard_odd_coverage_summary.md"
    text = "\n".join(
        [
            "# Hazard And ODD Coverage Summary",
            "",
            CLAIM_BOUNDARY,
            "",
            f"- Executed/row-level records read: {row_count}",
            f"- Hazard statuses: {_status_counts_text(hazard_rows)}",
            f"- ODD boundary statuses: {_status_counts_text(odd_rows)}",
            f"- Scenario contract statuses: {_status_counts_text(scenario_contract_rows)}",
            "",
            "## Interpretation Caveats",
            "",
            "- `covered` requires at least one non-caveated executed row.",
            "- `partial` preserves fallback, degraded, failed, or not_available row caveats.",
            "- `missing` means metadata maps the category but no executed row represented it.",
            "- `excluded` comes from ODD exclusions or fail-closed scenario certification metadata.",
            "- `unavailable` means an optional metadata surface or campaign table was absent.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")
    return path


def _write_status_figure(
    output: Path,
    hazard_rows: Sequence[Mapping[str, Any]],
    odd_rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Write a small status-count figure."""

    path = output / "coverage_status_summary.png"
    counts = Counter()
    for row in [*hazard_rows, *odd_rows]:
        counts[str(row["status"])] += 1
    labels = sorted(counts)
    values = [counts[label] for label in labels]
    colors = [_status_color(label) for label in labels]

    fig = Figure(figsize=(7, 3.5))
    ax = fig.subplots()
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Rows")
    ax.set_title("Hazard and ODD coverage status")
    ax.set_ylim(0, max(values or [1]) + 1)
    ax.grid(axis="y", color="#d0d7de", linewidth=0.8, alpha=0.8)
    for index, value in enumerate(values):
        ax.text(index, value + 0.05, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return path


def _write_checksums(output: Path, paths: Iterable[Path]) -> Path:
    """Write a SHA-256 manifest for generated artifacts."""

    checksum_path = output / "checksums.sha256"
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(paths, key=lambda item: item.relative_to(output).as_posix())
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows as a deterministic CSV file."""

    fieldnames = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _cell(row.get(field)) for field in fieldnames})


def _matching_rows(rows: Sequence[CampaignRow], mappings: Sequence[Any]) -> list[CampaignRow]:
    """Return campaign rows matching any scenario ID or family in the mappings."""

    scenario_ids = {scenario for item in mappings for scenario in item.scenario_ids}
    scenario_families = {family for item in mappings for family in item.scenario_families}
    return [
        row
        for row in rows
        if (row.scenario_id and row.scenario_id in scenario_ids)
        or (row.scenario_family and row.scenario_family in scenario_families)
    ]


def _rows_for_scenario_contracts(
    rows: Sequence[CampaignRow],
    contracts: Sequence[ScenarioContract],
) -> list[CampaignRow]:
    """Return rows matching scenario contract names or families."""

    scenario_ids = {contract.scenario_ref.scenario_name for contract in contracts}
    scenario_families = {contract.scenario_ref.scenario_family for contract in contracts}
    return [
        row
        for row in rows
        if (row.scenario_id and row.scenario_id in scenario_ids)
        or (row.scenario_family and row.scenario_family in scenario_families)
    ]


def _coverage_status(
    *,
    matched: Sequence[CampaignRow],
    good: Sequence[CampaignRow],
    caveated: Sequence[CampaignRow],
    mapped_scenarios: Sequence[str],
    excluded_scenarios: set[str],
) -> str:
    """Classify coverage using executed evidence and metadata exclusions."""

    mapped_set = {value for value in mapped_scenarios if value}
    if mapped_set and mapped_set.issubset(excluded_scenarios):
        return "excluded"
    if good and not caveated:
        return "covered"
    if matched:
        return "partial"
    return "missing"


def _coverage_reason(
    status: str,
    matched: Sequence[CampaignRow],
    caveated: Sequence[CampaignRow],
    mapped_scenarios: Sequence[str],
) -> str:
    """Return a short human-readable status reason."""

    if status == "covered":
        return "represented by non-caveated executed row evidence"
    if status == "partial":
        if caveated:
            labels = sorted({label for row in caveated for label in row.execution_labels})
            return "represented only or partly by caveated rows: " + ", ".join(labels)
        return "metadata matched executed rows, but no non-caveated evidence row was identified"
    if status == "excluded":
        return "all mapped scenarios are excluded by scenario_cert.v1 metadata"
    if mapped_scenarios:
        return "metadata present but no executed row matched: " + ", ".join(mapped_scenarios)
    if matched:
        return "executed rows matched without explicit metadata"
    return "metadata unavailable for this coverage row"


def _odd_supported_status(
    linked: Sequence[ScenarioContract],
    matched: Sequence[CampaignRow],
    good: Sequence[CampaignRow],
    caveated: Sequence[CampaignRow],
) -> str:
    """Classify a supported ODD claim boundary."""

    if not linked:
        return "missing"
    if good and not caveated:
        return "covered"
    return "partial"


def _odd_supported_reason(status: str) -> str:
    """Return a compact reason for supported-claim rows."""

    return {
        "covered": "linked scenario contracts have non-caveated executed row evidence",
        "partial": "ODD metadata is linked, but executed evidence is absent or caveated",
        "missing": "ODD claim has no linked scenario contract evidence in this rollup",
    }.get(status, "status classified by ODD metadata")


def _executed_evidence_summary(rows: Sequence[CampaignRow]) -> dict[str, Any]:
    """Summarize normalized campaign row evidence."""

    return {
        "row_count": len(rows),
        "non_caveated_row_count": sum(1 for row in rows if not row.caveated),
        "caveated_row_count": sum(1 for row in rows if row.caveated),
        "scenario_ids": sorted({row.scenario_id for row in rows if row.scenario_id}),
        "scenario_families": sorted({row.scenario_family for row in rows if row.scenario_family}),
        "execution_labels": sorted({label for row in rows for label in row.execution_labels}),
    }


def _excluded_scenarios(cert_records: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return scenario IDs that scenario certificates classify as excluded."""

    excluded: set[str] = set()
    for record in cert_records:
        scenario_id = _cell(record.get("scenario_id") or record.get("scenario"))
        classification = _cell(record.get("classification")).lower()
        eligibility = _cell(record.get("benchmark_eligibility")).lower()
        if scenario_id and (
            classification in _EXCLUDED_CERT_STATUSES or eligibility in _EXCLUDED_CERT_STATUSES
        ):
            excluded.add(scenario_id)
    return excluded


def _as_record_list(raw: Any) -> list[dict[str, Any]]:
    """Normalize supported JSON/YAML shapes into record dictionaries."""

    if isinstance(raw, Mapping):
        if isinstance(raw.get("certificates"), list):
            return [dict(item) for item in raw["certificates"] if isinstance(item, Mapping)]
        return [dict(raw)]
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def _execution_labels(row: Mapping[str, str]) -> list[str]:
    """Extract normalized execution-status labels from a raw row."""

    labels: list[str] = []
    for field in (
        "execution_mode",
        "status",
        "availability_status",
        "readiness_status",
        "benchmark_success",
    ):
        value = _cell(row.get(field)).lower()
        if value:
            labels.append(value)
    return labels or ["unknown"]


def _first_present(row: Mapping[str, str], fields: Sequence[str]) -> str:
    """Return the first non-empty row value for candidate fields."""

    for field in fields:
        value = _cell(row.get(field))
        if value:
            return value
    return ""


def _row_scenarios(rows: Sequence[CampaignRow]) -> list[str]:
    """Return stable scenario identity labels for matched rows."""

    values = {row.scenario_id or row.scenario_family for row in rows}
    return sorted(value for value in values if value)


def _input_ref(path: Path | None, status: str, reason: str) -> dict[str, Any]:
    """Return a provenance/status record for an input file."""

    if path is None:
        return {"path": None, "status": status, "reason": reason, "sha256": None}
    resolved = path.resolve()
    sha = sha256_file(resolved) if resolved.exists() and resolved.is_file() else None
    return {"path": resolved.as_posix(), "status": status, "reason": reason, "sha256": sha}


def _source_refs(
    row_sources: Sequence[Mapping[str, Any]],
    inputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return available source references with checksums."""

    refs: list[dict[str, Any]] = []
    for item in [*row_sources, *inputs]:
        if item.get("status") == "available" and item.get("path"):
            refs.append({"path": item["path"], "sha256": item.get("sha256")})
    return refs


def _generated_artifacts(paths: Mapping[str, Path]) -> list[dict[str, str]]:
    """Return path/checksum records for generated artifacts."""

    return [
        {"artifact_id": artifact_id, "path": path.as_posix(), "sha256": sha256_file(path)}
        for artifact_id, path in paths.items()
    ]


def _status_counts_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render status counts for Markdown."""

    counts = Counter(str(row.get("status", "unknown")) for row in rows)
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts)) or "none"


def _status_color(status: str) -> str:
    """Return a fixed color for a coverage status."""

    return {
        "covered": "#2da44e",
        "partial": "#bf8700",
        "missing": "#cf222e",
        "excluded": "#8250df",
        "unavailable": "#6e7781",
    }.get(status, "#57606a")


def _cell(value: object) -> str:
    """Normalize optional CSV/JSON values to compact text."""

    if value is None:
        return ""
    return str(value).strip()


def _git_commit() -> str:
    """Return the current Git commit, or ``unknown`` outside Git."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-id", default="hazard_odd_coverage")
    parser.add_argument("--hazard-traceability", type=Path)
    parser.add_argument("--odd-contract", type=Path, action="append", default=[])
    parser.add_argument("--scenario-contract", type=Path, action="append", default=[])
    parser.add_argument("--scenario-cert", type=Path, action="append", default=[])
    parser.add_argument("--stress-uncertainty-coverage", type=Path)
    return parser


def command_from_args(args: argparse.Namespace) -> str:
    """Return a reproducible command string for provenance."""

    parts = [
        "uv run python scripts/tools/hazard_odd_coverage_rollup.py",
        f"--campaign-root {args.campaign_root.as_posix()}",
        f"--output {args.output.as_posix()}",
        f"--report-id {args.report_id}",
    ]
    if args.hazard_traceability is not None:
        parts.append(f"--hazard-traceability {args.hazard_traceability.as_posix()}")
    for path in args.odd_contract:
        parts.append(f"--odd-contract {path.as_posix()}")
    for path in args.scenario_contract:
        parts.append(f"--scenario-contract {path.as_posix()}")
    for path in args.scenario_cert:
        parts.append(f"--scenario-cert {path.as_posix()}")
    if args.stress_uncertainty_coverage is not None:
        parts.append(f"--stress-uncertainty-coverage {args.stress_uncertainty_coverage.as_posix()}")
    return " ".join(parts)


def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Run the rollup from parsed CLI arguments."""

    return build_hazard_odd_coverage_rollup(
        campaign_root=args.campaign_root,
        output=args.output,
        report_id=args.report_id,
        hazard_traceability_path=args.hazard_traceability,
        odd_contract_paths=args.odd_contract,
        scenario_contract_paths=args.scenario_contract,
        scenario_cert_paths=args.scenario_cert,
        stress_uncertainty_coverage_path=args.stress_uncertainty_coverage,
        command=command_from_args(args),
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "SCHEMA_VERSION",
    "build_arg_parser",
    "build_hazard_odd_coverage_rollup",
    "command_from_args",
    "run_from_args",
]
