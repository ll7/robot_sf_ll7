#!/usr/bin/env python3
"""Render compact Package A transfer-report evidence for issue #3078.

The renderer is deterministic and evidence-boundary preserving: it validates the
Package A decision packet, reads canonical result-store rows when supplied, and
writes compact review artifacts. It never executes campaigns, submits compute,
or promotes paper-facing claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scripts.tools.campaign_result_store import read_parquet_frame
from scripts.validation.check_package_a_readiness import build_decision_packet

DEFAULT_READINESS_MANIFEST = Path("configs/benchmarks/issue_3078_package_a_readiness.yaml")
DEFAULT_HELDOUT_PARTITION_MANIFEST = Path(
    "configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml"
)
REPORT_SCHEMA_VERSION = "package_a_transfer_report.v1"
ELIGIBLE_EVIDENCE_STATUSES = {"native", "adapter"}
TABLE_FIELDS = (
    "surface",
    "planner",
    "scenario_family",
    "episode_count",
    "eligible_episode_count",
    "row_status_counts",
    "mean_snqi",
)


@dataclass(frozen=True)
class SurfaceFamilies:
    """Scenario-family groups declared by the held-out transfer partition manifest."""

    benchmark: tuple[str, ...]
    heldout: tuple[str, ...]

    @property
    def benchmark_labeling_mode(self) -> str:
        """Return how benchmark-set rows are identified for reporting."""
        if self.benchmark:
            return "explicit_benchmark_set_families"
        if self.heldout:
            return "inferred_by_excluding_heldout_families"
        return "undeclared"

    @property
    def benchmark_labeling_warning(self) -> str | None:
        """Return the warning shown when benchmark-set labeling is inferred."""
        if self.benchmark:
            return None
        if self.heldout:
            return (
                "benchmark-set families are not declared; benchmark_set rows are inferred as "
                "the complement of heldout_family_evaluation.scenario_families"
            )
        return (
            "benchmark-set and held-out scenario families are both undeclared; renderer cannot "
            "label transfer surfaces"
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _surface_families(partition_manifest: Path) -> SurfaceFamilies:
    payload = _load_yaml_mapping(partition_manifest)
    benchmark = _families_from_section(payload.get("benchmark_set_evaluation"))
    heldout = _families_from_section(payload.get("heldout_family_evaluation"))
    return SurfaceFamilies(benchmark=tuple(sorted(benchmark)), heldout=tuple(sorted(heldout)))


def _families_from_section(section: Any) -> set[str]:
    if not isinstance(section, dict):
        return set()
    families = section.get("scenario_families")
    if not isinstance(families, list):
        return set()
    return {str(family) for family in families if str(family).strip()}


def _empty_table_rows() -> list[dict[str, str]]:
    return []


def _table_rows(result_store: Path | None, families: SurfaceFamilies) -> list[dict[str, str]]:
    if result_store is None:
        return _empty_table_rows()
    if not families.benchmark and not families.heldout:
        raise ValueError(
            "held-out partition manifest must declare benchmark_set_evaluation.scenario_families "
            "or heldout_family_evaluation.scenario_families before rendering transfer surfaces"
        )

    frame = read_parquet_frame(result_store / "episodes.parquet")
    if "snqi" not in frame.columns:
        frame = frame.assign(snqi=None)

    rows: list[dict[str, str]] = []
    family_column = frame["scenario_family"].astype(str)
    surfaces = [
        ("benchmark_set", family_column.isin(families.benchmark))
        if families.benchmark
        else ("benchmark_set", ~family_column.isin(families.heldout)),
        ("heldout_family", family_column.isin(families.heldout)),
    ]
    for surface, mask in surfaces:
        surface_frame = frame[mask]
        if surface_frame.empty:
            continue
        grouped = surface_frame.groupby(["planner", "scenario_family"], dropna=False)
        for (planner, scenario_family), group in grouped:
            statuses = [str(value) for value in group["row_status"].tolist()]
            status_counts = {status: statuses.count(status) for status in sorted(set(statuses))}
            eligible = group[group["row_status"].astype(str).isin(ELIGIBLE_EVIDENCE_STATUSES)]
            mean_snqi = None
            if not eligible.empty and "snqi" in eligible.columns:
                mean_value = eligible["snqi"].dropna().mean()
                if not math.isnan(float(mean_value)):
                    mean_snqi = f"{float(mean_value):.6f}"
            rows.append(
                {
                    "surface": surface,
                    "planner": str(planner),
                    "scenario_family": str(scenario_family),
                    "episode_count": str(len(group)),
                    "eligible_episode_count": str(len(eligible)),
                    "row_status_counts": json.dumps(status_counts, sort_keys=True),
                    "mean_snqi": mean_snqi or "",
                }
            )
    return sorted(rows, key=lambda row: (row["surface"], row["planner"], row["scenario_family"]))


def _transfer_delta_rows(table_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_planner: dict[str, dict[str, list[float]]] = {}
    for row in table_rows:
        if not row["mean_snqi"]:
            continue
        by_planner.setdefault(row["planner"], {}).setdefault(row["surface"], []).append(
            float(row["mean_snqi"])
        )

    delta_rows: list[dict[str, str]] = []
    for planner in sorted(by_planner):
        benchmark_values = by_planner[planner].get("benchmark_set", [])
        heldout_values = by_planner[planner].get("heldout_family", [])
        benchmark_mean = _mean(benchmark_values)
        heldout_mean = _mean(heldout_values)
        delta = None
        if benchmark_mean is not None and heldout_mean is not None:
            delta = heldout_mean - benchmark_mean
        delta_rows.append(
            {
                "planner": planner,
                "benchmark_set_mean_snqi": _format_float(benchmark_mean),
                "heldout_family_mean_snqi": _format_float(heldout_mean),
                "transfer_delta_snqi": _format_float(delta),
                "claim_eligible": "false",
                "claim_boundary": "diagnostic_only_until_claim_card_review",
            }
        )
    return delta_rows


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_manifest(output_dir: Path, table_files: list[str]) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "package-a-transfer-report-tables-manifest.v1",
        "issue": 3078,
        "files": [{"path": filename, "role": Path(filename).stem} for filename in table_files],
    }
    (tables_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_claim_card(output_dir: Path, classification: str, rows: list[dict[str, str]]) -> None:
    non_promotable_counts: dict[str, int] = {}
    for row in rows:
        counts = json.loads(row["row_status_counts"])
        for status, count in counts.items():
            if status not in ELIGIBLE_EVIDENCE_STATUSES:
                non_promotable_counts[status] = non_promotable_counts.get(status, 0) + int(count)
    payload = {
        "schema_version": "package-a-transfer-claim-card.v1",
        "issue": 3078,
        "claim_status": "not_reviewed",
        "classification": classification,
        "claim_boundary": (
            "Diagnostic transfer report only; no paper-facing or benchmark-strength claim is "
            "promoted by this renderer."
        ),
        "eligible_row_statuses": sorted(ELIGIBLE_EVIDENCE_STATUSES),
        "non_promotable_row_status_counts": dict(sorted(non_promotable_counts.items())),
    }
    (output_dir / "claim_card.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _write_readme(
    output_dir: Path,
    *,
    classification: str,
    decision_reasons: list[str],
    result_store: Path | None,
) -> None:
    status = "blocked" if classification != "diagnostic_review_ready" else "diagnostic"
    result_store_text = str(result_store) if result_store else "not supplied"
    reasons = "\n".join(f"- {reason}" for reason in decision_reasons) or "- none"
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Issue #3078 Package A Transfer Report",
                "",
                "This compact report renders Package A transfer evidence inputs without running a campaign.",
                "",
                f"- Evidence status: `{status}`",
                f"- Decision-packet classification: `{classification}`",
                f"- Result store: `{result_store_text}`",
                "- Claim boundary: diagnostic-only until seed analysis, result-store validation, held-out leakage audit, and claim-card review all pass.",
                "- Forbidden actions confirmed: no benchmark campaign run, no Slurm/GPU submission, no paper-facing claim edit.",
                "",
                "## Decision Reasons",
                reasons,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_reproduction(
    output_dir: Path,
    *,
    readiness_manifest: Path,
    partition_manifest: Path,
    result_store: Path | None,
    seed_analysis_report: Path | None,
) -> None:
    command_parts = [
        "uv run python scripts/tools/build_package_a_transfer_report.py",
        f"--output-dir {output_dir}",
        f"--readiness-manifest {readiness_manifest}",
        f"--heldout-partition-manifest {partition_manifest}",
    ]
    if result_store:
        command_parts.append(f"--result-store {result_store}")
    if seed_analysis_report:
        command_parts.append(f"--seed-analysis-report {seed_analysis_report}")
    (output_dir / "reproduction.md").write_text(
        "\n".join(
            [
                "# Reproduction",
                "",
                "Renderer command:",
                "",
                "```bash",
                " \\\n  ".join(command_parts),
                "```",
                "",
                "This command renders compact evidence only. It does not run Package A campaigns.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_artifact_manifest(output_dir: Path, files: list[str]) -> None:
    payload = {
        "schema_version": "package-a-transfer-artifact-manifest.v1",
        "issue": 3078,
        "artifact_policy": "compact_tracked_evidence_only",
        "files": [
            {"path": filename, "sha256": _sha256(output_dir / filename)} for filename in files
        ],
    }
    (output_dir / "artifact_manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _write_checksums(output_dir: Path) -> None:
    lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(f"{_sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_report(
    *,
    output_dir: Path,
    readiness_manifest: Path,
    heldout_partition_manifest: Path,
    result_store: Path | None,
    seed_analysis_report: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    """Render compact Package A report and return summary metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_reports = [seed_analysis_report] if seed_analysis_report else []
    result_stores = [result_store] if result_store else []
    decision_packet = build_decision_packet(
        readiness_manifest,
        repo_root=repo_root,
        result_stores=result_stores,
        seed_analysis_reports=seed_reports,
        heldout_partition_manifests=[heldout_partition_manifest],
    )
    packet_payload = decision_packet.to_dict()
    (output_dir / "package_a_decision_packet.json").write_text(
        json.dumps(packet_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    invalid_supplied_evidence = any(
        not item["ok"]
        for key in ("result_stores", "seed_analysis_reports", "heldout_partition_manifests")
        for item in packet_payload[key]
    )
    if invalid_supplied_evidence:
        raise ValueError("supplied Package A evidence failed decision-packet validation")

    families = _surface_families(repo_root / heldout_partition_manifest)
    table_rows = _table_rows(repo_root / result_store if result_store else None, families)
    baseline_rows = [row for row in table_rows if row["surface"] == "benchmark_set"]
    heldout_rows = [row for row in table_rows if row["surface"] == "heldout_family"]
    transfer_rows = _transfer_delta_rows(table_rows)
    surface_labeling_warning = families.benchmark_labeling_warning

    _write_csv(output_dir / "baseline_table.csv", TABLE_FIELDS, baseline_rows)
    _write_csv(output_dir / "heldout_family_table.csv", TABLE_FIELDS, heldout_rows)
    _write_csv(
        output_dir / "transfer_delta.csv",
        (
            "planner",
            "benchmark_set_mean_snqi",
            "heldout_family_mean_snqi",
            "transfer_delta_snqi",
            "claim_eligible",
            "claim_boundary",
        ),
        transfer_rows,
    )
    (output_dir / "leakage_audit.md").write_text(
        "\n".join(
            [
                "# Leakage Audit",
                "",
                f"- Partition manifest: `{heldout_partition_manifest}`",
                f"- Benchmark-set labeling mode: `{families.benchmark_labeling_mode}`",
                (
                    f"- Surface-labeling warning: {surface_labeling_warning}"
                    if surface_labeling_warning
                    else "- Surface-labeling warning: none"
                ),
                f"- Benchmark-set families: {', '.join(families.benchmark) or 'none declared'}",
                f"- Held-out families: {', '.join(families.heldout) or 'none declared'}",
                "- Status: manifest validation delegated to the Package A decision packet.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    table_files = ["baseline_table.csv", "heldout_family_table.csv", "transfer_delta.csv"]
    _write_manifest(output_dir, table_files)
    _write_claim_card(output_dir, packet_payload["classification"], table_rows)
    _write_readme(
        output_dir,
        classification=packet_payload["classification"],
        decision_reasons=packet_payload["reasons"],
        result_store=result_store,
    )
    _write_reproduction(
        output_dir,
        readiness_manifest=readiness_manifest,
        partition_manifest=heldout_partition_manifest,
        result_store=result_store,
        seed_analysis_report=seed_analysis_report,
    )

    files = [
        "README.md",
        "artifact_manifest.yaml",
        "baseline_table.csv",
        "heldout_family_table.csv",
        "leakage_audit.md",
        "package_a_decision_packet.json",
        "reproduction.md",
        "claim_card.yaml",
        "tables/manifest.json",
        "transfer_delta.csv",
    ]
    _write_artifact_manifest(
        output_dir, [path for path in files if path != "artifact_manifest.yaml"]
    )
    _write_checksums(output_dir)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "rendered",
        "classification": packet_payload["classification"],
        "output_dir": str(output_dir),
        "files": sorted(
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--readiness-manifest", type=Path, default=DEFAULT_READINESS_MANIFEST)
    parser.add_argument(
        "--heldout-partition-manifest",
        type=Path,
        default=DEFAULT_HELDOUT_PARTITION_MANIFEST,
    )
    parser.add_argument("--result-store", type=Path)
    parser.add_argument("--seed-analysis-report", type=Path)
    parser.add_argument("--json", action="store_true", help="Print compact JSON summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Package A transfer-report renderer CLI."""
    args = _build_parser().parse_args(argv)
    try:
        summary = render_report(
            output_dir=args.output_dir,
            readiness_manifest=args.readiness_manifest,
            heldout_partition_manifest=args.heldout_partition_manifest,
            result_store=args.result_store,
            seed_analysis_report=args.seed_analysis_report,
            repo_root=_repo_root(),
        )
    except (ImportError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Rendered Package A transfer report: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
