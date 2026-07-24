#!/usr/bin/env python3
"""Build a fail-closed Chapter 8 statistics reproducibility packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.research.ch8_statistics import evaluate_statistic

SCHEMA_VERSION = "issue_4445.ch8_statistics_reproducibility.v1"
DEFAULT_MANIFEST = Path(
    "docs/context/evidence/issue_4445_ch8_statistics_reproducibility/source_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("output/analysis/issue_4445_ch8_statistics_reproducibility")
BOUNDARIES = (
    "diagnostic-only reproducibility packet; not benchmark evidence",
    "no paper or dissertation claim is established by this script alone",
    "missing source data or expected targets produce blocked status",
)


def _hydrate_statistic_data(spec: dict[str, Any], manifest_dir: Path) -> None:
    source_query = spec.get("source_query")
    if not isinstance(source_query, dict) or "input_file" not in source_query:
        return

    input_file_name = Path(source_query["input_file"]).name
    input_file_path = manifest_dir / input_file_name

    if not input_file_path.is_file():
        spec["data"] = {}
        return

    try:
        if input_file_name.endswith(".csv"):
            rows = []
            with input_file_path.open(encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
            spec["data"] = {
                "rows": rows,
                "metric": source_query.get("metric"),
                "x_field": source_query.get("x_field"),
                "y_field": source_query.get("y_field"),
                "samples": source_query.get("bootstrap_samples"),
                "seed": source_query.get("seed"),
                "planner": source_query.get("planner"),
            }
        elif input_file_name.endswith(".json"):
            with input_file_path.open(encoding="utf-8") as f:
                payload = json.load(f)
            spec["data"] = {"json_payload": payload}
    except (OSError, ValueError, KeyError, csv.Error):
        spec["data"] = {}


def build_packet(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    """Build JSON and Markdown packet outputs from ``manifest_path``."""

    manifest_path = manifest_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)

    manifest_dir = manifest_path.parent
    for spec in manifest["statistics"]:
        _hydrate_statistic_data(spec, manifest_dir)

    results = [evaluate_statistic(spec).to_json() for spec in manifest["statistics"]]
    packet = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "issue": 4445,
        "title": manifest["title"],
        "evidence_tier": "diagnostic-only",
        "claim_boundaries": list(BOUNDARIES),
        "manifest_path": _public_path(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_status": manifest["source_status"],
        "overall_status": _overall_status(results),
        "statistics": results,
    }
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    readme_path = output_dir / "README.md"
    checksums_path = output_dir / "SHA256SUMS"
    summary_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_text = _render_markdown(packet)
    report_path.write_text(report_text, encoding="utf-8")
    readme_path.write_text(report_text, encoding="utf-8")

    checksum_files = [
        summary_path,
        report_path,
        readme_path,
        manifest_path,
    ]
    for name in ("campaign_table.csv", "scenario_family_breakdown.csv"):
        path = manifest_dir / name
        if path.is_file():
            checksum_files.append(path)

    _write_checksums(
        checksums_path,
        checksum_files,
    )
    packet["outputs"] = {
        "summary_json": _public_path(summary_path),
        "report_markdown": _public_path(report_path),
        "readme_markdown": _public_path(readme_path),
        "sha256sums": _public_path(checksums_path),
    }
    return packet


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {SCHEMA_VERSION}")
    if int(payload.get("issue", 0)) != 4445:
        raise ValueError("manifest issue must be 4445")
    statistics = payload.get("statistics")
    if not isinstance(statistics, list) or not statistics:
        raise ValueError("manifest statistics must be a non-empty list")
    for index, entry in enumerate(statistics):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest statistics[{index}] must be a JSON object")
    for field in ("title", "source_status"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"manifest requires non-empty {field}")
    return payload


def _overall_status(results: list[dict[str, Any]]) -> str:
    statuses = {result["status"] for result in results}
    if all(status == "matches_expected" for status in statuses):
        return "reproducible"
    if any(status.startswith("blocked_") for status in statuses):
        return "blocked"
    if "computed_mismatch" in statuses:
        return "mismatch"
    return "computed_with_gaps"


def _render_markdown(packet: dict[str, Any]) -> str:
    lines = [
        "# Issue #4445 Chapter 8 Statistics Reproducibility Packet",
        "",
        f"- **Overall status**: `{packet['overall_status']}`",
        f"- **Evidence tier**: `{packet['evidence_tier']}`",
        f"- **Source status**: {packet['source_status']}",
        f"- **Manifest**: `{packet['manifest_path']}`",
        f"- **Manifest SHA-256**: `{packet['manifest_sha256']}`",
        "",
        "## Statistics",
        "",
    ]
    for result in packet["statistics"]:
        lines.extend(
            [
                f"### {result['id']}",
                "",
                f"- **Kind**: `{result['kind']}`",
                f"- **Status**: `{result['status']}`",
                f"- **Computed**: `{json.dumps(result['computed'], sort_keys=True)}`",
                f"- **Expected**: `{json.dumps(result['expected'], sort_keys=True)}`",
            ]
        )
        if result["blockers"]:
            lines.append(f"- **Blockers**: {'; '.join(result['blockers'])}")
        lines.append("")
    lines.extend(["## Claim Boundaries", ""])
    lines.extend(f"- {boundary}" for boundary in packet["claim_boundaries"])
    lines.append("")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_checksums(path: Path, files: list[Path]) -> None:
    lines = [f"{_sha256(file_path)}  {_public_path(file_path)}" for file_path in files]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _public_path(path: Path) -> str:
    resolved = path.resolve()
    for anchor in ("docs", "configs", "scripts", "tests", "output"):
        if anchor in resolved.parts:
            index = resolved.parts.index(anchor)
            return str(Path(*resolved.parts[index:]))
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return resolved.name


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    packet = build_packet(args.manifest, args.output_dir)
    print(f"overall_status={packet['overall_status']}")
    print(f"summary_json={packet['outputs']['summary_json']}")
    print(f"report_markdown={packet['outputs']['report_markdown']}")


if __name__ == "__main__":
    main()
