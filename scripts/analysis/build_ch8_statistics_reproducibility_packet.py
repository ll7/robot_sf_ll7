#!/usr/bin/env python3
"""Build a fail-closed Chapter 8 statistics reproducibility packet."""

from __future__ import annotations

import argparse
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


def build_packet(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    """Build JSON and Markdown packet outputs from ``manifest_path``."""

    manifest_path = manifest_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
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
    summary_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_render_markdown(packet), encoding="utf-8")
    packet["outputs"] = {
        "summary_json": _public_path(summary_path),
        "report_markdown": _public_path(report_path),
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
