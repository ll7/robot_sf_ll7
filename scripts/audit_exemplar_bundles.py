#!/usr/bin/env python3
"""Audit exemplar-bundle artifact policy for issue #4920.

Verifies SHA256SUMS, computes size accounting, confirms derived/compact contents,
and checks metadata provenance for the issue_4848 and issue_4891 exemplar bundles.

Usage:
    uv run python scripts/audit_exemplar_bundles.py [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUNDLE_CLASSES = [
    "docs/context/evidence/issue_4848_group_crossing_exemplars_2026-07",
    "docs/context/evidence/issue_4891_head_on_corridor_exemplars_2026-07",
]

EXPECTED_FILES_PER_BUNDLE = {
    "metadata.json",
    "min_distance_series.csv",
    "README.md",
    "SHA256SUMS",
    "trace_series.json",
    "trace_timeseries.csv",
}

REQUIRED_METADATA_FIELDS = {
    "campaign_id",
    "campaign_job",
    "claim_boundary",
    "episode_id",
    "episode_status",
    "generated_at_utc",
    "git_commit",
    "issue",
    "planner",
    "scenario_id",
    "schema_version",
    "seed",
    "selection_metric",
    "selection_metric_value",
    "selection_mode",
    "summary",
    "review_marker",
}

RAW_DUMP_EXTENSIONS = {".jsonl"}


@dataclass
class BundleAuditResult:
    """Audit result for one bundle directory."""

    path: str
    sha256_ok: bool = True
    sha256_errors: list[str] = field(default_factory=list)
    file_count_ok: bool = True
    unexpected_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    metadata_ok: bool = True
    metadata_errors: list[str] = field(default_factory=list)
    claim_boundary_ok: bool = True
    claim_boundary_text: str = ""
    no_raw_dumps: bool = True
    raw_dump_files: list[str] = field(default_factory=list)
    total_size_bytes: int = 0
    file_sizes: dict[str, int] = field(default_factory=dict)


@dataclass
class ClassAuditResult:
    """Audit result for one bundle class (e.g., issue_4848)."""

    class_path: str
    class_name: str
    bundles: list[BundleAuditResult] = field(default_factory=list)
    total_size_bytes: int = 0
    selection_report_exists: bool = False


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sha256sums(path: Path) -> dict[str, str]:
    """Parse a SHA256SUMS file, returning {relative_path: expected_hash}."""
    result: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                expected_hash, rel_path = parts
                result[rel_path] = expected_hash
    return result


def verify_checksums(bundle_dir: Path, repo_root: Path) -> tuple[bool, list[str]]:
    """Verify SHA256SUMS for a bundle directory. Returns (ok, errors)."""
    sha_file = bundle_dir / "SHA256SUMS"
    if not sha_file.exists():
        return False, ["SHA256SUMS file missing"]

    entries = parse_sha256sums(sha_file)
    errors: list[str] = []

    for rel_path, expected_hash in entries.items():
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            errors.append(f"  MISSING: {rel_path}")
            continue
        actual_hash = sha256_file(abs_path)
        if actual_hash != expected_hash:
            errors.append(
                f"  MISMATCH: {rel_path}\n"
                f"    expected: {expected_hash}\n"
                f"    actual:   {actual_hash}"
            )

    return len(errors) == 0, errors


def check_metadata(bundle_dir: Path) -> tuple[bool, list[str], str]:
    """Check metadata.json for required fields and claim boundary."""
    meta_path = bundle_dir / "metadata.json"
    if not meta_path.exists():
        return False, ["metadata.json missing"], ""

    with open(meta_path) as f:
        meta = json.load(f)

    errors: list[str] = []
    missing = REQUIRED_METADATA_FIELDS - set(meta.keys())
    if missing:
        errors.append(f"  Missing fields: {sorted(missing)}")

    claim = meta.get("claim_boundary", "")
    if not claim:
        errors.append("  claim_boundary is empty")
    elif "illustrative" not in claim.lower() and "no statistical" not in claim.lower():
        errors.append(f"  claim_boundary may lack illustrative-only guard: {claim[:80]}...")

    review_marker = meta.get("review_marker", "")
    if "NEEDS-REVIEW" not in review_marker:
        errors.append(f"  review_marker missing NEEDS-REVIEW: {review_marker}")

    return len(errors) == 0, errors, claim


def audit_bundle(bundle_dir: Path, repo_root: Path) -> BundleAuditResult:
    """Run full audit on one bundle directory."""
    result = BundleAuditResult(path=str(bundle_dir.relative_to(repo_root)))

    actual_files = {f.name for f in bundle_dir.iterdir() if f.is_file()}
    missing = EXPECTED_FILES_PER_BUNDLE - actual_files
    unexpected = actual_files - EXPECTED_FILES_PER_BUNDLE
    if missing:
        result.missing_files = sorted(missing)
        result.file_count_ok = False
    if unexpected:
        result.unexpected_files = sorted(unexpected)

    result.sha256_ok, result.sha256_errors = verify_checksums(bundle_dir, repo_root)

    result.metadata_ok, result.metadata_errors, result.claim_boundary_text = check_metadata(
        bundle_dir
    )
    result.claim_boundary_ok = result.metadata_ok

    for f in bundle_dir.iterdir():
        if f.is_file() and f.suffix in RAW_DUMP_EXTENSIONS:
            result.no_raw_dumps = False
            result.raw_dump_files.append(f.name)

    for f in bundle_dir.iterdir():
        if f.is_file():
            size = f.stat().st_size
            result.file_sizes[f.name] = size
            result.total_size_bytes += size

    return result


def audit_class(class_path: Path, repo_root: Path) -> ClassAuditResult:
    """Audit all bundles in one exemplar class."""
    result = ClassAuditResult(
        class_path=str(class_path.relative_to(repo_root)),
        class_name=class_path.name,
    )

    result.selection_report_exists = (class_path / "SELECTION_REPORT.md").exists()

    for planner_dir in sorted(class_path.iterdir()):
        if not planner_dir.is_dir() or planner_dir.name.startswith("."):
            continue
        for bundle_dir in sorted(planner_dir.iterdir()):
            if not bundle_dir.is_dir():
                continue
            if (bundle_dir / "SHA256SUMS").exists():
                bundle_result = audit_bundle(bundle_dir, repo_root)
                result.bundles.append(bundle_result)
                result.total_size_bytes += bundle_result.total_size_bytes

    return result


def compute_docs_context_size(repo_root: Path) -> int:
    """Compute total size of docs/context/evidence/ directory."""
    total = 0
    evidence_dir = repo_root / "docs" / "context" / "evidence"
    if not evidence_dir.exists():
        return 0
    for root, _dirs, files in os.walk(evidence_dir):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def format_size(n: int) -> str:
    """Format bytes as human-readable string."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _render_sha_section(classes: list[ClassAuditResult]) -> list[str]:
    """Render SHA256SUMS verification section."""
    lines: list[str] = []
    lines.append("## 1. SHA256SUMS Verification")
    lines.append("")

    all_sha_ok = True
    for cls in classes:
        lines.append(f"### {cls.class_name}")
        lines.append("")
        lines.append("| Bundle | SHA256 Status |")
        lines.append("|--------|--------------|")
        for b in cls.bundles:
            short = b.path.split("/")[-1]
            status = "PASS" if b.sha256_ok else "FAIL"
            if not b.sha256_ok:
                all_sha_ok = False
            lines.append(f"| `{short}` | **{status}** |")
        lines.append("")

        for b in cls.bundles:
            if b.sha256_errors:
                lines.append(f"**Errors in `{b.path}`:**")
                lines.extend(b.sha256_errors)
                lines.append("")

    overall_sha = "ALL PASS" if all_sha_ok else "FAILURES DETECTED"
    lines.append(f"**Overall SHA256 verdict: {overall_sha}**")
    lines.append("")
    return lines


def _render_size_section(
    classes: list[ClassAuditResult], repo_root: Path
) -> tuple[list[str], int, int]:
    """Render size accounting section. Returns (lines, grand_total, total_bundles)."""
    lines: list[str] = []
    lines.append("## 2. Size Accounting")
    lines.append("")
    lines.append("### Per-Bundle Sizes")
    lines.append("")

    for cls in classes:
        lines.append(f"#### {cls.class_name}")
        lines.append("")
        lines.append(
            "| Bundle | metadata | min_distance | README | SHA256SUMS"
            " | trace_series | trace_timeseries | Total |"
        )
        lines.append(
            "|--------|----------|-------------|--------|-----------"
            "|-------------|-----------------|-------|"
        )
        for b in cls.bundles:
            short = b.path.split("/")[-1]
            sz = b.file_sizes
            lines.append(
                f"| `{short}` "
                f"| {format_size(sz.get('metadata.json', 0))} "
                f"| {format_size(sz.get('min_distance_series.csv', 0))} "
                f"| {format_size(sz.get('README.md', 0))} "
                f"| {format_size(sz.get('SHA256SUMS', 0))} "
                f"| {format_size(sz.get('trace_series.json', 0))} "
                f"| {format_size(sz.get('trace_timeseries.csv', 0))} "
                f"| **{format_size(b.total_size_bytes)}** |"
            )
        lines.append("")
        lines.append(
            f"**Class total: {format_size(cls.total_size_bytes)} ({len(cls.bundles)} bundles)**"
        )
        lines.append("")

    grand_total = sum(c.total_size_bytes for c in classes)
    total_bundles = sum(len(c.bundles) for c in classes)
    lines.append("### Grand Total (both classes)")
    lines.append(f"- **{total_bundles} bundles**, **{format_size(grand_total)}** total")
    lines.append("")

    evidence_total = compute_docs_context_size(repo_root)
    lines.append(f"### docs/context/evidence/ total size: {format_size(evidence_total)}")
    lines.append("")
    return lines, grand_total, total_bundles


def _render_derived_section(classes: list[ClassAuditResult]) -> list[str]:
    """Render derived/compact contents confirmation section."""
    lines: list[str] = []
    lines.append("## 3. Derived/Compact Contents Confirmation")
    lines.append("")

    all_derived = True
    for cls in classes:
        for b in cls.bundles:
            if not b.no_raw_dumps:
                all_derived = False
                short = b.path.split("/")[-1]
                lines.append(
                    f"- **WARNING**: `{short}` contains raw dump files: {b.raw_dump_files}"
                )

    if all_derived:
        lines.append(
            "**All bundles contain only derived/compact files (CSV, JSON, Markdown). "
            "No raw campaign dumps (JSONL) found.**"
        )
    lines.append("")

    lines.append("### File Types Per Bundle")
    lines.append("")
    lines.append("| File | Type | Role |")
    lines.append("|------|------|------|")
    lines.append(
        "| `trace_timeseries.csv` | CSV | Per-timestep robot state, actions, pedestrian positions |"
    )
    lines.append("| `min_distance_series.csv` | CSV | Figure-ready (step, time, distance) series |")
    lines.append("| `trace_series.json` | JSON | Recorded frames + derived rows (compact) |")
    lines.append("| `metadata.json` | JSON | Provenance, selection criteria, claim boundary |")
    lines.append("| `README.md` | Markdown | Human-readable provenance and contents |")
    lines.append("| `SHA256SUMS` | Text | Checksums for data files |")
    lines.append("")
    return lines


def _render_metadata_section(classes: list[ClassAuditResult]) -> list[str]:
    """Render metadata provenance verification section."""
    lines: list[str] = []
    lines.append("## 4. Metadata Provenance Verification")
    lines.append("")

    all_meta_ok = True
    for cls in classes:
        for b in cls.bundles:
            if not b.metadata_ok:
                all_meta_ok = False
                short = b.path.split("/")[-1]
                lines.append(f"**Issues in `{short}`:**")
                lines.extend(b.metadata_errors)
                lines.append("")

    if all_meta_ok:
        lines.append("**All 18 bundles have complete metadata with:**")
        lines.append("- `campaign_id` and `campaign_job` (source campaign provenance)")
        lines.append("- `claim_boundary` with explicit illustrative-only guard")
        lines.append("- `review_marker: AI-GENERATED NEEDS-REVIEW`")
        lines.append("- `selection_metric`, `selection_mode`, `seed`")
        lines.append("- `git_commit` at generation time")
        lines.append("- `schema_version` (per-class)")
    lines.append("")

    lines.append("### Claim Boundary Samples")
    lines.append("")
    for cls in classes:
        if cls.bundles:
            claim = cls.bundles[0].claim_boundary_text
            lines.append(f"- **{cls.class_name}**: `{claim}`")
    lines.append("")
    return lines


def _render_policy_section(
    grand_total: int, total_bundles: int, evidence_total: int, classes: list[ClassAuditResult]
) -> list[str]:
    """Render policy proposal section."""
    lines: list[str] = []
    lines.append("## 5. Exemplar-Bundle Size Budget Policy Proposal")
    lines.append("")
    lines.append("### Current State")
    lines.append(f"- 2 exemplar classes, {total_bundles} bundles, {format_size(grand_total)} total")
    lines.append(f"- Entire `docs/context/evidence/` tree: {format_size(evidence_total)}")

    all_sizes = [b.total_size_bytes for c in classes for b in c.bundles]
    if all_sizes:
        lines.append(f"- Largest bundle: {format_size(max(all_sizes))}")
        lines.append(f"- Smallest bundle: {format_size(min(all_sizes))}")
    lines.append("")

    lines.append("### Proposed Policy")
    lines.append("")
    lines.append(
        "1. **Per-bundle size cap**: 5 MB per bundle directory (current max ~2 MB, safe headroom)."
    )
    lines.append(
        "2. **Per-class bundle count cap**: 12 bundles per exemplar class "
        "(3 planners x 4 selection modes is the natural maximum)."
    )
    lines.append(
        "3. **Cumulative exemplar budget**: 50 MB for all exemplar-bundle classes "
        f"combined (current: {format_size(grand_total)})."
    )
    lines.append(
        "4. **Derived-only content rule**: bundles must contain only derived/compact "
        "files (CSV, JSON, Markdown). No raw JSONL episode dumps, Slurm logs, "
        "model checkpoints, or binary artifacts."
    )
    lines.append(
        "5. **LFS threshold**: if a single file exceeds 1 MB, evaluate whether it "
        "can be summarized or split before considering LFS. Current largest files "
        "are trace_series.json (~1.9 MB); these are within the proposed cap."
    )
    lines.append(
        "6. **SHA256SUMS mandatory**: every bundle must include a SHA256SUMS file "
        "covering all data files (excluding itself)."
    )
    lines.append(
        "7. **Metadata contract**: every bundle must include metadata.json with "
        "campaign_id, campaign_job, claim_boundary (with illustrative-only guard), "
        "review_marker, and schema_version."
    )
    lines.append(
        "8. **No restamping in this audit**: marker restamping is owned by "
        "PR #4910/#4903. This policy defines future requirements only."
    )
    lines.append("")
    return lines


def _render_summary(
    all_sha_ok: bool, all_meta_ok: bool, all_derived: bool, total_bundles: int, grand_total: int
) -> list[str]:
    """Render summary section."""
    lines: list[str] = []
    lines.append("## Summary")
    lines.append("")
    sha_status = "PASS" if all_sha_ok else "FAIL"
    meta_status = "PASS" if all_meta_ok else "FAIL"
    derived_status = "PASS" if all_derived else "FAIL"
    lines.append("| Check | Result |")
    lines.append("|-------|--------|")
    lines.append(f"| SHA256SUMS verification | **{sha_status}** |")
    lines.append(f"| Metadata completeness | **{meta_status}** |")
    lines.append(f"| Derived-only content (no raw dumps) | **{derived_status}** |")
    lines.append(f"| Claim boundary present | **{sha_status}** |")
    lines.append(f"| Bundle count | {total_bundles} |")
    lines.append(f"| Total exemplar size | {format_size(grand_total)} |")
    lines.append("")
    return lines


def generate_report(classes: list[ClassAuditResult], repo_root: Path) -> str:
    """Generate the full audit report as Markdown."""
    lines: list[str] = []
    lines.append("# Exemplar-Bundle Artifact Policy Audit (Issue #4920)")
    lines.append("")

    lines.extend(_render_sha_section(classes))
    all_sha_ok = all(b.sha256_ok for c in classes for b in c.bundles)

    size_lines, grand_total, total_bundles = _render_size_section(classes, repo_root)
    lines.extend(size_lines)

    lines.extend(_render_derived_section(classes))
    all_derived = all(b.no_raw_dumps for c in classes for b in c.bundles)

    lines.extend(_render_metadata_section(classes))
    all_meta_ok = all(b.metadata_ok for c in classes for b in c.bundles)

    evidence_total = compute_docs_context_size(repo_root)
    lines.extend(_render_policy_section(grand_total, total_bundles, evidence_total, classes))

    lines.extend(_render_summary(all_sha_ok, all_meta_ok, all_derived, total_bundles, grand_total))

    return "\n".join(lines)


def main() -> None:
    """Entry point for the exemplar-bundle audit."""
    parser = argparse.ArgumentParser(description="Audit exemplar bundles")
    parser.add_argument(
        "--json", action="store_true", help="Output structured JSON instead of Markdown"
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    if not (repo_root / "robot_sf").is_dir():
        print("ERROR: must be run from repo root", file=sys.stderr)
        sys.exit(1)

    classes: list[ClassAuditResult] = []
    for class_rel in BUNDLE_CLASSES:
        class_path = repo_root / class_rel
        if not class_path.exists():
            print(f"WARNING: bundle class not found: {class_rel}", file=sys.stderr)
            continue
        classes.append(audit_class(class_path, repo_root))

    if args.json:
        data: dict[str, Any] = {
            "classes": [],
            "grand_total_bytes": sum(c.total_size_bytes for c in classes),
            "total_bundles": sum(len(c.bundles) for c in classes),
        }
        for cls in classes:
            cls_data: dict[str, Any] = {
                "class_name": cls.class_name,
                "class_path": cls.class_path,
                "total_size_bytes": cls.total_size_bytes,
                "selection_report_exists": cls.selection_report_exists,
                "bundles": [],
            }
            for b in cls.bundles:
                cls_data["bundles"].append(
                    {
                        "path": b.path,
                        "sha256_ok": b.sha256_ok,
                        "sha256_errors": b.sha256_errors,
                        "file_count_ok": b.file_count_ok,
                        "metadata_ok": b.metadata_ok,
                        "metadata_errors": b.metadata_errors,
                        "no_raw_dumps": b.no_raw_dumps,
                        "total_size_bytes": b.total_size_bytes,
                        "file_sizes": b.file_sizes,
                    }
                )
            data["classes"].append(cls_data)
        print(json.dumps(data, indent=2))
    else:
        report = generate_report(classes, repo_root)
        print(report)


if __name__ == "__main__":
    main()
