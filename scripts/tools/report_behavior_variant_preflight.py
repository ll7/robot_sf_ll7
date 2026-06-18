"""Report benchmark validity for pedestrian/social-force behavior variants.

The report is intentionally conservative: it inventories implemented variants, checks local
runtime prerequisites for adapter-backed variants, and records whether each row can support
benchmark evidence or only diagnostic/unavailable evidence.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOCIAL_NAVIGATION_PYENVS_ROOT = Path("output/repos/Social-Navigation-PyEnvs")
AMMV_EVIDENCE = (
    Path("docs/context/evidence/issue_2432_ammv_trace_selection_2026-06-06/summary.json"),
    Path("docs/context/evidence/issue_2434_ammv_scenario_sweep_2026-06-06/summary.json"),
)


@dataclass(frozen=True)
class VariantSpec:
    """Static definition for one behavior-model variant row."""

    variant_id: str
    display_name: str
    family: str
    execution_mode: str
    required_paths: tuple[Path, ...]
    evidence_paths: tuple[Path, ...] = ()
    external_repo_required: bool = False
    required_python_package: str | None = None
    required_python_version: str | None = None
    benchmark_valid_when_available: bool = False
    diagnostic_only_reason: str | None = None


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        variant_id="social_force",
        display_name="Native Robot SF Social Force",
        family="native",
        execution_mode="adapter",
        required_paths=(
            Path("robot_sf/baselines/social_force.py"),
            Path("robot_sf/planner/socnav.py"),
            Path("configs/baselines/social_force_default.yaml"),
        ),
        benchmark_valid_when_available=True,
    ),
    VariantSpec(
        variant_id="ammv_social_force",
        display_name="AMMV-aware Robot SF Social Force",
        family="native_diagnostic",
        execution_mode="adapter",
        required_paths=(
            Path("robot_sf/baselines/social_force.py"),
            Path("configs/baselines/social_force_ammv_aware.yaml"),
        ),
        evidence_paths=AMMV_EVIDENCE,
        diagnostic_only_reason=(
            "existing same-seed adapter-mode evidence found no default-vs-AMMV frame or episode "
            "metric delta; use as diagnostic mechanism evidence until a behavioral-difference "
            "execution path is proven"
        ),
    ),
    VariantSpec(
        variant_id="social_navigation_pyenvs_orca",
        display_name="Social-Navigation-PyEnvs ORCA",
        family="external_adapter",
        execution_mode="adapter",
        required_paths=(Path("robot_sf/planner/social_navigation_pyenvs_orca.py"),),
        external_repo_required=True,
        benchmark_valid_when_available=True,
    ),
    VariantSpec(
        variant_id="social_navigation_pyenvs_socialforce",
        display_name="Social-Navigation-PyEnvs SocialForce",
        family="external_adapter",
        execution_mode="adapter",
        required_paths=(Path("robot_sf/planner/social_navigation_pyenvs_force_model.py"),),
        external_repo_required=True,
        required_python_package="socialforce",
        required_python_version="0.2.3",
        benchmark_valid_when_available=True,
    ),
    VariantSpec(
        variant_id="social_navigation_pyenvs_sfm_helbing",
        display_name="Social-Navigation-PyEnvs SFM Helbing",
        family="external_adapter",
        execution_mode="adapter",
        required_paths=(Path("robot_sf/planner/social_navigation_pyenvs_force_model.py"),),
        external_repo_required=True,
        benchmark_valid_when_available=True,
    ),
    VariantSpec(
        variant_id="social_navigation_pyenvs_hsfm_new_guo",
        display_name="Social-Navigation-PyEnvs HSFM New Guo",
        family="external_adapter",
        execution_mode="adapter",
        required_paths=(Path("robot_sf/planner/social_navigation_pyenvs_hsfm.py"),),
        external_repo_required=True,
        benchmark_valid_when_available=True,
    ),
)


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _generated_at() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is not None:
        try:
            return datetime.fromtimestamp(int(epoch), UTC).isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _path_status(repo_root: Path, paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.as_posix(),
            "exists": (repo_root / path).exists(),
        }
        for path in paths
    ]


def _display_path(repo_root: Path, path: Path) -> str:
    """Return a repo-relative path when possible."""
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return path.as_posix()
    if relative.parts and relative.parts[0] == "output":
        return "<local-artifact-root>/" + "/".join(relative.parts[1:])
    return relative.as_posix()


def _package_status(package: str | None, expected_version: str | None) -> dict[str, Any] | None:
    if package is None:
        return None
    spec = importlib.util.find_spec(package)
    if spec is None:
        return {
            "package": package,
            "status": "missing",
            "expected_version": expected_version,
        }
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    status = (
        "available" if expected_version is None or version == expected_version else "wrong_version"
    )
    return {
        "package": package,
        "status": status,
        "version": version,
        "expected_version": expected_version,
    }


def _load_ammv_result(repo_root: Path, evidence_paths: tuple[Path, ...]) -> dict[str, Any]:
    loaded: list[dict[str, Any]] = []
    for path in evidence_paths:
        full_path = repo_root / path
        if not full_path.exists():
            loaded.append({"path": path.as_posix(), "status": "missing"})
            continue
        payload = json.loads(full_path.read_text(encoding="utf-8"))
        result = payload.get("result") or {}
        loaded.append(
            {
                "path": path.as_posix(),
                "status": "available",
                "schema_version": payload.get("schema_version"),
                "candidate_pairs_compared": result.get("candidate_pairs_compared"),
                "non_identical_pairs_found": result.get("non_identical_pairs_found"),
                "max_per_frame_abs_delta": result.get("max_per_frame_abs_delta"),
                "decision": result.get("decision"),
            }
        )
    return {
        "evidence": loaded,
        "all_available": all(item["status"] == "available" for item in loaded),
        "behavioral_delta_found": any(
            (item.get("non_identical_pairs_found") or 0) > 0 for item in loaded
        ),
    }


def _classify_variant(
    spec: VariantSpec,
    *,
    repo_root: Path,
    external_repo_root: Path,
) -> dict[str, Any]:
    required_paths = _path_status(repo_root, spec.required_paths)
    missing_paths = [item["path"] for item in required_paths if not item["exists"]]
    dependencies: list[dict[str, Any]] = []
    if spec.external_repo_required:
        dependencies.append(
            {
                "dependency": "Social-Navigation-PyEnvs checkout",
                "path": _display_path(repo_root, external_repo_root),
                "status": "available" if external_repo_root.exists() else "missing",
            }
        )
    package = _package_status(spec.required_python_package, spec.required_python_version)
    if package is not None:
        dependencies.append({"dependency": "python package", **package})

    missing_dependencies = [
        item for item in dependencies if item.get("status") not in {"available", None}
    ]

    ammv_summary = None
    if spec.evidence_paths:
        ammv_summary = _load_ammv_result(repo_root, spec.evidence_paths)

    if missing_paths:
        availability_status = "not_available"
        row_status = "unavailable/excluded"
        benchmark_validity = "not_available"
        reason = f"missing required repository paths: {', '.join(missing_paths)}"
    elif missing_dependencies:
        availability_status = "not_available"
        row_status = "unavailable/excluded"
        benchmark_validity = "not_available"
        reason = "missing or incompatible runtime prerequisites"
    elif spec.diagnostic_only_reason is not None:
        availability_status = "available"
        row_status = "diagnostic_only"
        benchmark_validity = "diagnostic_only"
        reason = spec.diagnostic_only_reason
    elif spec.benchmark_valid_when_available:
        availability_status = "available"
        row_status = "benchmark_valid_candidate"
        benchmark_validity = "benchmark_valid_candidate"
        reason = "required repository paths and runtime prerequisites are present"
    else:
        availability_status = "unknown"
        row_status = "unknown"
        benchmark_validity = "unknown"
        reason = "no classification rule matched"

    row = {
        "variant_id": spec.variant_id,
        "display_name": spec.display_name,
        "family": spec.family,
        "execution_mode": spec.execution_mode,
        "availability_status": availability_status,
        "row_status": row_status,
        "benchmark_validity": benchmark_validity,
        "benchmark_success": availability_status == "available"
        and benchmark_validity == "benchmark_valid_candidate",
        "reason": reason,
        "required_paths": required_paths,
        "dependencies": dependencies,
    }
    if ammv_summary is not None:
        row["same_seed_evidence_summary"] = ammv_summary
    return row


def build_report(
    *,
    repo_root: Path = REPO_ROOT,
    social_navigation_pyenvs_root: Path | None = None,
) -> dict[str, Any]:
    """Build the behavior-variant validity report."""
    repo_root = repo_root.resolve()
    if social_navigation_pyenvs_root is None:
        social_navigation_pyenvs_root = repo_root / DEFAULT_SOCIAL_NAVIGATION_PYENVS_ROOT
    elif not social_navigation_pyenvs_root.is_absolute():
        social_navigation_pyenvs_root = repo_root / social_navigation_pyenvs_root
    social_navigation_pyenvs_root = social_navigation_pyenvs_root.resolve()

    rows = [
        _classify_variant(
            spec,
            repo_root=repo_root,
            external_repo_root=social_navigation_pyenvs_root,
        )
        for spec in VARIANTS
    ]
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["benchmark_validity"])
        counts[status] = counts.get(status, 0) + 1

    return {
        "schema_version": "issue_3064_behavior_variant_preflight.v1",
        "issue": 3064,
        "generated_at": _generated_at(),
        "git_head": _git_head(repo_root),
        "social_navigation_pyenvs_root": _display_path(repo_root, social_navigation_pyenvs_root),
        "claim_boundary": (
            "Preflight/inventory evidence only. Rows marked diagnostic_only, not_available, "
            "fallback, or degraded must not be counted as benchmark-success evidence."
        ),
        "status_counts": counts,
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report for human review."""
    lines = [
        "# Issue #3064 Behavior Variant Preflight",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Git head: `{report['git_head']}`",
        f"- Social-Navigation-PyEnvs root: `{report['social_navigation_pyenvs_root']}`",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Classification",
        "",
        "| Variant | Mode | Availability | Row status | Benchmark validity | Reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            "| "
            f"`{row['variant_id']}` | `{row['execution_mode']}` | "
            f"`{row['availability_status']}` | `{row['row_status']}` | "
            f"`{row['benchmark_validity']}` | {row['reason']} |"
        )

    lines.extend(["", "## Runtime Limitations", ""])
    for row in report["rows"]:
        deps = row.get("dependencies") or []
        if not deps:
            continue
        lines.append(f"- `{row['variant_id']}`:")
        for dep in deps:
            descriptor = dep.get("dependency", dep.get("package", "dependency"))
            lines.append(f"  - {descriptor}: `{dep.get('status')}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `social_force` is the current benchmark-valid native/adapter baseline candidate.",
            "- `ammv_social_force` remains diagnostic-only under the existing same-seed evidence.",
            "- Social-Navigation-PyEnvs rows are excluded unless their checkout and runtime "
            "dependencies are present in the local environment.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], *, output_json: Path, output_md: Path | None) -> None:
    """Write JSON and optional Markdown reports."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to inspect.",
    )
    parser.add_argument(
        "--social-navigation-pyenvs-root",
        type=Path,
        default=DEFAULT_SOCIAL_NAVIGATION_PYENVS_ROOT,
        help="Path to the Social-Navigation-PyEnvs checkout, relative to repo root by default.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the report generator CLI."""
    args = parse_args()
    report = build_report(
        repo_root=args.repo_root,
        social_navigation_pyenvs_root=args.social_navigation_pyenvs_root,
    )
    write_report(report, output_json=args.output_json, output_md=args.output_md)
    print(
        json.dumps(
            {
                "output_json": args.output_json.as_posix(),
                "output_md": args.output_md.as_posix() if args.output_md else None,
                "status_counts": report["status_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
