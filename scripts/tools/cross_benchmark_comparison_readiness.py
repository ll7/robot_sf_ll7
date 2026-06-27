#!/usr/bin/env python3
"""Report prerequisite readiness for the cross-benchmark policy comparison (#3287).

Issue #3287 proposes evaluating the same navigation policy across Robot SF and external
social-navigation suites *after* scenario converters, metric wrappers, and external benchmark
assets exist. That is a downstream benchmark campaign, not something this helper runs. The
campaign is explicitly blocked on converter (#3285) and metric-wrapper (#3286) readiness plus
external benchmark assets (#1456 / #1498 / #2414 / #3161 / #2918), so this helper deliberately
does **not** access external assets, run a campaign, or claim cross-suite equivalence.

Instead it answers a narrow, repeatable question: *are the local prerequisites in place to
stage the comparison campaign, and which standing blockers remain?* For each prerequisite
family it inventories the expected repository artifacts and classifies the family as:

- ``ready``   -- every expected local artifact is present;
- ``blocked`` -- one or more expected artifacts are missing (the default for external assets,
  which are never staged in-repo);
- ``waived``  -- a maintainer has explicitly waived the family with a recorded reason. Per the
  issue acceptance criteria, prerequisites must be "satisfied or explicitly waived"; a waiver
  always carries a reason so the waiver trail is auditable.

The report is presence-only and fail-closed: ``campaign_authorized`` is always ``False`` and
``run_gates`` lists the standing blockers, because authorizing the actual cross-suite run stays
gated on the open prerequisite issues, staged external assets, and a maintainer decision. A
"prerequisites ready" report must never be mistaken for "authorized to claim equivalence".

Example:
    uv run python scripts/tools/cross_benchmark_comparison_readiness.py
    uv run python scripts/tools/cross_benchmark_comparison_readiness.py --json
    uv run python scripts/tools/cross_benchmark_comparison_readiness.py \
        --waive external_assets:"assets staged out-of-band on the cluster"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Prerequisite family lifecycle states, ordered by escalating readiness. A family is only
# counted toward "prerequisites ready" when it is ``ready`` or explicitly ``waived``.
FAMILY_STATES = ("blocked", "waived", "ready")

# External benchmark assets are never staged inside this repository (license + size), so the
# external-asset family is a standing blocker until the assets are staged out-of-band or the
# maintainer explicitly waives it. These asset ids are owned by the canonical external-data
# registry in ``scripts/tools/manage_external_data.py``; a regression test guards that they
# still exist there so this reference cannot silently drift.
SOCIAL_NAV_EXTERNAL_ASSET_IDS: tuple[str, ...] = (
    "socnavbench-s3dis-eth",
    "socnavbench-control",
)

# Standing blockers that gate the actual campaign run regardless of local file presence. These
# are recorded so a "prerequisites ready" report is never read as "authorized to run / claim
# equivalence".
RUN_GATES: tuple[str, ...] = (
    "converter (#3285) and metric-wrapper (#3286) prerequisites satisfied or explicitly waived",
    "external benchmark assets (#1456 / #1498 / #2414 / #3161 / #2918) staged out-of-band",
    "campaign manifest pins policies, versions, scenarios, metrics, seeds, and external assets",
    "sim-to-sim differences reported as limitations; no direct-equivalence claim where mappings "
    "are approximate",
    "maintainer authorizes the cross-suite comparison run",
)


@dataclass(frozen=True)
class PrerequisiteFamily:
    """Static definition of one prerequisite family and its expected repository artifacts."""

    family_id: str
    display_name: str
    description: str
    # Repository-relative paths that must all exist for the family to be ``ready``.
    required_paths: tuple[Path, ...]
    # Issue(s) that own this prerequisite, for cross-reference in the report.
    source_issues: tuple[int, ...] = ()
    # When True, the family is a standing external blocker: it never auto-classifies as ``ready``
    # from local file presence and can only clear via an explicit waiver.
    external_blocker: bool = False
    notes: str = ""


# The four prerequisite families named by the issue scope: converter readiness, metric-wrapper
# status, policy metadata, and external-asset blockers.
PREREQUISITE_FAMILIES: tuple[PrerequisiteFamily, ...] = (
    PrerequisiteFamily(
        family_id="converter",
        display_name="Scenario interop converter (#3285)",
        description=(
            "Robot SF <-> external scenario converter plus its intermediate-representation "
            "schema and CLI, required before scenarios can be aligned across suites."
        ),
        required_paths=(
            Path("robot_sf/benchmark/scenario_interop.py"),
            Path("robot_sf/benchmark/schemas/scenario_interop_ir.v1.json"),
            Path("scripts/tools/convert_scenario_interop.py"),
        ),
        source_issues=(3285,),
        notes="Introduced by #3285; blocked here until that converter contract lands on main.",
    ),
    PrerequisiteFamily(
        family_id="metric_wrapper",
        display_name="Cross-benchmark metric wrappers (#3286)",
        description=(
            "Trace-derived metric correspondence layer and its mapping table, required so "
            "metrics can be aligned (exact / approximate / unavailable) across suites."
        ),
        required_paths=(
            Path("robot_sf/benchmark/cross_benchmark_metrics.py"),
            Path("configs/benchmarks/cross_benchmark_metric_mapping_v1.yaml"),
        ),
        source_issues=(3286,),
        notes="Wrapper preserves approximate/unavailable rows; it is not simulator-parity proof.",
    ),
    PrerequisiteFamily(
        family_id="policy_metadata",
        display_name="Campaign policy metadata / manifest",
        description=(
            "A campaign manifest pinning the one or two comparison policies with versions, "
            "scenarios, metrics, seeds, and external assets, so the run is mechanical and "
            "reproducible once converter, metric, and asset prerequisites clear."
        ),
        required_paths=(Path("configs/benchmarks/cross_benchmark_policy_comparison_v1.yaml"),),
        source_issues=(3287,),
        notes=(
            "Manifest scaffold is downstream campaign-design work (deferred); blocked until the "
            "manifest is authored. Authoring it must not assert cross-suite equivalence."
        ),
    ),
    PrerequisiteFamily(
        family_id="external_assets",
        display_name="External social-nav benchmark assets",
        description=(
            "External social-navigation suite assets (e.g. the SocNavBench family) that must be "
            "staged out-of-band; never staged in-repo and never accessed by this helper."
        ),
        required_paths=(),
        source_issues=(1456, 1498, 2414, 3161, 2918),
        external_blocker=True,
        notes=(
            "Standing external blocker. Stage via "
            "`scripts/tools/manage_external_data.py stage <asset-id> --source <path>` or waive "
            "explicitly. Relevant asset ids: " + ", ".join(SOCIAL_NAV_EXTERNAL_ASSET_IDS) + "."
        ),
    ),
)


@dataclass
class PathStatus:
    """Presence record for a single expected prerequisite path."""

    path: str
    exists: bool


@dataclass
class FamilyReadiness:
    """Readiness classification for one prerequisite family."""

    family_id: str
    display_name: str
    status: str  # one of FAMILY_STATES
    paths: list[PathStatus]
    missing_paths: list[str] = field(default_factory=list)
    source_issues: list[int] = field(default_factory=list)
    description: str = ""
    external_blocker: bool = False
    waiver_reason: str | None = None
    notes: str = ""


class WaiverError(ValueError):
    """Raised when a waiver is malformed (unknown family or missing reason)."""


def _classify_paths(repo_root: Path, paths: tuple[Path, ...]) -> tuple[list[PathStatus], list[str]]:
    """Return per-path presence records and the list of missing repository-relative paths.

    A path counts as present if the file or directory exists, so directory-shaped
    prerequisites (config families, schema dirs) are valid.
    """
    statuses: list[PathStatus] = []
    missing: list[str] = []
    for rel in paths:
        exists = (repo_root / rel).exists()
        statuses.append(PathStatus(path=rel.as_posix(), exists=exists))
        if not exists:
            missing.append(rel.as_posix())
    return statuses, missing


def evaluate_family(
    repo_root: Path,
    family: PrerequisiteFamily,
    waivers: dict[str, str] | None = None,
) -> FamilyReadiness:
    """Classify a single prerequisite family as ready, blocked, or waived.

    An explicit waiver (a non-empty reason keyed by ``family_id``) always wins, mirroring the
    issue's "satisfied or explicitly waived" acceptance criterion. Otherwise the family is
    ``ready`` only when it is not an external blocker and every required path is present.
    """
    waivers = waivers or {}
    statuses, missing = _classify_paths(repo_root, family.required_paths)

    if family.family_id in waivers:
        status = "waived"
        waiver_reason = waivers[family.family_id]
    elif family.external_blocker:
        status = "blocked"
        waiver_reason = None
    else:
        status = "ready" if not missing else "blocked"
        waiver_reason = None

    return FamilyReadiness(
        family_id=family.family_id,
        display_name=family.display_name,
        status=status,
        paths=statuses,
        missing_paths=missing,
        source_issues=list(family.source_issues),
        description=family.description,
        external_blocker=family.external_blocker,
        waiver_reason=waiver_reason,
        notes=family.notes,
    )


def evaluate_readiness(
    repo_root: Path = REPO_ROOT,
    waivers: dict[str, str] | None = None,
) -> dict:
    """Build the full presence-only readiness report for the #3287 comparison campaign.

    The payload separates *prerequisite readiness* (which this helper can verify from local
    files and recorded waivers) from *campaign authorization* (which it must never assert):
    ``campaign_authorized`` is always ``False`` and ``run_gates`` lists the standing blockers.
    """
    waivers = validate_waivers(waivers or {})
    families = [evaluate_family(repo_root, family, waivers) for family in PREREQUISITE_FAMILIES]

    cleared = all(fam.status in ("ready", "waived") for fam in families)
    prerequisites_status = "ready" if cleared else "blocked"

    return {
        "issue": 3287,
        "report": "cross-benchmark-comparison-readiness",
        "scope": (
            "presence-only local prerequisite check; does not access external assets, run a "
            "campaign, or claim cross-suite equivalence"
        ),
        "prerequisites_status": prerequisites_status,
        "campaign_authorized": False,
        "run_gates": list(RUN_GATES),
        "families": [_family_to_dict(fam) for fam in families],
    }


def validate_waivers(waivers: dict[str, str]) -> dict[str, str]:
    """Validate a waiver mapping; return it unchanged if every entry is well-formed.

    A waiver must name a known prerequisite family and carry a non-empty reason. This keeps
    waivers explicit and auditable rather than a silent way to mark a blocked prerequisite as
    cleared.
    """
    known = {family.family_id for family in PREREQUISITE_FAMILIES}
    for family_id, reason in waivers.items():
        if family_id not in known:
            supported = ", ".join(sorted(known))
            raise WaiverError(
                f"Unknown prerequisite family '{family_id}'. Supported families: {supported}."
            )
        if not reason or not reason.strip():
            raise WaiverError(f"Waiver for '{family_id}' must include a non-empty reason.")
    return waivers


def _family_to_dict(family: FamilyReadiness) -> dict:
    """Serialize a FamilyReadiness to a plain JSON-friendly dict."""
    payload: dict = {
        "id": family.family_id,
        "display_name": family.display_name,
        "status": family.status,
        "paths": [{"path": p.path, "exists": p.exists} for p in family.paths],
        "missing_paths": family.missing_paths,
        "source_issues": family.source_issues,
        "description": family.description,
        "external_blocker": family.external_blocker,
    }
    if family.waiver_reason is not None:
        payload["waiver_reason"] = family.waiver_reason
    if family.notes:
        payload["notes"] = family.notes
    return payload


def render_text(report: dict) -> str:
    """Render a compact human-readable readiness summary."""
    lines: list[str] = []
    lines.append("Cross-benchmark policy comparison readiness (#3287)")
    lines.append(f"  scope: {report['scope']}")
    lines.append(f"  prerequisites: {report['prerequisites_status'].upper()}")
    lines.append(f"  campaign authorized: {report['campaign_authorized']} (presence-only check)")
    lines.append("")

    for family in report["families"]:
        issues = (
            " ".join(f"#{num}" for num in family["source_issues"])
            if family["source_issues"]
            else "-"
        )
        lines.append(
            f"[{family['id']}] {family['display_name']} ({issues}): {family['status'].upper()}"
        )
        for path in family["paths"]:
            mark = "ok " if path["exists"] else "MISS"
            lines.append(f"    [{mark}] {path['path']}")
        if family.get("waiver_reason"):
            lines.append(f"    waived: {family['waiver_reason']}")
        elif family["missing_paths"]:
            lines.append(f"    blockers: {', '.join(family['missing_paths'])}")
        elif family["external_blocker"]:
            lines.append("    blockers: external assets staged out-of-band (none in-repo)")
        lines.append("")

    lines.append("Run gates (must clear before the campaign run; out of scope for this helper):")
    for gate in report["run_gates"]:
        lines.append(f"  - {gate}")
    return "\n".join(lines)


def _parse_waiver_args(raw_waivers: list[str] | None) -> dict[str, str]:
    """Parse repeated ``--waive family_id:reason`` CLI arguments into a mapping.

    The first colon splits family id from reason so reasons may themselves contain colons.
    """
    waivers: dict[str, str] = {}
    for entry in raw_waivers or []:
        family_id, sep, reason = entry.partition(":")
        if not sep:
            raise WaiverError(
                f"Waiver '{entry}' must be 'family_id:reason' with an explicit reason."
            )
        waivers[family_id.strip()] = reason.strip()
    return waivers


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable JSON report instead of the text summary.",
    )
    parser.add_argument(
        "--waive",
        action="append",
        metavar="FAMILY_ID:REASON",
        help=(
            "Explicitly waive a prerequisite family with a recorded reason. Repeatable. "
            "Example: --waive external_assets:'staged on the cluster'."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to evaluate prerequisites against (defaults to this checkout).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print the readiness report; exit 0 when prerequisites are cleared, else 1.

    The non-zero exit on blocked prerequisites is a fail-closed signal for callers staging the
    comparison campaign. It says nothing about campaign authorization, which stays gated on
    ``run_gates``.
    """
    args = _parse_args(argv)
    try:
        waivers = _parse_waiver_args(args.waive)
        report = evaluate_readiness(args.repo_root, waivers)
    except WaiverError as exc:
        print(f"error: {exc}")
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    return 0 if report["prerequisites_status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
