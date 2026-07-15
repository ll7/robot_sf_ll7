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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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


CAMPAIGN_MANIFEST_PATH = Path(
    "configs/benchmarks/cross_benchmark_policy_comparison_issue_3287.yaml"
)
LIMITATIONS_TEMPLATE_PATH = Path("docs/context/issue_3287_cross_benchmark_limitations_template.md")
REQUIRED_LIMITATION_SECTIONS: tuple[str, ...] = (
    "scenario_mapping_quality",
    "metric_denominator_differences",
    "observation_space_differences",
    "action_space_differences",
    "dynamics_and_pedestrian_model_differences",
    "unsupported_direct_equivalence_claims",
    "valid_bounded_comparison_statements",
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
        required_paths=(CAMPAIGN_MANIFEST_PATH, LIMITATIONS_TEMPLATE_PATH),
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


class CampaignManifestError(ValueError):
    """Raised when the issue #3287 campaign manifest scaffold is invalid."""


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


def _validate_manifest_header(manifest: dict, errors: list[str]) -> None:
    """Check issue identity and fail-closed campaign status."""
    expected_schema = "cross_benchmark_policy_comparison.issue_3287.v1"
    expected_values = {
        "schema_version": expected_schema,
        "issue": 3287,
        "status": "blocked_prerequisite",
        "campaign_authorized": False,
        "direct_equivalence_claim_allowed": False,
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            errors.append(f"{key} must be {expected!r}")


def _validate_non_empty_lists(manifest: dict, errors: list[str]) -> None:
    """Check the manifest carries required scaffold lists."""
    for key in (
        "policies",
        "robot_sf_scenarios",
        "external_benchmark_scenarios",
        "scenario_mapping_notes",
        "metric_mapping_notes",
        "seeds",
        "observation_action_space_caveats",
        "pedestrian_model_and_dynamics_caveats",
        "external_asset_provenance",
    ):
        value = manifest.get(key)
        if not isinstance(value, list) or not value:
            errors.append(f"{key} must be a non-empty list")


def _validate_limitations(manifest: dict, errors: list[str]) -> None:
    """Check the manifest points at the required limitations template sections."""
    if manifest.get("limitations_template") != LIMITATIONS_TEMPLATE_PATH.as_posix():
        errors.append(f"limitations_template must be {LIMITATIONS_TEMPLATE_PATH.as_posix()}")

    sections = manifest.get("limitations_sections")
    if not isinstance(sections, list):
        errors.append("limitations_sections must be a list")
        return

    missing = sorted(set(REQUIRED_LIMITATION_SECTIONS) - set(sections))
    if missing:
        errors.append("limitations_sections missing: " + ", ".join(missing))


def _validate_prerequisite_gate(manifest: dict, errors: list[str]) -> None:
    """Check prerequisite gates are explicit and waivers carry reasons."""
    prereqs = manifest.get("prerequisite_gate", {})
    if not isinstance(prereqs, dict):
        errors.append("prerequisite_gate must be a mapping")
        return

    for gate in ("converter", "metric_wrapper", "external_assets", "policy_compatibility"):
        entry = prereqs.get(gate)
        if not isinstance(entry, dict):
            errors.append(f"prerequisite_gate.{gate} must be a mapping")
            continue
        status = entry.get("status")
        if status not in {"ready", "blocked", "waived"}:
            errors.append(f"prerequisite_gate.{gate}.status must be ready, blocked, or waived")
        if status == "waived" and not str(entry.get("waiver_reason", "")).strip():
            errors.append(f"prerequisite_gate.{gate}.waiver_reason required when waived")


def validate_campaign_manifest(
    manifest_path: Path = REPO_ROOT / CAMPAIGN_MANIFEST_PATH,
) -> dict:
    """Validate the blocked #3287 campaign-manifest scaffold."""
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    errors: list[str] = []

    _validate_manifest_header(manifest, errors)
    _validate_non_empty_lists(manifest, errors)
    _validate_limitations(manifest, errors)
    _validate_prerequisite_gate(manifest, errors)

    if errors:
        raise CampaignManifestError("; ".join(errors))
    return manifest


# Tokens that must never appear in the authorized canary slice (issue #5783): they would mean
# the slice regressed to a placeholder/blocked scaffold instead of a concrete runnable canary.
CANARY_SLICE_FORBIDDEN_TOKENS = (
    "tbd",
    "to_be_selected",
    "to_be_selected_when_unblocked",
    "blocked_prerequisite",
    "blocked_external_input",
    "scaffold_only",
    "unvalidated_scaffold",
    "campaign_authorized: false",
)


@dataclass
class CanarySliceReport:
    """Result of validating the issue #5783 canary slice in the campaign manifest."""

    issue: int
    status: str
    canary_authorized: bool
    ok: bool
    policy_id: str | None
    policy_version: str | None
    algo: str | None
    algo_config: str | None
    robot_sf_scenario_id: str | None
    socnavbench_scenario_id: str | None
    seed: int | None
    external_asset_id: str | None
    metric_id: str | None
    limitation_flags: tuple[str, ...]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the canary-slice report to a JSON-friendly mapping."""
        return {
            "schema": "cross_benchmark_canary_slice.v1",
            "issue": self.issue,
            "status": self.status,
            "canary_authorized": self.canary_authorized,
            "ok": self.ok,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "algo": self.algo,
            "algo_config": self.algo_config,
            "robot_sf_scenario_id": self.robot_sf_scenario_id,
            "socnavbench_scenario_id": self.socnavbench_scenario_id,
            "seed": self.seed,
            "external_asset_id": self.external_asset_id,
            "metric_id": self.metric_id,
            "limitation_flags": list(self.limitation_flags),
            "errors": self.errors,
            "claim_boundary": (
                "diagnostic sim-to-sim canary only; not simulator equivalence, policy "
                "superiority, or a campaign/benchmark/paper claim"
            ),
        }


def _canary_slice_missing_report(errors: list[str]) -> CanarySliceReport:
    """Build the fail-closed report returned when the canary_slice block is absent."""
    errors.append("canary_slice block missing or not a mapping")
    return CanarySliceReport(
        issue=5783,
        status="missing",
        canary_authorized=False,
        ok=False,
        policy_id=None,
        policy_version=None,
        algo=None,
        algo_config=None,
        robot_sf_scenario_id=None,
        socnavbench_scenario_id=None,
        seed=None,
        external_asset_id=None,
        metric_id=None,
        limitation_flags=(),
        errors=errors,
    )


def _check_concrete_fields(
    block: dict,
    fields: tuple[tuple[str, Any], ...],
    errors: list[str],
) -> None:
    """Append an error for every required concrete field that is empty/placeholder."""
    for label, value in fields:
        if not value or not str(value).strip():
            errors.append(f"canary_slice.{label} must be concrete (no empty/placeholder)")


def _check_forbidden_tokens(slice_block: dict, errors: list[str]) -> None:
    """Append an error for every forbidden placeholder/blocked token found in the slice."""
    lowered = yaml.safe_dump(slice_block, default_flow_style=True).lower()
    for token in CANARY_SLICE_FORBIDDEN_TOKENS:
        if token in lowered:
            errors.append(f"canary_slice contains forbidden token {token!r}")


def validate_canary_slice(
    manifest_path: Path = REPO_ROOT / CAMPAIGN_MANIFEST_PATH,
) -> CanarySliceReport:
    """Validate the concrete, authorized issue #5783 canary slice.

    The slice must be authorized, pin a concrete policy (no TBD/blocked tokens), map one
    concrete Robot SF scenario to one SocNavBench scenario with a seed, name a cross-suite
    metric, and list limitation flags. Fails closed on any placeholder/blocked field.
    """
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    slice_block = manifest.get("canary_slice")
    errors: list[str] = []
    if not isinstance(slice_block, dict):
        return _canary_slice_missing_report(errors)

    status = slice_block.get("status")
    if status != "authorized_canary":
        errors.append(f"canary_slice.status must be 'authorized_canary', got {status!r}")
    if not slice_block.get("canary_authorized", False):
        errors.append("canary_slice.canary_authorized must be true")

    _check_forbidden_tokens(slice_block, errors)

    policy = slice_block.get("policy") or {}
    if not isinstance(policy, dict):
        errors.append("canary_slice.policy must be a mapping")
    policy_id = policy.get("policy_id")
    policy_version = policy.get("version")
    algo = policy.get("algo")
    algo_config = policy.get("algo_config")
    _check_concrete_fields(
        policy if isinstance(policy, dict) else {},
        (
            ("policy.policy_id", policy_id),
            ("policy.version", policy_version),
            ("policy.algo", algo),
            ("policy.algo_config", algo_config),
        ),
        errors,
    )

    mapping = slice_block.get("scenario_mapping") or {}
    if not isinstance(mapping, dict):
        errors.append("canary_slice.scenario_mapping must be a mapping")
    robot_sf_scenario_id = mapping.get("robot_sf_scenario_id")
    socnavbench_scenario_id = mapping.get("socnavbench_scenario_id")
    seed = mapping.get("seed")
    external_asset_id = mapping.get("external_asset_id")
    _check_concrete_fields(
        mapping if isinstance(mapping, dict) else {},
        (
            ("scenario_mapping.robot_sf_scenario_id", robot_sf_scenario_id),
            ("scenario_mapping.socnavbench_scenario_id", socnavbench_scenario_id),
            ("scenario_mapping.external_asset_id", external_asset_id),
        ),
        errors,
    )
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append("canary_slice.scenario_mapping.seed must be an integer")

    metric_id = slice_block.get("metric_id")
    _check_concrete_fields(slice_block, (("metric_id", metric_id),), errors)

    limitation_flags = tuple(
        mapping.get("limitation_flags") or slice_block.get("limitation_flags") or ()
    )
    if not limitation_flags:
        errors.append("canary_slice.limitation_flags must be a non-empty list")

    return CanarySliceReport(
        issue=int(slice_block.get("issue", 5783)),
        status=str(status) if status is not None else "missing",
        canary_authorized=bool(slice_block.get("canary_authorized", False)),
        ok=not errors,
        policy_id=policy_id if isinstance(policy_id, str) else None,
        policy_version=policy_version if isinstance(policy_version, str) else None,
        algo=algo if isinstance(algo, str) else None,
        algo_config=algo_config if isinstance(algo_config, str) else None,
        robot_sf_scenario_id=(
            robot_sf_scenario_id if isinstance(robot_sf_scenario_id, str) else None
        ),
        socnavbench_scenario_id=(
            socnavbench_scenario_id if isinstance(socnavbench_scenario_id, str) else None
        ),
        seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
        external_asset_id=external_asset_id if isinstance(external_asset_id, str) else None,
        metric_id=metric_id if isinstance(metric_id, str) else None,
        limitation_flags=limitation_flags,
        errors=errors,
    )


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
    parser.add_argument(
        "--validate-manifest",
        action="store_true",
        help="Validate the issue #3287 campaign-manifest scaffold before reporting readiness.",
    )
    parser.add_argument(
        "--validate-canary-slice",
        action="store_true",
        help=(
            "Validate the issue #5783 canary slice (concrete policy, scenario mapping, "
            "metric, and limitation flags). Fails closed on placeholder/blocked fields."
        ),
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
        if args.validate_canary_slice:
            report = validate_canary_slice(args.repo_root / CAMPAIGN_MANIFEST_PATH).to_dict()
            if not report["ok"]:
                raise CampaignManifestError("; ".join(report["errors"]))
        elif args.validate_manifest:
            validate_campaign_manifest(args.repo_root / CAMPAIGN_MANIFEST_PATH)
            report = evaluate_readiness(args.repo_root, waivers)
        else:
            report = evaluate_readiness(args.repo_root, waivers)
    except (WaiverError, CampaignManifestError) as exc:
        # Diagnostics go to stderr so stdout stays reserved for the text/JSON report.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.validate_canary_slice:
        status = "OK" if report["ok"] else "BLOCKED"
        print(
            f"canary slice {status}: policy={report['policy_id']}@{report['policy_version']} "
            f"seed={report['seed']} metric={report['metric_id']}"
        )
    else:
        print(render_text(report))
    if args.validate_canary_slice:
        return 0 if report["ok"] else 1
    return 0 if report["prerequisites_status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
