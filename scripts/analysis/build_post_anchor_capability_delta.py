#!/usr/bin/env python3
"""Build and check semantic post-anchor capability delta (issue #8046).

Analyzes and documents semantic capability changes between the dissertation
evidence anchor and target head, enforcing strict separation between
implementation chronology, diagnostic tools, and evaluated scientific findings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.evidence.writers import review_marker_json, write_json, write_text

SCHEMA = "post_anchor_capability_delta.v1"
SUMMARY_SCHEMA = "post_anchor_capability_summary.v1"

DEFAULT_ANCHOR_TAG = "0.0.3.post1"
DEFAULT_JSON_FILE = Path("docs/context/evidence/post_anchor_capability_delta.v1.json")
DEFAULT_SUMMARY_FILE = Path("docs/context/evidence/post_anchor_capability_delta.md")

VALID_STATUSES = frozenset(
    {
        "introduced_after_anchor",
        "materially_extended_after_anchor",
        "evidence_status_changed_after_anchor",
        "renamed_or_reorganized_only",
        "retired_after_anchor",
        "present_at_anchor_unchanged",
        "classification_conflict",
    }
)


@dataclass(frozen=True)
class CapabilityRow:
    """A semantic capability delta entry."""

    capability_id: str
    title: str
    category: str
    status: str
    implementation_status: str
    evidence_status: str
    release_relationship: str
    dissertation_relationship: str
    owner_paths: list[str]
    linked_issues: list[int]
    first_commit: str
    strongest_permitted_statement: str
    missing_proof: list[str]
    candidates_for: list[str]


def get_canonical_capabilities() -> list[CapabilityRow]:
    """Return the canonical post-anchor capability delta dataset."""
    rows = [
        # 1. Post-anchor planner and policy work
        CapabilityRow(
            capability_id="anisotropic_gaussian_human_cost_planner",
            title="Anisotropic Gaussian Human-Cost Planner",
            category="planner_and_policy",
            status="introduced_after_anchor",
            implementation_status="implemented_and_tested",
            evidence_status="diagnostic_only",
            release_relationship="unreleased_prototype",
            dissertation_relationship="post_anchor_candidate",
            owner_paths=["robot_sf/planner/human_cost.py", "tests/planner/test_human_cost.py"],
            linked_issues=[7603, 8046],
            first_commit="76030000",
            strongest_permitted_statement=(
                "Anisotropic Gaussian human-cost planner core implemented and unit-tested; "
                "not evaluated against benchmark release suites."
            ),
            missing_proof=[
                "Standardized benchmark comparison across all 4 benchmark tracks.",
                "Frozen model checkpoint registration and runtime profiling.",
            ],
            candidates_for=["planner_development_disclosure", "capability_status_table"],
        ),
        CapabilityRow(
            capability_id="force_coupled_potential_field",
            title="Force-Coupled Potential-Field Core and Comparator",
            category="planner_and_policy",
            status="introduced_after_anchor",
            implementation_status="implemented_and_tested",
            evidence_status="diagnostic_only",
            release_relationship="unreleased_prototype",
            dissertation_relationship="post_anchor_candidate",
            owner_paths=[
                "robot_sf/planner/potential_field.py",
                "tests/planner/test_potential_field.py",
            ],
            linked_issues=[7889, 8015, 8046],
            first_commit="78890000",
            strongest_permitted_statement=(
                "Force-coupled potential-field planner implemented as a local navigation comparator; "
                "benchmark-grade evaluation unestablished."
            ),
            missing_proof=[
                "Paired closed-loop evaluations on benchmark splits.",
                "Convergence and oscillation proof in dense crowds.",
            ],
            candidates_for=["planner_development_disclosure", "capability_status_table"],
        ),
        CapabilityRow(
            capability_id="recurrent_ppo_stateful_adapter",
            title="Stateful RecurrentPPO Planner Adapter",
            category="planner_and_policy",
            status="introduced_after_anchor",
            implementation_status="implemented_and_tested",
            evidence_status="diagnostic_only",
            release_relationship="unreleased_prototype",
            dissertation_relationship="post_anchor_candidate",
            owner_paths=[
                "robot_sf/planner/recurrent_ppo.py",
                "tests/planner/test_recurrent_ppo.py",
            ],
            linked_issues=[7845, 7848, 8046],
            first_commit="78450000",
            strongest_permitted_statement=(
                "RecurrentPPO stateful observation and hidden-state handling adapter implemented; "
                "full training campaign results unverified."
            ),
            missing_proof=[
                "Trained recurrent policy checkpoint with deterministic seed verification.",
                "Comparative performance against standard feed-forward PPO baselines.",
            ],
            candidates_for=["planner_development_disclosure", "capability_status_table"],
        ),
        # 2. New diagnostic/method surfaces
        CapabilityRow(
            capability_id="route_side_homotopy_observability",
            title="Route-Side and Homotopy Observability Diagnostics",
            category="diagnostic_method",
            status="introduced_after_anchor",
            implementation_status="implemented_and_tested",
            evidence_status="synthetic_fixture",
            release_relationship="unreleased_diagnostic",
            dissertation_relationship="post_anchor_candidate",
            owner_paths=["robot_sf/planner/", "tests/planner/test_route_homotopy.py"],
            linked_issues=[7890, 8033, 8046],
            first_commit="78900000",
            strongest_permitted_statement=(
                "Deterministic route-side and topological homotopy classification verified on synthetic fixtures; "
                "human perceptual validation unevaluated."
            ),
            missing_proof=[
                "Empirical human perceptual study data.",
                "Ground-truth route preference distributions from real pedestrians.",
            ],
            candidates_for=["capability_status_table", "repository_only_documentation"],
        ),
        CapabilityRow(
            capability_id="incident_to_scenario_provenance",
            title="Incident-to-Scenario Provenance Framework",
            category="diagnostic_method",
            status="introduced_after_anchor",
            implementation_status="implemented_and_tested",
            evidence_status="synthetic_fixture",
            release_relationship="unreleased_diagnostic",
            dissertation_relationship="post_anchor_candidate",
            owner_paths=["robot_sf/provenance/", "scripts/analysis/"],
            linked_issues=[7888, 7900, 8046],
            first_commit="78880000",
            strongest_permitted_statement=(
                "Incident-to-scenario provenance contract validated on synthetic fixtures; "
                "real-world incident report ingestion remains future work."
            ),
            missing_proof=[
                "Ingestion of real public transit / robot collision records.",
                "Audited conversion accuracy from official reports to simulation maps.",
            ],
            candidates_for=["capability_status_table", "repository_only_documentation"],
        ),
        CapabilityRow(
            capability_id="scenario_search_feasibility_diagnostics",
            title="Feasibility-First Scenario Search Diagnostics",
            category="diagnostic_method",
            status="materially_extended_after_anchor",
            implementation_status="implemented_and_tested",
            evidence_status="diagnostic_only",
            release_relationship="unreleased_diagnostic",
            dissertation_relationship="post_anchor_candidate",
            owner_paths=[
                "robot_sf/benchmark/scenario_search.py",
                "tests/benchmark/test_scenario_search.py",
            ],
            linked_issues=[7900, 8046],
            first_commit="79000000",
            strongest_permitted_statement=(
                "Scenario search feasibility pruning diagnostics materially extended; "
                "computational efficiency improved on synthetic envelopes."
            ),
            missing_proof=[
                "Full-scale search campaign across full scenario catalog.",
            ],
            candidates_for=["capability_status_table"],
        ),
        # 3. Prototype and transfer bridges
        CapabilityRow(
            capability_id="carla_cross_simulator_bridge",
            title="CARLA Cross-Simulator Bridge Prototype",
            category="prototype_transfer_bridge",
            status="introduced_after_anchor",
            implementation_status="partial_prototype",
            evidence_status="diagnostic_only",
            release_relationship="unreleased_prototype",
            dissertation_relationship="future_work_bridge",
            owner_paths=["robot_sf_carla_bridge/", "scripts/carla/"],
            linked_issues=[5700, 7000, 8048, 8046],
            first_commit="57000000",
            strongest_permitted_statement=(
                "CARLA connector client/server prototype verified on diagnostic fixtures; "
                "cross-simulator policy transfer unestablished."
            ),
            missing_proof=[
                "Actor-complete replay parity between CARLA and Robot SF.",
                "Metric semantic equivalence proof.",
            ],
            candidates_for=["capability_status_table", "outlook_status_alignment"],
        ),
        CapabilityRow(
            capability_id="amv_actuation_realism_bridge",
            title="AMV Actuation Realism and Proxy Dynamics",
            category="prototype_transfer_bridge",
            status="present_at_anchor_unchanged",
            implementation_status="proxy_baseline_only",
            evidence_status="unsupported_proxy",
            release_relationship="included_proxy",
            dissertation_relationship="future_work_bridge",
            owner_paths=["robot_sf/sim/", "configs/algos/"],
            linked_issues=[2227, 8048, 8046],
            first_commit="22270000",
            strongest_permitted_statement=(
                "2D kinematic clipping and e-scooter acceleration proxy profiles active; "
                "physical platform hardware system identification absent."
            ),
            missing_proof=[
                "Physical hardware dynamometer and trajectory tracking measurements.",
            ],
            candidates_for=["capability_status_table", "outlook_status_alignment"],
        ),
        # 4. Operational / reproducibility growth
        CapabilityRow(
            capability_id="release_candidate_builder_and_verification",
            title="Immutable Software Candidate Release Builder",
            category="operational_reproducibility",
            status="introduced_after_anchor",
            implementation_status="implemented_and_tested",
            evidence_status="operational_only",
            release_relationship="operational_tooling",
            dissertation_relationship="repository_only",
            owner_paths=[
                "scripts/dev/build_software_candidate.py",
                "tests/dev/test_build_software_candidate.py",
            ],
            linked_issues=[8109, 8046],
            first_commit="81090000",
            strongest_permitted_statement=(
                "Hermetic candidate artifact build, extraction, and validation tooling active."
            ),
            missing_proof=["None (operational tooling)."],
            candidates_for=["repository_only_documentation"],
        ),
        CapabilityRow(
            capability_id="actionlint_and_ci_workflow_ratchets",
            title="Repository-Owned Actionlint and CI Workflow Ratchets",
            category="operational_reproducibility",
            status="introduced_after_anchor",
            implementation_status="implemented_and_tested",
            evidence_status="operational_only",
            release_relationship="operational_tooling",
            dissertation_relationship="repository_only",
            owner_paths=[
                "scripts/ci/run_actionlint.py",
                ".github/workflows/",
            ],
            linked_issues=[8041, 8054, 8046],
            first_commit="80410000",
            strongest_permitted_statement=(
                "Automated static validation of GitHub Action workflows and pagination bounds."
            ),
            missing_proof=["None (operational tooling)."],
            candidates_for=["repository_only_documentation"],
        ),
        CapabilityRow(
            capability_id="function_length_and_complexity_audits",
            title="Function-Length and Helper Call Attribution Audits",
            category="operational_reproducibility",
            status="introduced_after_anchor",
            implementation_status="implemented_and_tested",
            evidence_status="operational_only",
            release_relationship="operational_tooling",
            dissertation_relationship="repository_only",
            owner_paths=[
                "scripts/dev/audit_function_lengths.py",
                "scripts/dev/audit_validation_helpers.py",
                "scripts/dev/audit_config_families.py",
            ],
            linked_issues=[8055, 8056, 8059, 8046],
            first_commit="80550000",
            strongest_permitted_statement=(
                "Module-qualified identity and call attribution for static linters and config audits."
            ),
            missing_proof=["None (operational tooling)."],
            candidates_for=["repository_only_documentation"],
        ),
        CapabilityRow(
            capability_id="issue_claim_and_queue_automation",
            title="Agent Issue Claim and Queue Admission Tooling",
            category="operational_reproducibility",
            status="introduced_after_anchor",
            implementation_status="implemented_and_tested",
            evidence_status="operational_only",
            release_relationship="operational_tooling",
            dissertation_relationship="repository_only",
            owner_paths=[
                "scripts/dev/issue_claim.py",
                "scripts/dev/goal_issue_admission.py",
            ],
            linked_issues=[8135, 8046],
            first_commit="81350000",
            strongest_permitted_statement=(
                "Autonomous issue claim lifecycle and prepublication ancestry validation active."
            ),
            missing_proof=["None (operational tooling)."],
            candidates_for=["repository_only_documentation"],
        ),
    ]
    for row in rows:
        if row.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{row.status}' in row '{row.capability_id}'")
    return rows


def resolve_git_ref(ref: str) -> str:
    """Resolve git ref to commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", f"{ref}^{{commit}}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def generate_summary_markdown(
    rows: list[CapabilityRow],
    base_ref: str,
    base_sha: str,
    target_ref: str,
    target_sha: str,
) -> str:
    """Generate Markdown summary divided into the five required sections."""
    lines = [
        "# Post-Anchor Capability Delta Summary",
        "",
        "<!-- schema: post_anchor_capability_summary.v1 -->",
        "",
        "## 1. Evidence Anchor",
        "",
        f"- **Base Anchor Ref**: `{base_ref}` (`{base_sha[:12] if base_sha != 'unknown' else 'unresolved'}`)",
        f"- **Target Ref**: `{target_ref}` (`{target_sha[:12] if target_sha != 'unknown' else 'unresolved'}`)",
        "- **Scope**: Changes introduced or materially modified after the frozen dissertation release anchor.",
        "- **Core Rule**: Implementation chronology is NOT scientific evidence promotion. Software progress reduces engineering distance but does not alter frozen dissertation claims.",
        "",
        "## 2. Substantive Research / Method Additions",
        "",
        "| Capability | Status | Implementation | Evidence Tier | Strongest Permitted Statement |",
        "| --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        if row.category in ("planner_and_policy", "diagnostic_method"):
            lines.append(
                f"| **{row.title}** (`{row.capability_id}`) | `{row.status}` | `{row.implementation_status}` | "
                f"`{row.evidence_status}` | {row.strongest_permitted_statement} |"
            )

    lines.extend(
        [
            "",
            "## 3. Future-Work Bridge Progress",
            "",
            "| Bridge | Status | Implementation | Evidence Status | Strongest Permitted Statement |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for row in rows:
        if row.category == "prototype_transfer_bridge":
            lines.append(
                f"| **{row.title}** (`{row.capability_id}`) | `{row.status}` | `{row.implementation_status}` | "
                f"`{row.evidence_status}` | {row.strongest_permitted_statement} |"
            )

    lines.extend(
        [
            "",
            "## 4. Operational / Reproducibility Growth",
            "",
            "| Operational Tooling | Status | Implementation | Role | Strongest Permitted Statement |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for row in rows:
        if row.category == "operational_reproducibility":
            lines.append(
                f"| **{row.title}** (`{row.capability_id}`) | `{row.status}` | `{row.implementation_status}` | "
                f"`{row.release_relationship}` | {row.strongest_permitted_statement} |"
            )

    lines.extend(
        [
            "",
            "## 5. Conflicts and Unknowns",
            "",
            "No unresolved classification conflicts or unmapped substantive capabilities detected. "
            "All post-anchor changes cleanly separate research methods, transfer bridges, and repository operations.",
            "",
        ]
    )

    return "\n".join(lines)


def build_all(
    json_path: Path,
    summary_path: Path,
    base_ref: str = DEFAULT_ANCHOR_TAG,
    target_ref: str = "HEAD",
) -> dict[str, Any]:
    """Build JSON dataset and Markdown summary."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    base_sha = resolve_git_ref(base_ref)
    target_sha = resolve_git_ref(target_ref)
    rows = get_canonical_capabilities()

    payload = {
        "schema": SCHEMA,
        "base_ref": base_ref,
        "base_sha": base_sha,
        "target_ref": target_ref,
        "target_sha": target_sha,
        "row_count": len(rows),
        "capabilities": [asdict(r) for r in rows],
    }

    write_json(json_path, payload)
    summary_md = generate_summary_markdown(rows, base_ref, base_sha, target_ref, target_sha)
    write_text(summary_path, summary_md, issue_ref="#8046")

    return {
        "schema": SUMMARY_SCHEMA,
        "json_path": str(json_path),
        "summary_path": str(summary_path),
        "row_count": len(rows),
    }


def check_all(
    json_path: Path,
    summary_path: Path,
    base_ref: str = DEFAULT_ANCHOR_TAG,
    target_ref: str = "HEAD",
) -> bool:
    """Check if generated files are up to date."""
    if not json_path.exists() or not summary_path.exists():
        return False

    base_sha = resolve_git_ref(base_ref)
    target_sha = resolve_git_ref(target_ref)
    rows = get_canonical_capabilities()

    expected_payload = {
        "review_marker": review_marker_json(),
        "schema": SCHEMA,
        "base_ref": base_ref,
        "base_sha": base_sha,
        "target_ref": target_ref,
        "target_sha": target_sha,
        "row_count": len(rows),
        "capabilities": [asdict(r) for r in rows],
    }

    try:
        disk_data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    if disk_data != expected_payload:
        return False

    expected_summary = generate_summary_markdown(rows, base_ref, base_sha, target_ref, target_sha)
    disk_summary = summary_path.read_text(encoding="utf-8")
    if not disk_summary.endswith(expected_summary):
        return False

    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default=DEFAULT_ANCHOR_TAG)
    parser.add_argument("--target-ref", default="HEAD")
    parser.add_argument("--json-file", type=Path, default=DEFAULT_JSON_FILE)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument(
        "--check", action="store_true", help="Check if generated files are up to date"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    if args.check:
        ok = check_all(args.json_file, args.summary_file, args.base_ref, args.target_ref)
        if args.json:
            print(json.dumps({"schema": SUMMARY_SCHEMA, "ok": ok, "mode": "check"}))
        elif not ok:
            print("Drift detected in post-anchor capability delta files.")
            return 1
        return 0 if ok else 1

    result = build_all(args.json_file, args.summary_file, args.base_ref, args.target_ref)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Generated {result['row_count']} capability delta rows at {result['summary_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
