#!/usr/bin/env python3
"""Build and check planner-development funnel and selection trace (issue #8045).

Analyzes and exports the planner development pipeline, documenting stage
transitions from exploratory candidate generation to the frozen 14-arm release
campaign roster, while establishing strict separation between candidate exploration,
diagnostic tuning, and evaluated dissertation benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.evidence.writers import review_marker_json, write_json, write_text

SCHEMA = "planner_development_funnel.v1"
SUMMARY_SCHEMA = "planner_development_summary.v1"

DEFAULT_RELEASE_MANIFEST = Path(
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml"
)
DEFAULT_JSON_FILE = Path("docs/context/evidence/planner_development_funnel.v1.json")
DEFAULT_SUMMARY_FILE = Path("docs/context/evidence/planner_development_funnel.md")

CANONICAL_14_RELEASE_ROSTER = (
    "prediction_planner",
    "goal",
    "social_force",
    "orca",
    "ppo",
    "socnav_sampling",
    "sacadrl",
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
    "guarded_ppo",
    "predictive_mppi",
    "risk_dwa",
)

VALID_RELATIONSHIPS = frozenset(
    {
        "included_exact_key",
        "family_represented_by_successor",
        "diagnostic_only",
        "post_anchor",
        "blocked_or_unavailable",
        "not_relevant",
    }
)

VALID_SELECTION_BIAS_NOTES = frozenset(
    {
        "held_out_release_surface_proven",
        "partially_overlapping_surface_disclosed",
        "same_family_but_distinct_seeds_proven",
        "development_and_evaluation_separation_unknown",
        "not_applicable",
    }
)


@dataclass(frozen=True)
class PlannerCandidateRecord:
    """Detailed record of a planner candidate across development stages."""

    candidate_id: str
    display_name: str
    family: str
    introduced_stage: str
    highest_stage_reached: str
    evidence_status: str
    relationship_to_release: str
    selection_bias_note: str
    disposition: str
    disposition_reason: str
    evidence_pointer: str
    strongest_permitted_statement: str


def get_canonical_candidate_records() -> list[PlannerCandidateRecord]:
    """Return comprehensive list of planner candidates across development history."""
    records = [
        # --- 14 Release Roster Arms ---
        PlannerCandidateRecord(
            candidate_id="prediction_planner",
            display_name="Prediction MPC Planner",
            family="predictive_mpc",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="same_family_but_distinct_seeds_proven",
            disposition="promoted_to_release",
            disposition_reason="Primary predictive control candidate with dynamic obstacle forecasting.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="goal",
            display_name="Direct Goal Following",
            family="baseline_kinematic",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="held_out_release_surface_proven",
            disposition="promoted_to_release",
            disposition_reason="Null-avoidance lower baseline.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="social_force",
            display_name="Helbing-Molnar Social Force Model",
            family="force_field",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="held_out_release_surface_proven",
            disposition="promoted_to_release",
            disposition_reason="Canonical classical microscopic force baseline.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="orca",
            display_name="Optimal Reciprocal Collision Avoidance",
            family="velocity_obstacle",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="held_out_release_surface_proven",
            disposition="promoted_to_release",
            disposition_reason="Standard multi-agent reciprocal collision avoidance baseline.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="ppo",
            display_name="Feed-Forward PPO Policy",
            family="reinforcement_learning",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="same_family_but_distinct_seeds_proven",
            disposition="promoted_to_release",
            disposition_reason="Standard feed-forward model-free RL baseline.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="socnav_sampling",
            display_name="Social Navigation Sampling",
            family="sampling_based",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="held_out_release_surface_proven",
            disposition="promoted_to_release",
            disposition_reason="Trajectory rollout sampling baseline.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="sacadrl",
            display_name="SA-CADRL Collision Avoidance",
            family="reinforcement_learning",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="held_out_release_surface_proven",
            disposition="promoted_to_release",
            disposition_reason="Socially-aware value network baseline.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="scenario_adaptive_hybrid_orca_v1",
            display_name="Scenario-Adaptive Hybrid ORCA v1",
            family="hybrid_rule",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="partially_overlapping_surface_disclosed",
            disposition="promoted_to_release",
            disposition_reason="Adaptive switching between rule-based modes and reciprocal avoidance.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="scenario_adaptive_hybrid_orca_v2_collision_guard",
            display_name="Scenario-Adaptive Hybrid ORCA v2 (Collision Guard)",
            family="hybrid_rule",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="partially_overlapping_surface_disclosed",
            disposition="promoted_to_release",
            disposition_reason="Enhanced safety filter and emergency brake arbitration over v1.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="hybrid_rule_v3_fast_progress_static_escape",
            display_name="Hybrid Rule v3 Fast Progress (Discrete)",
            family="hybrid_rule",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="partially_overlapping_surface_disclosed",
            disposition="promoted_to_release",
            disposition_reason="High-efficiency rule arbitration with static obstacle escape maneuvers.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="hybrid_rule_v3_fast_progress_static_escape_continuous",
            display_name="Hybrid Rule v3 Fast Progress (Continuous)",
            family="hybrid_rule",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="partially_overlapping_surface_disclosed",
            disposition="promoted_to_release",
            disposition_reason="Continuous-action formulation of static escape and fast progress rules.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="guarded_ppo",
            display_name="Guarded PPO Policy",
            family="hybrid_learning",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="partially_overlapping_surface_disclosed",
            disposition="promoted_to_release",
            disposition_reason="Learned policy shielded by deterministic kinematic safety filter.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="predictive_mppi",
            display_name="Predictive MPPI Controller",
            family="sampling_based",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="same_family_but_distinct_seeds_proven",
            disposition="promoted_to_release",
            disposition_reason="Model Predictive Path Integral control using sampled trajectory perturbations.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        PlannerCandidateRecord(
            candidate_id="risk_dwa",
            display_name="Risk-Aware Dynamic Window Approach",
            family="dynamic_window",
            introduced_stage="candidate_generation",
            highest_stage_reached="release_campaign",
            evidence_status="release_evaluated",
            relationship_to_release="included_exact_key",
            selection_bias_note="partially_overlapping_surface_disclosed",
            disposition="promoted_to_release",
            disposition_reason="Risk-weighted dynamic window velocity evaluation successor.",
            evidence_pointer="configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml",
            strongest_permitted_statement="Evaluated across 20,160 rows of the frozen 0.0.3.post1 release campaign.",
        ),
        # --- Predecessors / Exploratory / Diagnostic Candidates ---
        PlannerCandidateRecord(
            candidate_id="dwa_classic",
            display_name="Classic Dynamic Window Approach",
            family="dynamic_window",
            introduced_stage="candidate_generation",
            highest_stage_reached="diagnostic_stress",
            evidence_status="diagnostic_only",
            relationship_to_release="family_represented_by_successor",
            selection_bias_note="development_and_evaluation_separation_unknown",
            disposition="superseded_by_successor",
            disposition_reason="Superseded by risk_dwa with dynamic obstacle risk scoring.",
            evidence_pointer="configs/algos/dwa_classic.yaml",
            strongest_permitted_statement="Diagnostic exploratory baseline; not included in frozen 14-arm roster.",
        ),
        PlannerCandidateRecord(
            candidate_id="chance_constrained_mpc_gmm",
            display_name="Chance-Constrained MPC (GMM)",
            family="predictive_mpc",
            introduced_stage="candidate_generation",
            highest_stage_reached="diagnostic_stress",
            evidence_status="diagnostic_only",
            relationship_to_release="diagnostic_only",
            selection_bias_note="not_applicable",
            disposition="retained_as_diagnostic",
            disposition_reason="High computational latency during dense crowd stress tests.",
            evidence_pointer="configs/algos/chance_constrained_mpc_gmm.yaml",
            strongest_permitted_statement="Diagnostic research candidate; latency bounds precluded release admission.",
        ),
        PlannerCandidateRecord(
            candidate_id="diffusion_policy_smoke",
            display_name="Diffusion Policy Adapter",
            family="generative_policy",
            introduced_stage="candidate_generation",
            highest_stage_reached="smoke_nominal",
            evidence_status="smoke_only",
            relationship_to_release="blocked_or_unavailable",
            selection_bias_note="not_applicable",
            disposition="blocked_unsupported_runtime",
            disposition_reason="Real-time execution latency exceeded benchmark timeout budget.",
            evidence_pointer="configs/algos/diffusion_policy_issue_4010_smoke.yaml",
            strongest_permitted_statement="Implementation prototype; real-time inference unviable on CPU benchmark lanes.",
        ),
        # --- Post-Anchor Planners (Strictly Demarcated) ---
        PlannerCandidateRecord(
            candidate_id="anisotropic_gaussian_human_cost_planner",
            display_name="Anisotropic Gaussian Human-Cost Planner",
            family="predictive_human_cost",
            introduced_stage="post_anchor_generation",
            highest_stage_reached="smoke_nominal",
            evidence_status="diagnostic_only",
            relationship_to_release="post_anchor",
            selection_bias_note="not_applicable",
            disposition="unreleased_prototype",
            disposition_reason="Introduced in PR #7603 after frozen 0.0.3.post1 dissertation release anchor.",
            evidence_pointer="docs/context/evidence/post_anchor_capability_delta.v1.json",
            strongest_permitted_statement="Post-anchor development prototype; strictly excluded from dissertation release roster.",
        ),
        PlannerCandidateRecord(
            candidate_id="force_coupled_potential_field",
            display_name="Force-Coupled Potential-Field Planner",
            family="potential_field",
            introduced_stage="post_anchor_generation",
            highest_stage_reached="smoke_nominal",
            evidence_status="diagnostic_only",
            relationship_to_release="post_anchor",
            selection_bias_note="not_applicable",
            disposition="unreleased_prototype",
            disposition_reason="Introduced in PR #7889 after frozen 0.0.3.post1 dissertation release anchor.",
            evidence_pointer="docs/context/evidence/post_anchor_capability_delta.v1.json",
            strongest_permitted_statement="Post-anchor development prototype; strictly excluded from dissertation release roster.",
        ),
        PlannerCandidateRecord(
            candidate_id="recurrent_ppo_stateful_adapter",
            display_name="Stateful RecurrentPPO Adapter",
            family="reinforcement_learning",
            introduced_stage="post_anchor_generation",
            highest_stage_reached="smoke_nominal",
            evidence_status="diagnostic_only",
            relationship_to_release="post_anchor",
            selection_bias_note="not_applicable",
            disposition="unreleased_prototype",
            disposition_reason="Introduced in PR #7845 after frozen 0.0.3.post1 dissertation release anchor.",
            evidence_pointer="docs/context/evidence/post_anchor_capability_delta.v1.json",
            strongest_permitted_statement="Post-anchor development prototype; strictly excluded from dissertation release roster.",
        ),
    ]

    for record in records:
        if record.relationship_to_release not in VALID_RELATIONSHIPS:
            raise ValueError(
                f"Invalid relationship '{record.relationship_to_release}' in {record.candidate_id}"
            )
        if record.selection_bias_note not in VALID_SELECTION_BIAS_NOTES:
            raise ValueError(
                f"Invalid selection bias note '{record.selection_bias_note}' in {record.candidate_id}"
            )
    return records


def generate_funnel_markdown(records: list[PlannerCandidateRecord]) -> str:
    """Generate Markdown report divided into required views."""
    release_arms = [r for r in records if r.relationship_to_release == "included_exact_key"]
    post_anchor_arms = [r for r in records if r.relationship_to_release == "post_anchor"]
    exploratory_arms = [
        r for r in records if r.relationship_to_release not in ("included_exact_key", "post_anchor")
    ]

    lines = [
        "# Planner Development Funnel and Selection Trace",
        "",
        "<!-- schema: planner_development_summary.v1 -->",
        "",
        "## 1. Dissertation-Facing Compact Funnel View",
        "",
        "This view defines the standard evidence tier transitions without conflating exploratory search with benchmark claims.",
        "",
        "| Stage | Purpose | Typical Proof | Admissible Conclusion | Separation from Final Campaign |",
        "| --- | --- | --- | --- | --- |",
        "| **1. Candidate Generation** | Explore navigation mechanisms and prototypes | Config / method card | Idea or implementation exists | Not evidence |",
        "| **2. Smoke & Nominal Sanity** | Reject broken or unviable candidates | Deterministic smoke test | Executable under bounded fixture | Not ranking evidence |",
        "| **3. Diagnostic & Stress Studies** | Identify mechanism failure modes | Diagnostic artifact / run logs | Mechanism-specific observation | Not pooled with release |",
        "| **4. Roster Freeze** | Fix exact planner configuration identities | Hash-pinned release manifest | Experiment definitions locked | Precedes release execution |",
        "| **5. Release Campaign** | Evaluate frozen roster on benchmark splits | 20,160-row release bundle | Authoritative dissertation results | Published result surface |",
        "",
        "## 2. Frozen 14-Arm Release Roster Trace",
        "",
        f"Verified against release manifest `paper_experiment_matrix_v2_h600_s30_release_v0_0_3_post1.yaml` (total {len(release_arms)} arms).",
        "",
        "| Key | Display Name | Family | Selection Trace | Selection Bias / Separation Accounting |",
        "| --- | --- | --- | --- | --- |",
    ]

    for r in release_arms:
        lines.append(
            f"| `{r.candidate_id}` | **{r.display_name}** | `{r.family}` | {r.disposition_reason} | `{r.selection_bias_note}` |"
        )

    lines.extend(
        [
            "",
            "## 3. Exploratory and Diagnostic Candidates (Predecessors / Exclusions)",
            "",
            "Candidates developed or evaluated during exploratory phases that did not enter the frozen release roster.",
            "",
            "| Candidate | Family | Highest Stage | Relationship | Disposition Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for r in exploratory_arms:
        lines.append(
            f"| `{r.candidate_id}` | `{r.family}` | `{r.highest_stage_reached}` | `{r.relationship_to_release}` | {r.disposition_reason} |"
        )

    lines.extend(
        [
            "",
            "## 4. Post-Anchor Candidate Demarcation",
            "",
            "Planners introduced after the frozen `0.0.3.post1` dissertation evidence anchor. These candidates are strictly segregated from the historical benchmark roster.",
            "",
            "| Post-Anchor Candidate | Family | Evidence Tier | Status |",
            "| --- | --- | --- | --- |",
        ]
    )

    for r in post_anchor_arms:
        lines.append(
            f"| `{r.candidate_id}` | `{r.family}` | `{r.evidence_status}` | `{r.relationship_to_release}` (Strictly excluded from release roster) |"
        )

    lines.extend(
        [
            "",
            "## 5. Methodological Separation Summary",
            "",
            "- **Prospective Freeze**: The 14 release planners were frozen prior to running the 20,160-episode release matrix.",
            "- **No Post-Hoc Roster Alteration**: Exploratory candidates (e.g. GMM chance-constrained MPC) and post-anchor planners (e.g. RecurrentPPO, human-cost Gaussian) are never backported into the dissertation release bundle.",
            "- **Honest Bias Accounting**: Scenarios shared between diagnostic tuning and evaluation are disclosed as `partially_overlapping_surface_disclosed` rather than claiming artificial prospective isolation.",
            "",
        ]
    )

    return "\n".join(lines)


def build_all(
    json_path: Path,
    summary_path: Path,
    manifest_path: Path = DEFAULT_RELEASE_MANIFEST,
) -> dict[str, Any]:
    """Build JSON dataset and Markdown summary."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records = get_canonical_candidate_records()

    # Verify release roster exactness
    release_keys = tuple(
        r.candidate_id for r in records if r.relationship_to_release == "included_exact_key"
    )
    if len(release_keys) != 14 or release_keys != CANONICAL_14_RELEASE_ROSTER:
        raise ValueError(f"Release roster must match exactly 14 canonical keys, got {release_keys}")

    payload = {
        "schema": SCHEMA,
        "release_roster_count": len(release_keys),
        "total_candidate_count": len(records),
        "release_roster_keys": list(release_keys),
        "candidates": [asdict(r) for r in records],
    }

    write_json(json_path, payload)
    summary_md = generate_funnel_markdown(records)
    write_text(summary_path, summary_md, issue_ref="#8045")

    return {
        "schema": SUMMARY_SCHEMA,
        "json_path": str(json_path),
        "summary_path": str(summary_path),
        "release_roster_count": len(release_keys),
        "total_candidate_count": len(records),
    }


def check_all(
    json_path: Path,
    summary_path: Path,
    manifest_path: Path = DEFAULT_RELEASE_MANIFEST,
) -> bool:
    """Check if generated files match canonical state."""
    if not json_path.exists() or not summary_path.exists():
        return False

    records = get_canonical_candidate_records()
    release_keys = tuple(
        r.candidate_id for r in records if r.relationship_to_release == "included_exact_key"
    )

    expected_payload = {
        "review_marker": review_marker_json(),
        "schema": SCHEMA,
        "release_roster_count": len(release_keys),
        "total_candidate_count": len(records),
        "release_roster_keys": list(release_keys),
        "candidates": [asdict(r) for r in records],
    }

    try:
        disk_data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    if disk_data != expected_payload:
        return False

    expected_summary = generate_funnel_markdown(records)
    disk_summary = summary_path.read_text(encoding="utf-8")
    if not disk_summary.endswith(expected_summary):
        return False

    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-file", type=Path, default=DEFAULT_JSON_FILE)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    parser.add_argument(
        "--check", action="store_true", help="Check if generated files are up to date"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    if args.check:
        ok = check_all(args.json_file, args.summary_file, args.manifest_file)
        if args.json:
            print(json.dumps({"schema": SUMMARY_SCHEMA, "ok": ok, "mode": "check"}))
        elif not ok:
            print("Drift detected in planner development funnel files.")
            return 1
        return 0 if ok else 1

    result = build_all(args.json_file, args.summary_file, args.manifest_file)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Generated planner development funnel at {result['summary_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
