"""Cross-module pipeline contract for the issue #4142 dense DPCBF comparison.

The dense-comparison pipeline landed as three separate slices, each with its own module,
schema constant, and focused test file:

- PR #4299 -- readiness preflight (:mod:`robot_sf.benchmark.issue_4142_dpcbf_dense_readiness`),
- PR #4318 -- packet-consuming run planner (:mod:`robot_sf.benchmark.issue_4142_dpcbf_dense_runner`),
- PR #4345 -- plan-consuming summarizer (:mod:`robot_sf.benchmark.issue_4142_dpcbf_dense_summary`).

Each slice's own test file exercises that module plus readiness, but no single test drives all
three *top-level* entry points (``evaluate_readiness`` -> ``build_run_plan`` ->
``summarize_dense_comparison``) as one unit or pins the invariants that must hold *across* the
slices. This module is that consolidation capstone. It guards the two fragmentation risks a
multi-PR pipeline is exposed to:

1. **Schema-lineage drift.** The packet YAML on disk, the readiness ``PACKET_SCHEMA_VERSION``,
   the plan ``PLAN_SCHEMA_VERSION``, and the summary ``SUMMARY_SCHEMA_VERSION`` must stay a
   single coherent lineage: packet ``.comparison.v1`` -> plan ``.comparison_plan.v1`` ->
   summary ``.comparison_summary.v1``. A later edit that bumps one without the others would
   silently break the chain.
2. **Vocabulary re-derivation.** The required-arms tuple and the fail-closed excluded-row
   statuses have a single canonical owner (readiness). The runner and summarizer must reuse
   *that object*, not re-hardcode a parallel copy that could drift.

Claim boundary: this is a diagnostic contract/regression guard. It asserts the pipeline's
schema and vocabulary stay internally consistent and stay fail-closed/execution-gated. It runs
no episodes, authorizes no campaign, and makes no safety-performance or collision-reduction
claim.
"""

from __future__ import annotations

import pathlib

import pytest

from robot_sf.benchmark import issue_4142_dpcbf_dense_readiness as readiness_mod
from robot_sf.benchmark import issue_4142_dpcbf_dense_runner as runner_mod
from robot_sf.benchmark import issue_4142_dpcbf_dense_summary as summary_mod
from robot_sf.benchmark.issue_4142_dpcbf_dense_readiness import (
    PACKET_PATH,
    PACKET_SCHEMA_VERSION,
    REQUIRED_ARMS,
    REQUIRED_EXCLUDED_ROW_STATUSES,
    evaluate_readiness,
)
from robot_sf.benchmark.issue_4142_dpcbf_dense_runner import (
    PLAN_SCHEMA_VERSION,
    DenseComparisonExecutionGatedError,
    build_run_plan,
    execute_run_plan,
)
from robot_sf.benchmark.issue_4142_dpcbf_dense_summary import (
    SUMMARY_SCHEMA_VERSION,
    summarize_dense_comparison,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_schema_versions_form_a_single_v1_lineage() -> None:
    """Packet, plan, and summary schema versions are a coherent ``...comparison*.v1`` lineage."""
    # The packet is the anchor; readiness owns its expected version.
    assert PACKET_SCHEMA_VERSION == "robot_sf.issue_4142_dpcbf_dense_comparison.v1"
    assert PLAN_SCHEMA_VERSION == "robot_sf.issue_4142_dpcbf_dense_comparison_plan.v1"
    assert SUMMARY_SCHEMA_VERSION == "robot_sf.issue_4142_dpcbf_dense_comparison_summary.v1"

    # All four output-contract schemas are distinct; nothing is accidentally reused.
    all_versions = {
        readiness_mod.SCHEMA_VERSION,
        PACKET_SCHEMA_VERSION,
        PLAN_SCHEMA_VERSION,
        SUMMARY_SCHEMA_VERSION,
    }
    assert len(all_versions) == 4

    # Plan and summary derive from the packet lineage, so both carry the shared stem.
    stem = PACKET_SCHEMA_VERSION.removesuffix(".v1")
    assert PLAN_SCHEMA_VERSION == f"{stem}_plan.v1"
    assert SUMMARY_SCHEMA_VERSION == f"{stem}_summary.v1"


def test_runner_reuses_the_readiness_packet_schema_constant() -> None:
    """The runner does not re-declare the packet schema; it imports the canonical one."""
    assert runner_mod.PACKET_SCHEMA_VERSION == PACKET_SCHEMA_VERSION


def test_required_arms_and_exclusions_have_one_shared_owner() -> None:
    """Runner and summarizer reuse readiness's arm/exclusion vocabulary objects, not copies.

    Object-identity (``is``) is deliberate: it fails closed if a future edit re-hardcodes a
    parallel list in one module, which is exactly the drift a multi-slice pipeline risks.
    """
    assert runner_mod.REQUIRED_ARMS is REQUIRED_ARMS
    assert summary_mod.REQUIRED_ARMS is REQUIRED_ARMS
    assert runner_mod.REQUIRED_EXCLUDED_ROW_STATUSES is REQUIRED_EXCLUDED_ROW_STATUSES

    # The tracked packet still declares exactly these three arms, in this order.
    assert REQUIRED_ARMS == (
        "cbf_off",
        "cbf_collision_cone_on",
        "cbf_dynamic_parabolic_v1_on",
    )


def test_end_to_end_real_packet_chains_readiness_plan_summary() -> None:
    """Drive all three public entry points on the tracked packet and pin the shared contract.

    This is the only test that runs readiness -> run plan -> summary as one unit and asserts
    the hand-offs agree: readiness status flows into the plan, the plan schema flows into the
    summary, the same three arms appear at every stage in packet order, and the fail-closed
    exclusion is carried verbatim from packet to plan.
    """
    readiness = evaluate_readiness(repo_root=REPO_ROOT, packet_path=PACKET_PATH)
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH)
    summary = summarize_dense_comparison(repo_root=REPO_ROOT, packet_path=PACKET_PATH)

    # Stage 1 -> 2: readiness is inputs-ready and its status is threaded into the plan.
    assert readiness.status == "inputs_ready_campaign_gated"
    assert readiness.inputs_ready is True
    assert plan.status == "plan_ready_campaign_gated"
    assert plan.is_executable_in_principle is True
    assert plan.readiness_status == readiness.status
    assert plan.packet_schema_version == PACKET_SCHEMA_VERSION

    # Stage 2 -> 3: the plan schema the summary reports is exactly the runner's plan schema.
    assert summary.plan_schema_version == plan.schema_version == PLAN_SCHEMA_VERSION
    assert summary.schema_version == SUMMARY_SCHEMA_VERSION

    # Same three arms, same order, at every stage.
    assert tuple(job.arm_key for job in plan.arms) == REQUIRED_ARMS
    assert tuple(arm.arm_key for arm in summary.arms) == REQUIRED_ARMS
    assert tuple(arm.arm_key for arm in readiness.arms) == REQUIRED_ARMS

    # Fail-closed exclusion is carried verbatim from packet into the plan, never weakened.
    assert set(REQUIRED_EXCLUDED_ROW_STATUSES).issubset(set(plan.excluded_row_statuses))
    assert plan.fallback_rows_are_success_evidence is False
    assert plan.fallback_excluded is True


def test_end_to_end_summary_is_gated_incomplete_until_execution() -> None:
    """With no authorized run, the summary is ``results_incomplete`` and never ``complete``.

    Running episodes is out of scope for every slice, so no arm artifacts exist on a clean
    tree. The summary must report that as incomplete coverage (a caveat state), not as a
    successful comparison.
    """
    summary = summarize_dense_comparison(repo_root=REPO_ROOT, packet_path=PACKET_PATH)
    assert summary.status == "results_incomplete"
    # Every required arm is present in the summary but flagged as missing its artifact.
    assert {arm.arm_key for arm in summary.arms} == set(REQUIRED_ARMS)
    assert all(not arm.artifact_present for arm in summary.arms)
    assert all(arm.success_evidence_rows == 0 for arm in summary.arms)


def test_execution_stays_authorization_gated_across_the_pipeline() -> None:
    """A fully resolved, ready plan still cannot execute -- the execution gate is absolute."""
    plan = build_run_plan(repo_root=REPO_ROOT, packet_path=PACKET_PATH)
    assert plan.is_executable_in_principle is True
    with pytest.raises(DenseComparisonExecutionGatedError):
        execute_run_plan(plan)
    # The runner's own gates are surfaced so ``plan_ready`` is never mistaken for a go-ahead.
    assert plan.runner_gates
    # Readiness campaign gates propagate unchanged into the plan.
    assert plan.campaign_gates == readiness_mod.CAMPAIGN_GATES
    assert plan.campaign_gates
