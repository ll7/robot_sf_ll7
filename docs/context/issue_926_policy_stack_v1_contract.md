# Issue #926: `policy_stack_v1` Minimal Contract

## Goal

Issue #926 is the first bounded slice under the larger #871 `policy_stack_v1` portfolio-planner
epic. It defines the minimum contract a future implementation must satisfy before any code claims a
new planner family is runnable or benchmark-ready.

This note is a design and scope boundary only. It does not add a planner entrypoint, benchmark row,
training run, or runtime behavior.

Runtime update: issue #1004 adds the first narrow `policy_stack_v1` entry point and records its
validation in
[Issue #1004 Policy Stack V1 Runtime](issue_1004_policy_stack_v1_runtime.md). This note remains the
contract and claim boundary for later #871 children.

## V1 Boundary

`policy_stack_v1` should start as a non-learning portfolio over existing in-repo proposal
generators. The initial implementation should prove the common proposal, risk scoring, shield, and
diagnostics contracts before adding learned scorers or new external planners.

Mandatory v1 surfaces:

- config-first planner entry under `configs/algos/`,
- common proposal payload with command, optional short trajectory, source planner key, provenance,
  availability status, and rejection reason,
- deterministic risk-score payload with progress, clearance, TTC/deadlock, command feasibility,
  and shield intervention fields,
- hard fail-closed behavior for invalid route/subgoal handoff and mandatory proposal failures,
- episode diagnostics that record selected proposal, rejected proposals, shield decisions, and
  degraded or unavailable proposal families.

Out of scope for v1:

- training a learned ranking model,
- CARLA or simulator-transfer logic,
- source-harness claims for external planner families,
- treating missing optional planners as successful fallback,
- paper-facing promotion before representative scenario proof exists.

## Proposal Availability Semantics

Use explicit statuses instead of implicit fallback:

| Status | Meaning | Benchmark interpretation |
|---|---|---|
| `native` | Proposal produced by an in-repo planner with its normal dependency surface. | Candidate may be scored. |
| `adapter` | Proposal produced through an adapter with explicit kinematics/provenance metadata. | Candidate may be scored, with caveat. |
| `unavailable` | Required dependency or artifact is missing before proposal generation. | Exclude from scoring; not a fallback success. |
| `rejected` | Proposal was generated but failed risk or shield checks. | Keep diagnostics; do not execute. |
| `degraded` | Proposal ran with a documented reduced contract. | Experimental only; never paper-facing by default. |

For the first runtime PR, prefer a small mandatory set that is already in-tree and dependency-light,
for example `goal` plus one local testing-only proposal. ORCA/HRVO/PPO/MPPI-style proposals should
remain optional until each proposal path has explicit dependency and action-contract checks.

## Diagnostics Contract

Each step-level diagnostic record should be JSON-compatible and include:

- `selected_proposal_key`,
- `selected_mode` (`native`, `adapter`, or `degraded`),
- `candidate_count`,
- `rejected_count`,
- `unavailable_count`,
- `shield_intervened`,
- `risk_score_components`,
- `rejection_reasons` keyed by proposal source.

The diagnostic contract should be additive to existing benchmark episode records. If the diagnostics
are too large for default output, store a bounded summary in the episode record and write detailed
per-step diagnostics to an optional sidecar artifact.

## Benchmark Claim Boundary

Until a runtime implementation passes representative scenario proof through the normal
policy-analysis or benchmark workflow, the only safe claim is:

> `policy_stack_v1` is a planned experimental portfolio contract over existing planner proposals.

After the first smoke implementation, the safe claim should still be experimental unless it proves:

- route/subgoal handoff does not dominate outcomes,
- proposal availability is explicit,
- missing optional planners do not silently become fallback success,
- diagnostics explain proposal selection and shield interventions,
- output labels distinguish native, adapter, rejected, degraded, and unavailable modes.

## Validation Path For Future Implementation

The first runtime PR should include:

```bash
uv run pytest tests/test_planner/test_policy_stack_v1.py -q
uv run python scripts/validation/run_policy_search_candidate.py --policy policy_stack_v1 ...
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

The exact smoke command should use a verified-simple scenario first, then one topology-heavy atomic
scenario after the route/subgoal contract is proven.

## Related Surfaces

- Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/871>
- Planner readiness matrix: [docs/benchmark_planner_family_coverage.md](../benchmark_planner_family_coverage.md)
- Planner-zoo context: [docs/ai/planner_zoo_context.md](../ai/planner_zoo_context.md)
- Existing experimental planner configs: `configs/algos/`
- Existing planner modules: `robot_sf/planner/`
