# Issue #1740 Actuation-Aware Learned-Policy Smoke Candidate

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1740>

Date: 2026-05-30

## Goal

Select the first concrete actuation-aware learned-policy smoke candidate before any implementation,
training, or benchmark campaign starts.

This is a candidate-selection note only. It does not run a smoke, train a policy, create benchmark
evidence, promote a checkpoint, or claim learned-policy superiority.

## Candidate Decision

Selected candidate: **`orca_residual_guarded_ppo_v0` ORCA-residual lineage smoke**.

Rationale:

- It has a concrete candidate surface in
  `configs/policy_search/candidates/orca_residual_guarded_ppo_v0.yaml`.
- It has a lineage packet in `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml`
  with an explicit runtime observation contract, residual bounds, hard-guard diagnostics, and
  fail-closed missing-artifact behavior.
- It has a documented preflight and smoke handoff in
  `docs/context/policy_search/SLURM/005_orca_residual_bc_lineage.md`.
- It is the only compared candidate that already expresses a learned-policy action relationship:
  bounded `policy_action - ORCA_action` residuals before the hard guard.
- Current evidence is still path-execution and lineage-readiness evidence only. The residual
  dataset and checkpoint pointers are pending, so the first follow-up must complete or validate
  that packet before any learned-policy or benchmark-strength claim.

## Candidate Comparison

| Candidate | Verdict | Reason |
|---|---|---|
| Actuation-penalized PPO fixture/config | Not first | Closest to the eventual AMV-specific learning hypothesis, but no actuation-penalized checkpoint, fixture config, or durable training dataset exists yet. |
| ORCA residual fixture | **Selected** | Concrete candidate YAML, lineage packet, local validator, smoke command, residual action contract, and hard-guard/fallback diagnostics already exist. Dataset/checkpoint artifacts remain the gate before learned-policy evidence. |
| Learned risk surface via `RiskSurfacePlannerAdapter` | Not first | Useful deterministic fixture and adapter diagnostic path, but it is a planner risk-surface producer rather than a learned-policy action head for this first smoke. It remains a good diagnostic-row candidate after the policy smoke is pinned down. |
| Planner arbitration | Not first | Needs durable labels, switching constraints, and hindsight-leakage controls before it is safe as a learned-policy smoke. Existing outcomes also make it a poor first bounded candidate. |
| Offline/sequence policy | Not first | Requires durable traces, schemas, and split provenance before any smoke or training path is meaningful. |

## Required Contract

Selected policy/family:

- `candidate_id`: `orca_residual_guarded_ppo_v0`
- Candidate config: `configs/policy_search/candidates/orca_residual_guarded_ppo_v0.yaml`
- Lineage packet: `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml`
- Handoff: `docs/context/policy_search/SLURM/005_orca_residual_bc_lineage.md`
- Compact contract evidence:
  `docs/context/evidence/issue_1428_orca_residual_lineage_2026-05-24/diagnostic_contract.json`

Observation contract:

- source: `runtime_socnav_struct`;
- required keys: `robot_state`, `goal_state`, `pedestrian_state`, and `occupancy_grid`;
- runtime-available state only;
- forbidden fields: scenario futures, benchmark labels, future oracle trajectories, and privileged
  map-solution features.

Action contract:

- residual action is `policy_action - ORCA_action`;
- residual is bounded before guarding: linear delta `0.25`, angular delta `0.35`;
- hard guard remains authoritative for evaluation;
- diagnostics must include `orca_command`, `raw_residual`, `bounded_residual`,
  `final_guarded_command`, `residual_clipping_rate`, `guard_veto_rate`, and
  `fallback_degraded_status`;
- fallback or degraded rows must not count as learned-policy success evidence.

Synthetic actuation slice:

- first local gate: ORCA-residual lineage packet validation and candidate smoke;
- AMV synthetic-actuation linkage should use
  `configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml` only after the residual
  packet has concrete dataset/checkpoint pointers or an explicitly deterministic fixture pointer;
- preferred first AMV scenario once that linkage exists: `classic_cross_trap_high`, because
  Issue #1569 identified it as the consensus hardest synthetic actuation scenario and it targets
  yaw/curvature deadlock.

Expected compact evidence:

- lineage preflight JSON from `validate_orca_residual_lineage_packet.py`;
- candidate smoke summary at
  `output/policy_search/orca_residual_guarded_ppo_v0/smoke/latest/summary.json`;
- compact diagnostic contract evidence for residual clipping, guard vetoes, and fallback/degraded
  status;
- a short interpretation that this is path-execution evidence only until durable residual dataset
  and checkpoint pointers are present.

## Fail-Closed Conditions

The first implementation/smoke issue must fail closed when:

- the teacher checkpoint, residual dataset, residual checkpoint, candidate YAML, or diagnostic
  contract path is missing when required;
- required observation keys are missing or forbidden fields are present;
- runtime inference is not deterministic enough for the smoke contract;
- kinematics/profile support is missing or mismatched;
- hard-guard diagnostics are absent;
- `fallback_degraded_status` is reported for rows proposed as positive learned-policy evidence;
- an AMV synthetic-actuation row cannot emit actuation diagnostics.

Fallback/degraded execution must not count as success evidence.

## First Command Shape

Implementation follow-up should first validate the existing lineage packet:

```bash
uv run python scripts/validation/validate_orca_residual_lineage_packet.py \
  --config configs/training/orca_residual/orca_residual_bc_issue_1428.yaml --json
```

Then run the existing candidate smoke only after the packet is concrete enough for the intended
claim:

```bash
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 uv run python \
  scripts/validation/run_policy_search_candidate.py --candidate orca_residual_guarded_ppo_v0 \
  --stage smoke --workers 1
```

If the packet still contains pending artifact pointers, the follow-up should complete the residual
dataset/checkpoint lineage before escalating to nominal sanity, AMV synthetic-actuation linkage, or
stress/full evaluation.

## Implementation Follow-Up

Canonical follow-up issue: <https://github.com/ll7/robot_sf_ll7/issues/1475>

Issue #1475 already owns the ORCA-residual smoke/nominal lineage gate for
`orca_residual_guarded_ppo_v0`: replace pending durable dataset, checkpoint, and diagnostic report
pointers, run bounded smoke first, and only consider nominal escalation after the smoke evidence is
concrete. Its non-claim boundary is:

- no benchmark campaign;
- no learned-policy superiority claim;
- no calibrated AMV claim before Issue #1559;
- no paper-facing benchmark claim;
- no fallback/degraded/unavailable rows counted as success.

## Validation

This selection was built from:

```bash
gh issue view 1740
sed -n '1,240p' docs/context/issue_1628_actuation_aware_learned_navigation.md
sed -n '1,220p' docs/context/issue_1685_dummy_learned_policy_adapter.md
sed -n '1,220p' docs/context/issue_1675_learned_risk_surface_interface.md
sed -n '1,180p' configs/policy_search/candidates/orca_residual_guarded_ppo_v0.yaml
sed -n '1,140p' docs/context/policy_search/SLURM/005_orca_residual_bc_lineage.md
sed -n '1,130p' configs/training/orca_residual/orca_residual_bc_issue_1428.yaml
rg -n "orca_residual_guarded_ppo_v0|run_policy_search_candidate|validate_orca_residual_lineage_packet|runtime_socnav_struct|fallback_degraded_rows_count_as_success" \
  configs docs scripts robot_sf tests -S
```

Validation for this docs-only change:

```bash
uv run python scripts/validation/validate_orca_residual_lineage_packet.py \
  --config configs/training/orca_residual/orca_residual_bc_issue_1428.yaml --json
BASE_REF=origin/issue-1628-actuation-aware-learned-nav \
  scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/issue-1628-actuation-aware-learned-nav...HEAD
```
