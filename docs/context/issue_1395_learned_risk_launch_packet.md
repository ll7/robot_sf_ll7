# Issue #1395 Learned Risk Model Launch Packet

Date: 2026-05-24

## Scope

This note records the pre-SLURM launch packet for `learned_risk_model_v1`. It does not train the
model, run stress/full-matrix evaluation, submit SLURM jobs, or promote a learned planner.

## Launch Packet

- Config: `configs/training/learned_risk_model_issue_1395_launch_packet.yaml`
- Validator: `scripts/validation/validate_learned_risk_launch_packet.py`
- SLURM handoff: `docs/context/policy_search/SLURM/001_learned_risk_model_v1.md`
- Candidate registry pointer: `docs/context/policy_search/candidate_registry.yaml`
- Evidence fixture:
  `docs/context/evidence/issue_1395_learned_risk_launch_packet_2026-05-24/`

## Slurm Execution Contract

The launch packet records the planned Slurm execution surface for issue #1472:
`slurm_execution.entrypoint`, `slurm_execution.config`, `slurm_execution.command`,
`slurm_execution.expected_output_root`, `slurm_execution.expected_log_path`, and
`slurm_execution.status_artifact_path`.

The validator requires those fields and rejects worktree-local `output/` paths for execution
artifacts. These fields are pre-submit contract metadata only; they do not launch training,
resolve private trace assets, or turn pending artifact aliases into evidence.

## Frozen Baseline

The launch packet freezes `hybrid_rule_v3_static_margin0_waypoint2` as the non-learning baseline
packet for first learned-risk comparison planning:

- Config: `configs/policy_search/candidates/hybrid_rule_v3_static_margin0_waypoint2.yaml`
- Scenario slices: `stress_slice`, `full_matrix`
- Seeds: `111`, `112`, `113`
- Source report:
  `docs/context/policy_search/reports/2026-04-30_best_non_learning_local_policy_report.md`

The compact `baseline_summary_stub.json` is only a preflight freeze record. Future training must
replace the pending durable artifact alias with concrete run artifacts before claiming results.

## Trace Contract

The validator requires each trace row to expose:

- `scenario_id`, `seed`, `candidate_id`, `termination_reason`,
- `metrics`, `trajectory_features`, and `labels`,
- feature inputs for clearance, local crowd distance, route progress, speed, and goal progress,
- labels for `collision`, `near_miss`, and `low_progress`.

Missing required trace fields fail closed. Worktree-local `output/` paths are rejected as frozen
artifacts.

## Safety Boundary

The learned model is explicitly auxiliary:

- hard guards remain authoritative,
- learned risk may only contribute an auxiliary cost term,
- required diagnostics include `learned_risk_score`, `hard_guard_decision`, and
  `auxiliary_cost_weight`.

## Validation

```bash
uv run python scripts/validation/validate_learned_risk_launch_packet.py \
  --config configs/training/learned_risk_model_issue_1395_launch_packet.yaml --json
```

Expected result: `status=valid`, candidate `learned_risk_model_v1`, frozen baseline
`hybrid_rule_v3_static_margin0_waypoint2`.

Targeted tests:

```bash
uv run pytest -q tests/training/test_learned_risk_launch_packet.py
```

## Follow-Up Boundary

The follow-up SLURM issue must record the exact branch/commit, materialized trace artifact URIs,
baseline artifact URI, local preflight result, submission command, stop gates, monitoring commands,
and accept/revise/reject criteria. It should not start until this packet is merged or the exact
collection/training commit is recorded.
