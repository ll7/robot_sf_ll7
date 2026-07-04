# Issue #4359 Collision-Similarity Durable Artifact Report

Evidence status: diagnostic-only analysis artifact. This directory records representative
`collision-scenario-similarity` reports against existing tracked evidence artifacts. It does not
promote a planner, benchmark, paper, or dissertation claim.

## Inputs

- `durable_issue1470_similarity_report.*` uses
  `docs/context/evidence/issue_1470_oracle_imitation_traces_12911_2026-06-17/oracle_imitation/issue1470_train_candidate_traces/train__hybrid_rule_v3_static_margin0_waypoint2__combined.jsonl`.
  The report selects two collision records and records singleton representative groups,
  nearest-neighbor examples, available external labels, and available trajectory-derived benchmark
  metric fields for those selected records.
- `labeled_trace_fixture_validation_report.*` uses
  `docs/context/evidence/issue_1395_learned_risk_launch_packet_2026-05-24/trace_contract_fixture.jsonl`.
  The report selects one near-miss record and records external-label plus raw trajectory-field
  availability where the fixture supports it.

## Reproduction

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run robot_sf_bench collision-scenario-similarity \
  --episodes-jsonl docs/context/evidence/issue_1470_oracle_imitation_traces_12911_2026-06-17/oracle_imitation/issue1470_train_candidate_traces/train__hybrid_rule_v3_static_margin0_waypoint2__combined.jsonl \
  --out-json docs/context/evidence/issue_4359_collision_similarity_durable_artifact_2026-07-04/durable_issue1470_similarity_report.json \
  --out-markdown docs/context/evidence/issue_4359_collision_similarity_durable_artifact_2026-07-04/durable_issue1470_similarity_report.md \
  --nearest-k 1 \
  --group-threshold 0.35

scripts/dev/run_worktree_shared_venv.sh -- uv run robot_sf_bench collision-scenario-similarity \
  --episodes-jsonl docs/context/evidence/issue_1395_learned_risk_launch_packet_2026-05-24/trace_contract_fixture.jsonl \
  --out-json docs/context/evidence/issue_4359_collision_similarity_durable_artifact_2026-07-04/labeled_trace_fixture_validation_report.json \
  --out-markdown docs/context/evidence/issue_4359_collision_similarity_durable_artifact_2026-07-04/labeled_trace_fixture_validation_report.md \
  --nearest-k 1 \
  --group-threshold 0.35
```

## Boundary

Similarity distances are local to the input artifact descriptor fields. External labels, raw
trajectory fields, and trajectory-derived metric fields are reported as descriptive validation
context only; they are not treated as benchmark truth or a new metric.

`validation_boundary_after_pr4410.md` records the integrated boundary after PRs #4371, #4386, and
#4410: durable collision-similarity metrics remain diagnostic-only, and replay-level validation is
still blocked until a durable source provides raw per-step trajectory arrays for the same records.
