# Issue #2912 Benchmark Row Claim v1 Enforcement

## Goal

Issue #2912 adds a row-level `BenchmarkClaim.v1` profile next to each static leaderboard row so
that benchmark tables and static summaries cannot imply unsupported claims.

## Decision

Every leaderboard page under `docs/leaderboards/` now has a machine-readable sidecar
`<page>.rows.json` that contains one `benchmark_row_claim.v1` record per Markdown row. This
record is a narrow row-level profile of the broader `benchmark_claim.v1` evidence boundary; it
uses row fields because static leaderboards do not carry full episode/provenance bundles inline.
The record enforces the fields required by the issue contract:

- `suite_id`
- `planner_id`
- `planner_mode` (`native`, `adapter`, `fallback`, `degraded`, `not_available`)
- `seeds`
- `metrics`
- `row_status`
- `exclusions`
- `artifact_uri`
- `claim_wording`
- `evidence_tier`
- `claim_boundary`

The validator is `robot_sf.benchmark.benchmark_row_claim`. It fails closed when:

- the payload does not match the JSON Schema (`benchmark_row_claim.v1`)
- `artifact_uri` points at a worktree-local `output/` path
- the referenced artifact does not exist
- `planner_mode` is `fallback`, `degraded`, or `not_available` but `row_status` claims success
- `claim_wording` uses ranking/superlative language above the row's evidence tier or status

## Accepted Example

```json
{
  "schema_version": "benchmark_row_claim.v1",
  "suite_id": "amv_actuation_smoke",
  "planner_id": "goal",
  "planner_mode": "native",
  "seeds": ["not_recorded"],
  "metrics": {
    "success": 0.0,
    "collision": 0.2,
    "near_miss": 1.0,
    "runtime_sec": 13.4813
  },
  "row_status": "successful_evidence",
  "exclusions": ["seeds_not_recorded_in_compact_summary", "synthetic_actuation_not_hardware_amv"],
  "artifact_uri": "docs/context/evidence/issue_1569_amv_actuation_smoke_2026-05-27/summary.json",
  "claim_wording": "Executable synthetic diagnostic row; no paper-facing AMV promotion claim.",
  "evidence_tier": "diagnostic",
  "claim_boundary": "Synthetic AMV actuation diagnostic only; not platform-class proxy or hardware-calibrated AMV evidence."
}
```

This is accepted because the claim wording matches the `diagnostic` evidence tier and the
`successful_evidence` row status, and the artifact URI points to a tracked repository file.

## Rejected Example

```json
{
  "schema_version": "benchmark_row_claim.v1",
  "suite_id": "amv_actuation_smoke",
  "planner_id": "orca",
  "planner_mode": "fallback",
  "seeds": ["not_recorded"],
  "metrics": {"success": 0.0},
  "row_status": "successful_evidence",
  "exclusions": ["fallback_execution"],
  "artifact_uri": "output/benchmarks/camera_ready/orca_episodes.jsonl",
  "claim_wording": "ORCA outperforms all baselines on AMV actuation.",
  "evidence_tier": "diagnostic",
  "claim_boundary": "Diagnostic fallback row only; no benchmark ranking claim."
}
```

This is rejected for four deterministic reasons:

1. `artifact_uri` starts with `output/` (worktree-local, not durable tracked evidence).
2. `planner_mode` is `fallback` but `row_status` is `successful_evidence`.
3. `claim_wording` contains the superlative `outperforms`, which is not allowed for a
   `diagnostic` tier row.
4. The wording makes a ranking/planner-ranking claim that is not supported by the diagnostic
   status.

## Validation Commands

Validate a single sidecar:

```bash
uv run python -m robot_sf.benchmark.cli validate-row-claims \
  --sidecar docs/leaderboards/smoke.rows.json
```

Validate all leaderboard sidecars:

```bash
uv run python -m robot_sf.benchmark.cli validate-row-claims --all
```

Run the contract tests:

```bash
uv run pytest tests/benchmark/test_benchmark_row_claim.py -q
```

## Related Surfaces

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/2912>
- Schema: `robot_sf/benchmark/schemas/benchmark_row_claim.v1.json`
- Validator/CLI: `robot_sf/benchmark/benchmark_row_claim.py`, `robot_sf/benchmark/cli.py`
- Leaderboard sidecars: `docs/leaderboards/*.rows.json`
- Leaderboard docs: `docs/leaderboards/README.md`
