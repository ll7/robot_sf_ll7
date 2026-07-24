<!-- AI-GENERATED (robot_sf#6137, 2026-07-23) — NEEDS-REVIEW -->
# Issue #6137 — #5302 Oracle-Gap Campaign Materialization (No-Submit)

Plain-language summary: this packet records that the frozen issue #5302 oracle-gap
analysis contract has been **materialized** into a deterministic, config-first campaign
runner with a no-submit preflight, a one-cell canary, and a family-disjoint episode
matrix. It is pre-registration / materialization **infrastructure**, not a benchmark
result, planner ranking, or paper claim.

## Classification

- `evidence_tier`: `diagnostic-only` (materialization infrastructure)
- `result_classification`: `not_applicable` (no benchmark result)
- `status`: `materialization`
- Claim boundary: this packet promotes **no** paper, dissertation, leaderboard,
  record-breaking, universal-planner, or selector claim. It submits no Slurm/GPU compute.
  It changes none of the frozen scientific semantics (roster, metrics, thresholds, family
  split, decision rules). Any result remains diagnostic until a separate, compute-authorized
  native campaign produces rows that pass the report checks in the frozen packet.

## What this materializes

The runner at `scripts/analysis/run_issue_5302_oracle_gap_campaign.py` consumes the frozen
packet `configs/analysis/issue_5302_oracle_gap_packet.yaml` and the family-disjoint
partition `configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml` and
produces three deterministic modes:

| Mode | What it does | Compute |
| --- | --- | --- |
| `preflight` (default) | Resolves all six arms (real loaders), family partitions, every cell, the seed schedule, the campaign denominator, per-arm config/checkpoint hashes, and output paths. Writes `reports/preflight.json`. | None |
| `matrix` | Writes the family-disjoint episode matrix: identical (cell, seed) units across all six arms, selection/evaluation families disjoint. | None |
| `canary` | Materializes exactly one frozen scenario cell × one seed × all six arms. Emits one native row per arm through an **injected execution seam**; fails closed without a seam. | None by default (seam is injected) |

## Frozen contract preserved

- **Roster**: exactly the six approved arms (`orca`, `ppo`, `prediction_planner`,
  `scenario_adaptive_hybrid_orca_v1`, `prediction_mpc`,
  `hybrid_rule_v3_fast_progress_static_escape_continuous`). Arm identity is resolved through
  the real loaders (`campaign_arm_admission`), not hardcoded.
- **Split**: selection and evaluation family sets are disjoint (the packet's validity rule).
- **Denominator**: `cells × seeds × six arms`, identical across arms.
- **Canary**: one held-out evaluation cell × seed 111 × six arms.
- **Fail-closed**: missing checkpoints/configs, fallback/degraded execution, duplicate/missing
  arms, denominator mismatch, family leakage, and checkpoint-hash drift all fail closed.

## Canonical commands

```bash
# No-submit preflight (deterministic, submits nothing):
uv run python scripts/analysis/run_issue_5302_oracle_gap_campaign.py preflight
# Family-disjoint episode matrix:
uv run python scripts/analysis/run_issue_5302_oracle_gap_campaign.py matrix
# One-cell canary (materializes episodes; execution needs an injected seam):
uv run python scripts/analysis/run_issue_5302_oracle_gap_campaign.py canary
# Focused tests (real loaders for arm identity, injected seam for the canary):
uv run pytest tests/analysis/test_run_issue_5302_oracle_gap_campaign.py -q
```

## Relationship to the frozen packet

This materialization reuses — and does not modify — the frozen packet, the packet checker
(`scripts/validation/check_issue_5302_oracle_gap_packet.py`), the arm-admission gate
(`robot_sf/benchmark/campaign_arm_admission.py`), and the partition manifest. The native
campaign execution itself (running episodes) is out of scope for this PR and routes through
the ops queue in a separate, compute-authorized follow-up under parent issue #5302.

## Residual risks / follow-ups

- The canary's default CLI path materializes episodes only; the actual execution seam (local
  `run_episode` or an ops-queue submitter) is wired by a future compute-authorized issue.
- The runner resolves scenario cells from the partition's scenario matrices; if a future
  partition amendment changes the family or cell surface, the denominator changes with it.
