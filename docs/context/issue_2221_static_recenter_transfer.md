# Issue #2221 Static Recenter Transfer Smoke

Issue: [#2221](https://github.com/ll7/robot_sf_ll7/issues/2221)
Date: 2026-06-04
Status: diagnostic transfer smoke; classified `slice_local`.

## Goal

Test whether the static-recentering mechanism that helped on the Issue #2180 one-factor h500
discovery slice transfers to the Issue #2128 held-out scenario-family pilot slice.

This note applies the protocol from
[issue_2232_planner_mechanism_transfer_benchmark.md](issue_2232_planner_mechanism_transfer_benchmark.md).
It does not make an OOD, paper-facing, or broad planner-causality claim.

## Frozen Scope

- Transfer axis: held-out scenario family.
- Baseline: `hybrid_rule_v3_fast_progress`.
- Mechanism row: `issue_2170_static_recenter_only`.
- Funnel config:
  `configs/policy_search/transfer/issue_2221_static_recenter_heldout_smoke.yaml`.
- Scenario matrix:
  `configs/scenarios/sets/issue_2128_heldout_family_transfer_pilot_eval.yaml`.
- Partition manifest:
  `configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml`.
- Seed list: `111`.
- Horizon: `500`.

## Result

Static recentering did not improve the held-out pilot slice over the matched base row.

| Candidate | Episodes | Success | Collision | Near Miss | Terminations | Failure mode |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `hybrid_rule_v3_fast_progress` | 2 | 0.5000 | 0.0000 | 0.5000 | `max_steps: 1`, `success: 1` | `near_miss_intrusive: 1` |
| `issue_2170_static_recenter_only` | 2 | 0.5000 | 0.0000 | 0.5000 | `max_steps: 1`, `success: 1` | `near_miss_intrusive: 1` |

Per-row outcomes were also unchanged:

| Scenario | Seed | Base outcome | Recenter outcome | Safety delta |
| --- | ---: | --- | --- | --- |
| `classic_station_platform_medium` | 111 | `max_steps`, 60 near misses, 0 collisions | `max_steps`, 60 near misses, 0 collisions | no change |
| `francis2023_intersection_wait` | 111 | success, 0 near misses, 0 collisions | success, 0 near misses, 0 collisions | no change |

## Classification

Classification: `slice_local`.

Rationale: the Issue #2232 support rule requires a beneficial held-out success or low-progress
delta without worsening collision or near-miss rate. The held-out smoke preserved the baseline
success, collision, near-miss, termination, and per-scenario outcomes exactly, so it does not
support promoting static recentering beyond the Issue #2180 discovery slice.

Confidence: about 0.75 for "no visible terminal-metric transfer lift on this pilot slice"; much
lower for any broader conclusion because the proof has only two episodes and one seed.

## Evidence

- Compact evidence:
  [evidence/issue_2221_static_recenter_transfer_2026-06-04/comparison_summary.json](evidence/issue_2221_static_recenter_transfer_2026-06-04/comparison_summary.json)
- Evidence manifest:
  [evidence/issue_2221_static_recenter_transfer_2026-06-04/manifest.md](evidence/issue_2221_static_recenter_transfer_2026-06-04/manifest.md)
- Baseline report:
  [policy_search/reports/2026-06-04_hybrid_rule_v3_fast_progress_full_matrix.md](policy_search/reports/2026-06-04_hybrid_rule_v3_fast_progress_full_matrix.md)
- Mechanism report:
  [policy_search/reports/2026-06-04_issue_2170_static_recenter_only_full_matrix.md](policy_search/reports/2026-06-04_issue_2170_static_recenter_only_full_matrix.md)

## Implication

Static recentering remains a useful local component from the Issue #2180 discovery slice, but this
held-out smoke makes it a lower-priority transfer candidate. The next research lane should favor a
mechanism with a clearer held-out hypothesis, such as topology-hypothesis planning in issue #2223,
or a broader pre-registered seed/family expansion if maintainers specifically want to challenge
this negative pilot.

## Validation

Executed from the `issue-2221-static-recentering-transfer` worktree with the shared-venv wrapper:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_fast_progress --stage full_matrix --funnel-config configs/policy_search/transfer/issue_2221_static_recenter_heldout_smoke.yaml --output-dir output/policy_search/issue2221/hybrid_rule_v3_fast_progress/heldout_smoke --workers 1 --allow-expensive-stage
```

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/run_policy_search_candidate.py --candidate issue_2170_static_recenter_only --stage full_matrix --funnel-config configs/policy_search/transfer/issue_2221_static_recenter_heldout_smoke.yaml --output-dir output/policy_search/issue2221/issue_2170_static_recenter_only/heldout_smoke --workers 1 --allow-expensive-stage
```
