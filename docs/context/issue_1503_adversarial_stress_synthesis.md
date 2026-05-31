# Issue #1503 Adversarial Stress-Coverage Synthesis (2026-05-31)

Date: 2026-05-31

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1503>
- <https://github.com/ll7/robot_sf_ll7/issues/1502>
- <https://github.com/ll7/robot_sf_ll7/issues/1501>
- <https://github.com/ll7/robot_sf_ll7/issues/1488>
- <https://github.com/ll7/robot_sf_ll7/issues/691>

## Scope

This note analyzes the compact, tracked #1502 evidence only. It does not run new adversarial
search, hydrate raw local `output/` bundles, or promote the result to paper-facing benchmark
evidence.

Evidence root:

`docs/context/evidence/issue_1502_adversarial_two_family_2026-05-31/`

Inputs:

- `row_status_summary.json`
- `archive.json`
- `goal_sampler_comparison.json`
- `orca_sampler_comparison.json`
- `classic_head_on_corridor_guided_summary.json`
- `classic_head_on_corridor_guided_report.md`
- `checksums.sha256`

The source execution was Slurm job `12664`, which completed successfully on `a30` at commit
`d4a49b26772f72ad082a62d1485611c22034a6d9`. The tracked evidence landed through PR #1777.

## Coverage Summary

The campaign covered two families, three search surfaces, and explicit unavailable rows:

| Family | Search surface | Covered rows | Explicit exclusions |
|---|---:|---:|
| `crossing_ttc` | random and Optuna/TPE candidate search for `goal` and `orca` | 4 | guided route search is not applicable to parametric CandidateSpec rows |
| `classic_head_on_corridor` | guided route search for `classic_global_theta_star` | 1 | random and Optuna/TPE CandidateSpec search are not applicable to route-level optimization |

Aggregate row accounting from `row_status_summary.json`:

| Status | Count | Interpretation |
|---|---:|---|
| `valid_failure` | 60 | replayable crossing/TTC collision candidates archived as failure evidence |
| `valid_non_failure` | 228 | valid candidate executions that did not fail |
| `invalid_candidate` | 736 | invalid crossing/TTC candidates; sampler-quality signal, not success evidence |
| `valid_route_trial` | 40 | feasible head-on route-search trials |
| `failed_trial` | 60 | infeasible head-on route-search trials, all `invalid_start_or_goal` |
| `not_available` | 3 | intentionally unsupported family/search combinations |

No fallback or degraded rows are counted as success evidence.

## Failure Diversity

The failure archive reports 60 crossing/TTC failures from 1024 source candidates and 4 source
manifests. The archived failures form two clusters:

| Cluster | Policy | Failure | Members | Representative |
|---|---|---|---:|---|
| `cluster_0000` | `goal` | collision / collision termination | 31 | `failure_0015` |
| `cluster_0001` | `orca` | collision / collision termination | 29 | `failure_0045` |

This is useful evidence that the run can find replayable collision failures for two planner rows
under a fixed crossing/TTC template. It is not yet diverse across failure mechanisms: both clusters
share the same template and collision failure type. The next campaign should not claim broad
stress coverage until it adds at least one additional replay-confirmed mechanism, family, or
route-level failure category.

## Search Efficiency

For crossing/TTC, random search produced more useful failures at this budget:

| Policy | Sampler | Attempts | Valid candidates | Invalid candidates | Valid failures | Failure / attempt | Failure / valid |
|---|---|---:|---:|---:|---:|---:|---:|
| `goal` | random | 256 | 134 | 122 | 26 | 10.2% | 19.4% |
| `goal` | Optuna/TPE | 256 | 10 | 246 | 5 | 2.0% | 50.0% |
| `orca` | random | 256 | 134 | 122 | 24 | 9.4% | 17.9% |
| `orca` | Optuna/TPE | 256 | 10 | 246 | 5 | 2.0% | 50.0% |

Optuna/TPE had a higher failure rate among its valid candidates, but it generated very few valid
candidates. At this bounded budget, the invalid-candidate rate dominates, so random search is the
better failure-discovery baseline for crossing/TTC unless the Optuna search space or constraints
are repaired.

Best objective values also favored random for both planner rows:

- `goal`: random `25`, Optuna/TPE `21`
- `orca`: random `24`, Optuna/TPE `13`

For `classic_head_on_corridor`, the guided route search found 40 feasible route trials out of 100.
Its top score was `0.387602`; the report decomposes that into failure proxy `0.222222`, delay proxy
`0.550396`, path inefficiency `0.000022`, and near-miss stress `1.0`. These values are promising
as route-stress discovery diagnostics, but they are not directly comparable to crossing/TTC
failure counts because the search object and validity filters differ.

## Replay And Determinism

The archive entries preserve replay commands for crossing/TTC failure cases. That satisfies the
Issue #1503 minimum for replayable compact evidence, but it does not prove replay determinism because
Issue #1502 did not run a separate replay-repeat sweep.

Current status:

- replay command provenance: present for archived crossing/TTC failures;
- deterministic re-execution proof: not run;
- head-on route override provenance: compact summary and route override path recorded, but raw
  route override YAML remains local-only under ignored `output/`;
- checksum coverage: compact tracked evidence is checksummed in `checksums.sha256`.

Before paper-facing claims, run a small replay-determinism gate that replays representative
failures from each archive cluster and the best head-on route override from a durable input path.

## Claim Boundary

Supported:

- #1502 completed the bounded two-family execution and explicit row accounting.
- Crossing/TTC random search outperformed Optuna/TPE in failures per attempted candidate at the
  fixed 256-candidate budget.
- Optuna/TPE currently suffers from very high invalid-candidate rates in this search space.
- The archive contains replay-command-backed crossing/TTC collision failures for both `goal` and
  `orca`.
- Guided route search can produce feasible head-on corridor stress routes under the compact budget.

Not supported:

- paper-facing benchmark improvement claims;
- direct absolute comparison between crossing/TTC failure counts and head-on route-search trial
  counts;
- broad failure-mode diversity beyond collision under the crossing/TTC template;
- replay determinism or durable raw-artifact recovery without an additional replay gate;
- treating invalid, failed-trial, fallback, degraded, or not-available rows as success evidence.

## Recommendation

Close #1503 as a completed analysis child after this synthesis lands, then update #1488 with two
follow-up gates:

1. Repair or constrain the crossing/TTC Optuna/TPE search space before treating it as a useful
   adversarial sampler.
2. Add a compact replay-determinism gate over representative archived failures and the best
   head-on route before any paper-facing stress-coverage language.

The next execution child should expand mechanism diversity rather than merely increasing the same
crossing/TTC budget.
