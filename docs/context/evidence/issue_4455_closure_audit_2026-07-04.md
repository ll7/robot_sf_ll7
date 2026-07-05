# Issue #4455 Closure And Integration Audit

Plain-language summary: merged PR #4458 delivered the public, CPU-only
perception-degradation ladder enablement requested by issue #4455. The issue
should remain open because the empirical ladder campaign and ranking/failure-mode
synthesis have not run. This audit is diagnostic-only documentation, not
benchmark evidence, and it does not promote any paper or dissertation claim.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/4455>
- Merged implementation PR: <https://github.com/ll7/robot_sf_ll7/pull/4458>
- Implementation merge commit: `c002413ec7be6c3e89165f477c12fa5cffe55a41`
- Merged closure-audit PR: <https://github.com/ll7/robot_sf_ll7/pull/4506>
- Closure-audit merge commit: `01b820995906be0f7c9fca90106e9b737715e568`
- Audit dates: 2026-07-04 closure audit; 2026-07-05 integration refresh

## Acceptance Mapping

| Acceptance criterion from #4455 | Delivered evidence | Status |
| --- | --- | --- |
| Degradation profiles config-declared validated. | PR #4458 added `configs/benchmarks/perception_degradation_profiles_v1.yaml`, `configs/benchmarks/perception_degradation/issue_4455_ladder_v1.yaml`, validation in `scripts/benchmark/build_perception_degradation_ladder_issue_4455.py`; focused tests load and check profile hashes. | Delivered by merge commit `c002413ec7be6c3e89165f477c12fa5cffe55a41`. |
| Degradation applies only planner input. | PR #4458 routes observation corruption through `robot_sf/benchmark/map_runner_episode.py` before planner invocation and records the diagnostic-only boundary in `docs/context/issue_4455_perception_degradation_ladder.md`. Simulator ground truth, collision detection, termination logic, and metric computation are not promoted by this audit. | Delivered for the enablement slice; campaign evidence remains future work. |
| Gaussian noise, fixed delay, track dropout, range-limited visibility supported. | PR #4458 extended `robot_sf/benchmark/observation_noise.py` with `pedestrian_position_noise_std_m`, `observation_delay_steps`, existing false-negative drop support, and `pedestrian_occlusion_max_range_m`; `tests/benchmark/test_observation_noise.py` covers deterministic noise, delay-state behavior, dropout, range occlusion, and fail-closed invalid parameters. | Delivered by focused runtime tests in PR #4458. |
| Episode rows include profile key, profile hash, counters. | PR #4458 records observation-noise metadata and stats in episode rows through `robot_sf/benchmark/map_runner_episode.py`, and the ladder builder emits per-profile configs that preserve profile key/hash identity. | Delivered for generated smoke configs; full campaign rows remain future evidence. |
| Pre-registration config pins planners, scenarios, seeds, profile ladder. | PR #4458 added `configs/benchmarks/issue_4455_perception_degradation_ladder_preregistration.yaml` and `configs/benchmarks/perception_degradation/issue_4455_ladder_v1.yaml` with hybrid/PPO/ORCA planners, planner-sanity scenario matrix, fixed seeds, and ordered profiles. | Delivered. |
| CPU smoke proves plumbing. | PR #4458 reported passing targeted tests plus ladder validation/generation commands in the PR body, including `tests/benchmark/test_observation_noise.py`, `tests/benchmark/test_perception_degradation_ladder_issue_4455.py`, ladder builder `--validate-only`, and generated-config path. | Delivered CPU-only plumbing; not benchmark result evidence. |
| No full campaign submission or claim edit included. | PR #4458 explicitly scoped itself enablement only, left full benchmark execution under the parent issue, and changed no paper/dissertation claim text. This closure audit likewise performs no compute submission and no claim promotion. | Satisfied. |

## Integration Status

The implementation contract is coherent across the merged slices:

| Slice | Merge commit | Integration role |
| --- | --- | --- |
| PR #4458 perception-degradation ladder enablement | `c002413ec7be6c3e89165f477c12fa5cffe55a41` | Adds schema/config, planner-input observation degradation, preregistration files, builder/checker, docs, and focused tests. |
| PR #4506 closure-audit evidence | `01b820995906be0f7c9fca90106e9b737715e568` | Records criterion-to-evidence mapping and keeps the issue open because empirical campaign execution remains pending. |

No new public schema fields, fail-closed blockers, or compatibility impacts are
introduced by this integration refresh. No transient target host, queue-routing,
or packet-lineage state is tracked here.

## Residual Work

The live issue maintainer propagation comment after PR #4458 says:

> Enablement-only; campaign execution remains open under this issue.

The remaining action is empirical, not another CPU-only guardrail. A future
authorized campaign should run the preregistered profile ladder, retain
provenance for generated rows, and synthesize whether planner ranking and
failure modes survive planner-input degradation. Until that run exists, issue
#4455 should remain open, and any ranking or failure-mode survival statement
should stay blocked.

Next empirical action: run generated configs from
`scripts/benchmark/build_perception_degradation_ladder_issue_4455.py --out-dir`
under an authorized campaign lane, then publish a diagnostic synthesis with
fallback/degraded exclusions. This PR does not submit that campaign.

## Local Verification

Audit-time validation docs-only slice:

```bash
uv run python scripts/dev/check_docs_evidence_integrity.py
```

No full benchmark campaign was run, no Slurm or GPU job was submitted, and no
paper or dissertation claim text was changed.
