# Issue #4455 Closure Audit

Plain-language summary: merged PR #4458 delivered the public CPU-only
perception-degradation ladder enablement requested by issue #4455. The issue
should remain open for the later empirical campaign named by the maintainer
propagation comment. This audit is not benchmark evidence, and it does not close
or promote any ranking or failure-mode claim.

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/4455>
- Merged implementation PR: <https://github.com/ll7/robot_sf_ll7/pull/4458>
- Merge commit: `c002413ec7be6c3e89165f477c12fa5cffe55a41`
- Audit date: 2026-07-04

## Acceptance Mapping

| Acceptance criterion from #4455 | Delivered evidence | Status |
| --- | --- | --- |
| Degradation profiles are config-declared and validated. | PR #4458 added `configs/benchmarks/perception_degradation_profiles_v1.yaml`, `configs/benchmarks/perception_degradation/issue_4455_ladder_v1.yaml`, and validation in `scripts/benchmark/build_perception_degradation_ladder_issue_4455.py`; focused tests load and check profile hashes. | Delivered by merge commit `c002413ec7be6c3e89165f477c12fa5cffe55a41`. |
| Degradation applies only to planner input. | PR #4458 routes observation corruption through `robot_sf/benchmark/map_runner_episode.py` before planner invocation and records the diagnostic-only boundary in `docs/context/issue_4455_perception_degradation_ladder.md`. No simulator ground truth, collision, termination, or metric contract is promoted by this audit. | Delivered for the enablement slice; campaign evidence remains future work. |
| Gaussian noise, fixed delay, track dropout, and range-limited visibility are supported. | PR #4458 extended `robot_sf/benchmark/observation_noise.py` with `pedestrian_position_noise_std_m`, `observation_delay_steps`, existing false-negative drop support, and `pedestrian_occlusion_max_range_m`; `tests/benchmark/test_observation_noise.py` covers deterministic noise, delay-state behavior, dropout, range occlusion, and fail-closed invalid parameters. | Delivered by focused runtime tests in PR #4458. |
| Episode rows include profile key, profile hash, and counters. | PR #4458 records observation-noise metadata and stats in episode rows through `robot_sf/benchmark/map_runner_episode.py`, and the ladder builder emits per-profile configs that preserve profile key/hash identity. | Delivered for generated smoke configs; full campaign rows are not yet produced. |
| Pre-registration config pins planners, scenarios, seeds, and profile ladder. | PR #4458 added `configs/benchmarks/issue_4455_perception_degradation_ladder_preregistration.yaml` and `configs/benchmarks/perception_degradation/issue_4455_ladder_v1.yaml`, pinning representative hybrid, PPO, and ORCA planners, a scenario matrix, seeds `[111, 112]`, and the ordered five-profile ladder. | Delivered. |
| CPU smoke proves plumbing. | PR #4458 reported passing targeted tests plus ladder validation/generation commands in the PR body, including `tests/benchmark/test_observation_noise.py`, `tests/benchmark/test_perception_degradation_ladder_issue_4455.py`, and the ladder builder `--validate-only` / generated-config path. | Delivered for CPU-only plumbing; not benchmark result evidence. |
| No full campaign submission or claim edit included. | PR #4458 explicitly scoped itself to enablement only, left full benchmark execution to the parent issue, and changed no paper/dissertation claim text. This closure audit likewise performs no compute submission and no claim promotion. | Satisfied. |

## Residual Work

The live issue has a maintainer propagation comment after PR #4458:

> Enablement-only; campaign execution remains open under issue.

That remaining action is empirical, not a missing CPU-only implementation
criterion. A future authorized campaign should run the preregistered profile
ladder, retain provenance for generated rows, and synthesize whether planner
ranking and failure modes survive planner-input degradation. Until that run
exists, issue #4455 should remain open, and any ranking or failure-mode survival
statement should stay blocked.

## Local Verification

Audit-time validation for this docs-only slice:

```bash
uv run python scripts/dev/check_docs_evidence_integrity.py
```

No full benchmark campaign was run, no Slurm or GPU job was submitted, and no
paper or dissertation claim text was changed.
