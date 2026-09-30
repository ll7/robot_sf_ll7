# PPOABL diagnostic plant ablations

Never use these branches for release or publication evidence. Each branch descends
from PPOEVAL V3 (`19a8b03229fe3f1742f78a377940bf9ef48c9e92`). A5 uses the
V2 comparator (`fec4090aa09c307bfff823dfbd5dad482da8b9d0`) policy and plant
behavior, relaxing only the plant speed cap.

| Branch suffix | Comparator | Plant speed m/s | Linear acceleration m/s² | Reverse | Angular acceleration rad/s² |
| --- | --- | ---: | --- | --- | --- |
| a1 | V3 | 2 | instantaneous | allowed | instantaneous |
| a2 | V3 | 3 | 1 (including braking) | allowed | instantaneous |
| a3 | V3 | 3 | instantaneous | forbidden | instantaneous |
| a4 | V3 | 3 | instantaneous | allowed | 1 |
| a5 | V2 | 3 | 1 (including braking) | forbidden | 1 |

All retain policy maximum bounds 3 m/s and 1 rad/s and registry-bound
velocity-delta semantics. A1-A4 retain V3 signed proposals; A5 retains V2
nonnegative policy clipping. A3 forbids reverse at the plant only. The guard
and its rollout settings are unchanged. The two PPO arms retain their original,
different checkpoints; comparisons are within arm.

The unchanged runner `scripts/benchmark/run_ppoeval.py` enforces the unchanged
16-scenario subset, dev seeds 1001-1010, both arms (320 episodes), H600/dt0.1,
clean exact source head, checkpoint checksums, fresh output IDs and complete
native traces. Diagnostic acquisition completion retains canonical campaign
failure/admission status separately. Guard best-effort outcomes are collected,
not admitted as successful benchmark evidence. Local smoke uses one PPO episode,
head-on corridor, seed 1001; one parent plus one worker, sequential variants.

`OMP_NUM_THREADS=1 uv run --no-sync pytest -n0 tests/benchmark/test_ppoabl.py -q`
checks actual episode-context wiring for every scenario and both arms, one-factor
settings differences, and independent numeric plant transitions (forward/reverse
saturation, braking, linear ramps, angular ramps/reversals at dt 0.1 and 0.2).
It also checks this branch's arm profile binding, unchanged policy bounds/delta
semantics, and top-level `algo` comparison identity. Before implementation,
all 10 variant/arm isolation cases failed on their intended physical factor and
the comparison fixture failed on a duplicate PPO identity.

Test-value-gate answers: behavior protected is single-factor plant isolation,
branch profile selection, signed velocity-delta proposal bounds, and distinct
arm aggregation. Credible regressions are restoring both acceleration limits,
using hard-coded V3 caps, binding a profile to the wrong variant, changing policy
clipping or semantics, and grouping by underlying policy metadata. Existing
`tests/differential_drive_test.py` and `tests/baselines/test_ppo_action_semantics.py`
do not exercise diagnostic campaign plant wiring or guarded-arm comparison.
Tests use real context and plant code; no production test-only seams.

Stage `scripts/benchmark/ppoabl.sbatch`; supply lowercase variant and full head
SHA. It requires that SHA to equal the variant branch's remote tip, uses a fresh
job-owned clone, checks both profile selectors, and runs preflight plus the
campaign at 4 CPUs / 32G / 2h. Results, receipts, manifests, traces and checksums
remain under `~/ppoabl_results/ppoabl-<variant>-<jobid>`. No job has been
submitted by preparing this packet. Preserve/copy these outputs before compute
access ends.

Use `scripts/analysis/compare_ppoeval.py --run a1=ROOT ... --output REPORT.json`.
It accepts V1-V3 and A1-A5 and requires top-level `algo` for arm identity.
Requested-command clipping diagnostics continue to use the shared [0,2] release
interval, counterfactually for all plants where that differs. Local smoke and
unit tests establish readiness, not attribution of the V3-to-V2 outcome drop.
