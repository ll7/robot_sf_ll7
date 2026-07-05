# Issue #4455 Perception-Degradation Ladder

Issue: [#4455](https://github.com/ll7/robot_sf_ll7/issues/4455)

Plain-language summary: this note records the enablement contract for a bounded
perception-degradation ladder requested by external dissertation review. Current
benchmark campaigns use oracle-grade planner input; the preregistered profiles
let a later campaign test whether planner ranking and failure modes survive
degraded planner input.

## Claim Boundary

This is an enablement and preregistration surface only. It is not benchmark
evidence, not a full campaign result, and not a paper or dissertation claim.

All degradation profiles apply to planner input only. Simulator ground truth,
collision detection, termination logic, trajectory recording, and metric
computation remain unchanged. Episode rows retain normalized
`observation_noise` block, `observation_noise_hash`, and
`observation_noise_stats` counter provenance.

## Profile Table

The canonical profile catalog is
`configs/benchmarks/perception_degradation_profiles_v1.yaml`.

| Profile | Hash | Planner-input perturbation |
| --- | --- | --- |
| `nominal_oracle` | `0a5609f0b2b1` | No-op oracle-grade baseline |
| `gaussian_track_noise_low` | `fc9abd0c2c88` | Gaussian pedestrian-track coordinate noise |
| `fixed_detection_delay_1` | `55e7ff4d05c0` | One-step pedestrian-observation delay |
| `false_negative_pedestrian_drop_10pct` | `c76cf85c757d` | 10% pedestrian track dropout |
| `occlusion_range_4m` | `600086b5b447` | Hide pedestrian tracks outside 4m range |

The preregistered ladder manifest is
`configs/benchmarks/perception_degradation/issue_4455_ladder_v1.yaml`; the
maintainer-requested wrapper path is
`configs/benchmarks/issue_4455_perception_degradation_ladder_preregistration.yaml`.

## Current Integration Status

Implementation and closure-audit slices have merged:

| Source | Status | Evidence boundary |
| --- | --- | --- |
| PR #4458, merge commit `c002413ec7be6c3e89165f477c12fa5cffe55a41` | Delivered enablement | Profile schema/configs, planner-input degradation behavior, preregistration, builder/checker, focused tests. |
| PR #4506, merge commit `01b820995906be0f7c9fca90106e9b737715e568` | Delivered audit note | Criterion-to-evidence mapping only; issue remains open for empirical campaign execution. |

The parent issue is not closable yet because the ladder campaign and
ranking/failure-mode synthesis have not run. That residual requires authorized
compute or an explicit decision to park the empirical result as unavailable. No
target host, queue-routing state, or private packet lineage is recorded in this
tracked note.

## Handoff

Use
`scripts/benchmark/build_perception_degradation_ladder_issue_4455.py --validate-only`
for a cheap profile/config check. Use the same script with `--out-dir` to
generate one camera-ready CPU smoke config per profile under local `output/`.
Those generated configs are disposable local validation artifacts unless a later
campaign promotes provenance.

Private-queue Slurm submission is intentionally out of scope for this note. The
next empirical action remains under issue #4455: run generated configs or a
capacity-appropriate campaign, then synthesize whether rank stability and
failure-mode stability survive the degradation ladder.
