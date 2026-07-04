# Issue #4455 Perception-Degradation Ladder

Issue: [#4455](https://github.com/ll7/robot_sf_ll7/issues/4455)

This note records the enablement contract for a bounded perception-degradation
ladder requested after external dissertation review. The current benchmark
campaigns use oracle-grade planner input; this slice pre-registers diagnostic,
non-calibrated stress profiles so a later campaign can measure whether planner
ranking and failure modes survive degraded planner input.

## Claim Boundary

This is an enablement and pre-registration surface only. It is not benchmark
evidence, not a full campaign result, and not a paper or dissertation claim.

All degradation profiles apply to planner input only. Simulator ground truth,
collision detection, termination logic, trajectory recording, and metric
computation remain unchanged. Episode rows retain the normalized
`observation_noise` block, `observation_noise_hash`, and
`observation_noise_stats` counters for provenance.

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

## Handoff

Use
`scripts/benchmark/build_perception_degradation_ladder_issue_4455.py --validate-only`
for a cheap profile/config check. Use the same script with `--out-dir` to
generate one camera-ready CPU smoke config per profile under local `output/`.
Those generated configs are disposable local validation artifacts unless a later
campaign promotes them with provenance.

Private-queue or Slurm submission is intentionally out of scope for this PR.
The next empirical action remains under issue #4455: run the generated configs
or a capacity-appropriate campaign, then synthesize whether rank stability and
failure-mode stability survive the degradation ladder.
