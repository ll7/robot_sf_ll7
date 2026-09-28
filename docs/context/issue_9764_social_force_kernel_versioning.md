# Issue #9764: social-force pair-kernel versioning

The social-force pair kernel has two versioned angle rules:

| Selector | Behavior | Resolution when omitted |
|---|---|---|
| `legacy_unwrapped_v1` | Preserve the historical raw `atan2` difference. | This remains the default. |
| `wrapped_v2` | Wrap the signed angle difference to `[-π, π)` before evaluating the anisotropic term. | Opt in explicitly. |

The selector is shared by the NumPy pairwise kernel in
`robot_sf/sim/pedestrian_model_variants.py` and the fast-pysf pedestrian kernel in
`fast-pysf/pysocialforce/forces.py`. Scenario overrides bind the simulator side;
`SocNavPlannerConfig` and the map-runner config boundary bind the planner side. Run metadata records
the selected version only when a selector was supplied, so legacy rows keep their prior metadata
shape.

`configs/algos/social_force_resolution_independent_v2_kernel_legacy_v1.yaml` and
`configs/algos/social_force_resolution_independent_v2_kernel_wrapped_v2.yaml` are matched diagnostic
configs. `configs/scenarios/classic_interactions_francis2023_goal_zone_entry_kernel_wrapped_v2.yaml`
is the corresponding simulator-side candidate matrix. The pinned 0.0.7 matrix
`configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml` remains byte-identical
to its published SHA-256 `03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c`.

## Release boundary

The 0.0.8 plan in #9668 requires the existing 0.0.7 metrics to reproduce and describes the new force
metrics and SNQI-v2 fields as the additions. The kernel correction can change simulation forces.
A distinct 0.0.8 candidate input,
`configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate.yaml`,
binds `wrapped_v2` in both the planner and scenario matrix. Its use as release evidence still
requires the #9668 row-equivalence gate; a mismatch is a stop condition. The paired kernel configs
above are diagnostic inputs and do not alter the frozen 0.0.7 input or claim
that the 0.0.8 equivalence gate passed. The separate #9758 pedestrian-contact question remains
unresolved by this angle correction.

## Evidence status

Focused regression tests cover both kernel implementations, the explicit selector path, legacy
default behavior, and metadata omission when the selector is absent. No general planner, pedestrian,
or dissertation claim follows from those tests.

## Separate #9758 recheck with `wrapped_v2` — 2026-09-28

The #9758 fix in merged PR #9788 is an explicit `surface_v3` pedestrian-term selector. To recheck
that contact behavior after the angle-wrap correction, the existing deterministic standing-pedestrian
and 1.3 m/s crossing rollouts now also run with both `surface_v3` and `wrapped_v2` selected. At both
`dt=0.1 s` and `dt=0.05 s`, the tests require swept clearance above 0.05 m; all four wrapped-kernel
cases pass. The focused kernel, planner, simulator, benchmark-registry, and metamorphic command
passes 180 tests with one strict xfail tracked under #9733. The exact code-tree run log is preserved
under `.git/codex-agent-runs/issue-9764-social-force-final/wrapped-kernel-recheck-5e197930/`.

This is a narrow compatibility recheck of those two fixtures, not the reviewer's complete six-case
probe set or evidence of general contact safety. The paired diagnostic below retains the legacy
pedestrian-force term. The candidate matrix selects `wrapped_v2` but does not select `surface_v3`;
this recheck does not change that release input or claim that the kernel wrap fixes the separate
pedestrian-force defect.

## Reproduced paired diagnostic — 2026-09-28

Earlier paired runs on seeds 111–113 used the reserved #9668 evaluation range. Their outputs remain
quarantined and uncited in the private host bundle `issue9764-social-force-kernel-paired-diagnostic-450ca42b`,
which carries a `QUARANTINED.md` marker. No values from those runs are used below. The corrected
103–105 record below uses the final checksummed rerun, not the earlier safe bundle.

Both selector arms ran on source commit `d2b6ac4e41c33290dfed477f1a1cb716009f07d2` (tree
`4bea8cfe50034a9d2cf2d81451b57382892f8283`) with the paired scenario and planner configs, seeds
103–105, 600-step horizon, 0.1 s step, one worker, recorded forces, and recorded simulation-step
traces. All 12 episode jobs completed without runner failures. The raw records verified the intended
kernel selector in both simulator and planner metadata, and all six pairs used the same map identity.

The exact legacy planner config is
`configs/algos/social_force_resolution_independent_v2_kernel_legacy_v1.yaml` (SHA-256
`82b769f2b4b19533339288aff1b8c02abea7427b708bd61bb520b5b069073c65`); its scenario config is
`configs/scenarios/issue_9764_kernel_paired_legacy_v1.yaml` (SHA-256
`142c15d3f61477644c754897bbae43732a7e7c2e1ea79b16f458307bc11c0aea`). The wrapped planner config
is `configs/algos/social_force_resolution_independent_v2_kernel_wrapped_v2.yaml` (SHA-256
`43e1e61d6fc50b22a95bbb698d612a63cf229052c043c1d2347c2d7271567c35`); its scenario config is
`configs/scenarios/issue_9764_kernel_paired_wrapped_v2.yaml` (SHA-256
`1014b8cbeaa529301128726637d92d2feb9fddde7baf604d43374172e11d6c62`).

The analyzer compares robot-attributable pedestrian-force vectors by the simulator's pedestrian
row index. A changed interaction is one paired vector whose L2 difference exceeds `1e-9 m/s²`.
The episode schema does not attach actor IDs to each force sample, so these counts are not attributed
to named pedestrians. Trajectory differences use aligned trace step/time and pedestrian actor IDs.
For pairs with different episode lengths, force and trajectory comparisons use the common prefix.

Across the six pairs, 555 of 4,898 compared robot-attributable force samples changed above the
threshold; 565 legacy samples and 560 wrapped samples were nonzero. The largest force-vector
difference was `0.580119708 m/s²`. Maximum same-index trajectory differences were `0.103352465 m`
for the robot and `0.675855495 m` for a pedestrian. Terminal outcomes matched in all six pairs: five
successes and one collision. The `classic_head_on_corridor_medium` seed 104 pair had 310 legacy
versus 309 wrapped trace/force samples; both ended in success and 309 common samples were compared.

| Scenario | Seed | Legacy outcome | Wrapped outcome | Changed force samples / compared | Robot max Δ (m) | Pedestrian max Δ (m) |
|---|---:|---|---|---:|---:|---:|
| `classic_group_crossing_medium` | 103 | success | success | 55 / 534 | 0.036253987 | 0.015236386 |
| `classic_group_crossing_medium` | 104 | success | success | 79 / 495 | 0.003812179 | 0.351632257 |
| `classic_group_crossing_medium` | 105 | collision | collision | 30 / 237 | 0.038425055 | 0.006943980 |
| `classic_head_on_corridor_medium` | 103 | success | success | 132 / 1,200 | 0.000062811 | 0.000041281 |
| `classic_head_on_corridor_medium` | 104 | success | success | 131 / 1,236 | 0.103352465 | 0.675855495 |
| `classic_head_on_corridor_medium` | 105 | success | success | 128 / 1,196 | 0.000019849 | 0.000043048 |

The checksummed private bundle is under
`.git/codex-agent-runs/issue-9764-social-force-final/paired-diagnostic-final-d2b6ac4e/`.
`SHA256SUMS` verified. The legacy and wrapped raw episode files have SHA-256
`4f425b031f6126b66c08d2ce6becacb428340232dc0adf297cbf287c4857720b` and
`d4f12e5f074d55083aa476eca088afa4d371ba81c073110552bb4083ec9ce15f`; the comparison report has
SHA-256 `f8fe389828e103f6e3e732b43cae0d224ee226ab5ec2f69dd08fc63631da12a2`, the run manifest has
`949db1eded2a758d3578b8bd0069cf14d19df6428a03d4c6e185f0ec03e5830e`, and the `SHA256SUMS` file
has `3f385cac7f6cf11647e34947f27685c47ee9b5cf65051661c418b679d4fd2411`. The bundle remains
host-local and classified as diagnostic evidence; it makes no release-row equivalence, general
planner-safety, or paper-facing claim.
