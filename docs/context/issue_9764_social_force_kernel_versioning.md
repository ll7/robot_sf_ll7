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
cases pass. The full focused set for this worktree passes 98 tests.

This is a narrow compatibility recheck of those two fixtures, not the reviewer's complete six-case
probe set or evidence of general contact safety. The #9764 paired diagnostic above used the default
legacy pedestrian term and recorded one collision in each kernel arm. Its referenced private bundle
was not present in the current task artifact directory, so those raw rows were not reanalyzed here.
The candidate matrix selects `wrapped_v2` but does not select `surface_v3`; this recheck does not
change that release input or claim that the kernel wrap fixes the separate pedestrian-force defect.

## Quarantine and replacement diagnostic — 2026-09-28

The earlier paired runs on seeds 111–113 used the reserved #9668 evaluation range. Their episode
records, comparison, interaction counts, and #9758 recheck are **quarantined and uncited**. They
remain in the private host bundle `issue9764-social-force-kernel-paired-diagnostic-450ca42b` for
custody only, with a `QUARANTINED.md` marker. No report should cite those values. The FIX review
classified this as a process slip, with no #9668 exception needed after replacement.

The paired diagnostic input files now select seeds 103–105 for the same two scenarios. The rerun
uses the committed source `cb5399fcedcb8b968f3ea74fe6b24fd5d16eeda8`, 600 steps, 0.1 s
step time, one worker, recorded forces and simulation-step traces. Its private bundle key is
`issue9764-social-force-kernel-paired-diagnostic-safe-103-105-cb5399fc`.

Both arms completed six episodes, all with terminal status and the expected selector and source
commit in each row. The paired analyzer verified equal scenario and metric parameters apart from
selector-dependent provenance, matched map identity, aligned trace times, and six matching terminal
outcomes. Five pairs ended in success in both arms; one pair ended in collision in both arms. The
largest same-index recorded robot or pedestrian position difference over common trace prefixes was
0.675855495 m. Several metrics changed. These observations are bounded to these six pairs.

The private bundle contains the raw episode rows, provenance receipts, logs, analyzer, comparison,
manifest, and `SHA256SUMS` for all ten files. The legacy and wrapped episode files have SHA-256
`ef231fb80c23b24a1e01f2d80ec4661c93687e2fd37ab91c4d098ad6f5b79a21` and
`4b139f40bd5a3bf6ac1683d989e04643e4dfbf0366f31e437a48b2f8410bb02e`; the comparison
has SHA-256 `e5cd2d275608a33f5dbc165e861f912e276e8118291e7b9e6a954fadf2bd0a6c`.
`sha256sum -c SHA256SUMS` passed for the new bundle. This remains diagnostic-only, with no
release-row equivalence or paper-facing claim.
