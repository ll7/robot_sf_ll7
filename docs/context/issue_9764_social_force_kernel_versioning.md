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
is the corresponding simulator-side diagnostic matrix. The pinned 0.0.7 matrix
`configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml` remains byte-identical
to its published SHA-256 `03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c`.

## Release boundary

The 0.0.8 plan in #9668 requires the existing 0.0.7 metrics to reproduce and describes the new force
metrics and SNQI-v2 fields as the additions. The kernel correction can change simulation forces, so
its use in that campaign must pass the row-equivalence gate; a mismatch is a stop condition. The
paired kernel configs above are diagnostic inputs and do not alter the frozen 0.0.7 input or claim
that the 0.0.8 equivalence gate passed. The separate #9758 pedestrian-contact question remains
unresolved by this angle correction.

## Evidence status

Focused regression tests cover both kernel implementations, the explicit selector path, legacy
default behavior, and metadata omission when the selector is absent. No general planner, pedestrian,
or dissertation claim follows from those tests.

## Paired source-pinned diagnostic — 2026-09-28

The two versioned kernel configs were run on source commit
`450ca42b7b98026949992284d37520322d14ed82`, using worktree-asserted imports, two scenarios, seeds
111–113, `horizon=600`, `dt=0.1`, one worker, and recorded forces plus simulation-step traces. Each
arm wrote six episodes; both completed without run failures. Scenario and metric parameters match
except for the selector-dependent algorithm-config hash and kernel-version field. Every episode row
records the expected source commit and kernel selector.

The source commit predates PR-base refresh and is not in the current PR ancestry. Its implementation,
config, scenario, and test files were compared with current-base implementation commit
`1340824b69d8cc6c059e2f0de41865bfb60310bb`; all compared files are byte-identical. The focused
regression suite was rerun after that transplant on PR head
`d1c83ea367aaf9b950b0dc7907fccc58aee69309` (85 passed, 1 expected xfail).

| Scenario | Seed | Legacy | Wrapped | Steps (legacy/wrapped) | Maximum recorded position difference |
|---|---:|---|---|---:|---:|
| `classic_group_crossing_medium` | 111 | success | success | 183/183 | 0.0666880473 m |
| `classic_group_crossing_medium` | 112 | success | success | 183/183 | 0.234921857 m |
| `classic_group_crossing_medium` | 113 | success | success | 192/192 | 0.335183203 m |
| `classic_head_on_corridor_medium` | 111 | collision | collision | 181/180 | 0.346257723 m |
| `classic_head_on_corridor_medium` | 112 | success | success | 309/308 | 1.27508777 m |
| `classic_head_on_corridor_medium` | 113 | success | success | 316/314 | 0.970636728 m |

Terminal outcomes matched in all six pairs. Values for energy, mean clearance, curvature, mean
pedestrian force, total robot-attributable force impulse, and active robot-force mean changed in all
six pairs. Position differences compare same-index trace states over each pair's common prefix; they
do not identify a causal interaction. This small diagnostic is not a 0.0.7 release-row comparison,
statistical estimate, or SNQI-equivalence result.

The exact runtime `SocialForce` inputs were captured in a second pass, with episode and step tags;
the six episode identities, full step ranges, selector metadata, source commit, and equality of
instrumented versus original metrics/traces were verified. On each captured state, both kernels
were evaluated for each directed pedestrian pair inside the recorded 20 m activation threshold,
with the recorded factor 5.1 applied. At the declared `>1e-12` force-vector difference threshold,
265 of 12,916 candidate directed interactions on legacy-arm states and 245 of 12,842 on wrapped-arm
states differed. Of those, 112 and 63 respectively changed from legacy force norm `<=1e-12` to
wrapped norm `>1e-12`. These counts repeat directed pairs per simulation step; they are
counterfactual kernel-sensitivity counts on the two recorded trajectories, not unique pairs or a
population frequency. Per-episode counts are in the raw `interaction_counts.json` and
`interaction_counts.md` files.

The separate #9758 `surface_v3` rollout was also re-run through the existing `_rollout_v3` test
helper with both explicit kernel selectors. The selector appeared in planner diagnostics, and the
rollout values matched exactly across selectors. Standing-pedestrian minimum clearance/final speed
were 0.397368619862 m / 0.015908807501 m/s at `dt=0.1` and 0.398551836827 m / 0.017524083907 m/s
at `dt=0.05`; crossing-pedestrian minimum clearance was 0.117046958640 m / 0.194831537657 m.
Clearance exceeded 0.05 m in each case and standing speed remained below 0.05 m/s. This rechecks
only the opt-in `surface_v3` planner path, which is separate from the pair-kernel branch.

Raw artifacts are retained at
`/home/luttkule/robot_sf_campaign_retrievals/20260928/issue9764-social-force-kernel-paired-diagnostic-450ca42b`.
`SHA256SUMS` in that directory verifies all 44 listed files. The legacy and wrapped episode files have
SHA-256 values `f4faf49f0646457eadb33da0700a9425d4f6292b6b64a7e7ed41300595ed11a6` and
`9351259aa243a7ab78cadd2d819d86ae3fb7201504f306582e71f9c1da6c3a15`; the compact summary has
SHA-256 `a9ce8f7ef9f4b21fb832f571bd021dde32328e075ad61978ca1a1670125572d3`. The `#9758` recheck
receipt has SHA-256 `36a6944f5fc6f90914328377a2d6d45113cd17eafd92a9d8418501362d41c983`; the interaction
count JSON and Markdown reports have SHA-256 `81c41f1c94d4b8fb51c36888f8ed8efb5ecc4fb6593d2f79a1cf5d0e89dd9e57` and
`0d8ea181ed575631106ee149d7d3f2aae09e16fdcc2e783a267395987722130c`. These raw files remain
local-host, `durable-required-private` diagnostic artifacts; they are not a published evidence bundle.
An initial context-capture harness attempt wrote no episode rows because it read a step index that the
runner helper does not accept; the failure and its empty outputs are preserved, and the corrected
capture pass completed with all calls tagged and verified.
