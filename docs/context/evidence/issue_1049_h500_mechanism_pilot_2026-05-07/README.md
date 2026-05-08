# Issue 1049 H500 Mechanism Pilot Evidence

Selected trace-backed fixed-h100 vs h500 ORCA pilot evidence for issue #1049.

## Contents

- `representative_trace_summary.json` and `.csv`: compact metrics extracted from selected traces.
- `traces/`: selected per-step trace JSON files, small enough to retain in git for review.
- `manifest.sha256`: checksums for the retained trace JSON files.

## Selected Cells

| Mechanism target | Scenario | Seed | Why selected |
|---|---|---:|---|
| `budget_limited_clean_completion` | `classic_bottleneck_low` | 111 | Fixed h100 times out; h500 succeeds cleanly. |
| `exposure_enabled_completion` | `classic_t_intersection_medium` | 111 | Fixed h100 times out; h500 succeeds with more force-exposure/comfort-pressure steps and lower pedestrian clearance. |
| `safety_regressed_completion` | `classic_merging_low` | 111 | Fixed h100 times out; h500 continues until a collision/force-exposure event. |

## Interpretation Boundary

The exposure representative did not trigger the discrete `near_misses` counter in this seed. The
supported claim is therefore exposure/comfort-pressure increase from per-step force exposure and
minimum pedestrian distance, not a near-miss timing proof. The safety representative supports the
opposite boundary: longer horizon can expose unsafe behavior that fixed h100 would have hidden as a
timeout.
