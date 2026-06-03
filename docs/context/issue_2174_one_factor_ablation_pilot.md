# Issue #2174 One-Factor Hybrid Component Ablation Pilot

Status: current, diagnostic-only.

Issue #2174 executes the first bounded child of the #2170 one-factor manifest for parent #2104. The
full manifest remains an h500 protocol; this PR adds the planned one-factor candidate configs and a
manifest-aware runner, then runs one local h80 comparison as a cheap proof of executability.

## What Changed

- Added the planned #2170 one-factor candidate configs:
  - `issue_2170_static_escape_only`;
  - `issue_2170_static_recenter_only`;
  - `issue_2170_static_escape_recenter_no_transit`;
  - `issue_2170_continuous_static_checks_only`;
  - `issue_2170_scenario_adaptive_orca_selector_only`.
- Registered those candidates as `experimental_spike` rows in
  `docs/context/policy_search/candidate_registry.yaml` so the existing candidate loader resolves
  them by name.
- Added `scripts/tools/run_one_factor_ablation_pilot.py`, a thin wrapper that builds a local funnel
  from the frozen manifest, runs selected comparisons through
  `scripts/validation/run_policy_search_candidate.py`, and emits compact effect rows.

## Pilot Result

Command:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/tools/run_one_factor_ablation_pilot.py \
  --comparison-id static_escape_only_minus_base --horizon 80 --workers 2 \
  --output-dir output/issue_2174/static_escape_h80_w2
```

Result on commit `715816738a143ca4c4984ddb5dc57876d6cf7171`:

| A | B | Success delta | Collision delta | Near-miss delta | Avg-speed delta | Runtime delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `issue_2170_static_escape_only` | `hybrid_rule_v3_fast_progress` | 0.000 | 0.000 | 0.000 | 0.000 | +16.328s |

Both rows completed 18/18 jobs at h80 with zero failed jobs. The effect row is therefore a valid
local diagnostic for this slice, but it is not h500 benchmark evidence and does not support a
planner-promotion claim.

## Interpretation

The first one-factor comparison found no measurable safety/progress difference on the h80 slice,
while the static-escape-only row took longer because its scenario overrides split execution into
family runs. Confidence is about 0.75 that static escape alone is not the main driver on this
short-horizon slice; confidence would change after the h500 horizon and the remaining comparison
rows are executed.

## Next Step

Run the remaining selected comparisons with the same tool, preferably at the manifest h500 horizon
or with a documented staged horizon ladder. Report each row as diagnostic-only until all comparator
rows execute without fallback, degraded, unavailable, or failed status.
