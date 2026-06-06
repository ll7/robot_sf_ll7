# Issue #2280 First Research-v1 Trace Review

Issue: [#2280](https://github.com/ll7/robot_sf_ll7/issues/2280)
Parent issue: [#2159](https://github.com/ll7/robot_sf_ll7/issues/2159)
Input selection: [issue_2269_research_v1_trace_case_selection.md](issue_2269_research_v1_trace_case_selection.md)
Date: 2026-06-05
Status: diagnostic-only first trace-review case.

## Goal

Render the first durable #2269 trace-review case into a compact why-first report without inventing
missing trace-viewer artifacts. The selected case is
`head_on_corridor_route_offset_response`, because #2269 marks it
`selected_trace_slices_available` and names it as the first render/review child.

Compact evidence summary:
`docs/context/evidence/issue_2280_first_trace_review_2026-06-05/summary.json`.

## Source Case

| Field | Value |
| --- | --- |
| Scenario | `classic_head_on_corridor_low` |
| Intervention | `pedestrian_route_offset` |
| Baseline variant | `classic_head_on_corridor_low_noop` |
| Perturbed variant | `classic_head_on_corridor_low_ped_route_offset_y025` |
| Planners | `goal`, `orca`, `scenario_adaptive_hybrid_orca_v2_collision_guard` |
| Seeds | `111`, `112`, `116`, `117` |
| Pair rows | `12 completed`, `0 excluded` |
| Trace source | `docs/context/evidence/issue_1939_corridor_trace_response_2026-05-31/closest_approach_trace_slices.json` |

The source artifact stores compact closest-approach frame windows, not full
`simulation_trace_export.v1` traces or rendered panels. That is enough for a first trace-review
report, but not enough for visual trace-viewer annotations.

## Observed Evidence

Across all 12 completed baseline/intervention pairs, the route-offset perturbation increased mean
closest-approach center distance and clearance by `+0.153489 m`. Mean progress increased by
`+0.506475 m`, but that aggregate is dominated by one hybrid-planner seed.

| Planner | Pairs | Mean clearance delta | Mean progress delta | Mean closest-time delta |
| --- | ---: | ---: | ---: | ---: |
| `goal` | 4 | `+0.159909 m` | `-0.024578 m` | `+0.025 s` |
| `orca` | 4 | `+0.157236 m` | `-0.049732 m` | `+0.025 s` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 4 | `+0.143321 m` | `+1.593735 m` | `-0.800 s` |

The strongest local explanation is the hybrid planner on seed `117`:

| Variant | Closest step | Time | Closest pedestrian | Clearance | Progress | Terminal reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| no-op | `79` | `8.0 s` | `0` | `1.773034 m` | `-5.800117 m` | `max_steps` |
| perturbed | `46` | `4.7 s` | `1` | `1.946370 m` | `+0.781392 m` | `max_steps` |

For that seed, the perturbed closest approach happens `3.3 s` earlier, with `+0.173336 m` clearance
and `+6.581509 m` progress relative to the no-op closest-approach frame. The closest pedestrian
also changes from index `0` in the no-op window to index `1` in the perturbed window, which means
the compact slice shows a changed local interaction geometry rather than a same-pedestrian
trajectory improvement.

## Mechanism Activation

```yaml
mechanism_activation:
  activated: true
  activation_count: 12
  changed_command_source: unknown
  changed_outcome: false
  likely_failure_reason: "Route-offset perturbation changed closest-approach geometry, but terminal outcomes stayed max_steps and command-source telemetry was not preserved."
```

`activated` is `true` because all 12 reviewed pairs use the materialized
`pedestrian_route_offset` intervention variant. `changed_command_source` is `unknown`: the compact
trace slices preserve closest-approach geometry and planner identity, but not selected command
source, arbitration, guard, or handoff timelines.

## Supported Explanation

The trace slices support a bounded mechanism explanation:

> In this head-on corridor slice, pedestrian route offset can move the nearest-interaction geometry
> away from the robot at closest approach. One hybrid seed also reaches its closest-approach window
> much earlier and farther along the route.

This is useful failure-case evidence because it shows the intervention can alter local geometry and
route-progress timing, especially for seed `117`.

## Rejected Or Unresolved Explanations

- **Rejected as a success claim:** the reviewed pairs do not show terminal improvement; the
  highlighted no-op and perturbed seed `117` rows both terminate by `max_steps`.
- **Rejected as broad planner evidence:** `goal` and `orca` have small negative mean progress
  deltas despite positive clearance deltas, and the hybrid progress signal is seed-local.
- **Unresolved command-source mechanism:** selected command source and arbitration timelines are
  not in the compact source artifact.
- **Unresolved visual explanation:** no trace-viewer annotation set or rendered panel was produced
  in this PR.

## Claim Boundary

This report is diagnostic trace-review evidence only. It is not benchmark-strength evidence, not a
paper-facing mechanism claim, and not the full #2159 research-v1 trace-review pack. It supports
using `head_on_corridor_route_offset_response` as the first reviewed case for #2159, while the AMV
specific cases from #2269 still need renderable trace exports.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_1939_corridor_trace_response_2026-05-31/closest_approach_trace_slices.json
python -m json.tool docs/context/evidence/issue_2280_first_trace_review_2026-06-05/summary.json
python - <<'PY'
from pathlib import Path
import yaml
data = yaml.safe_load(Path("docs/context/evidence/issue_2269_research_v1_trace_case_selection_2026-06-05/case_selection_manifest.yaml").read_text())
case = next(case for case in data["cases"] if case["id"] == "head_on_corridor_route_offset_response")
assert case["durable_case_ready"] is True
for path in case["source_evidence"]:
    assert Path(path).exists(), path
PY
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
