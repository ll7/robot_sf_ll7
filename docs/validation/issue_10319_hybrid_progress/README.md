<!-- AI-GENERATED (robot_sf#10319, 2026-10-12) - NEEDS-REVIEW -->
# Hybrid v4 progress diagnosis for issue #10319

Diagnostic-only, development seed 1001, seven named cells. No sealed or held-out
seed is executed. All 14 episodes use native execution with no fallback/degraded
markers and matched reset digests. This is a current-main mechanism comparison,
not a replay of the frozen 0.0.8 implementation or a release result. Current
pedestrian and simulator code may differ from the audit source. One seed cannot
estimate scenario-wide rates or establish a safety guarantee.

Execution source: `53721be3248600e08641a9816d094b605e3c93e4`.
Historical audit source: `373dbfde4f39667cf9e8732dabe7df5118bdeab1`.
Historical comparator: `07f7e8d43084de748915e1b1eb8b2a1603357c6e`.
The complete linked audit report, cell manifest, summary, representative U3
recordings and underlying trace arrays were read before these runs.

## Cause

The static/adaptive group stalls are a liveness limitation in the intentionally
conservative released v4 speed policy. They are not established physical
infeasibility. At current main,
[`hybrid_rule_local_planner.py:924`](../../../robot_sf/planner/hybrid_rule_local_planner.py#L924)
sets the band cap to zero below 0.15 m surface clearance despite a positive
inverse-braking cap; line 932 takes the minimum.
[`hybrid_rule_local_planner.py:1258`](../../../robot_sf/planner/hybrid_rule_local_planner.py#L1258)
then caps the dynamic window, and
[`hybrid_rule_local_planner.py:2064`](../../../robot_sf/planner/hybrid_rule_local_planner.py#L2064)
clips every generated command. The physical rollout and braking evaluator cannot
choose a translating action that was removed before evaluation.
The released thresholds and derivation are documented at
[`hybrid_rule_v4_clearance_braking.yaml:58`](../../../configs/algos/hybrid_rule_v4_clearance_braking.yaml#L58).

Static join dev1001 ends with surface gap 0.136815 m, a positive inverse-braking
cap of 0.189187 m/s, a final cap of zero, and a 31.5-second low-speed interval.
Static leave and both adaptive join identities show the same zero-cap mechanism.
Their matched cap-only episodes reach route completion without contact.

The station cell has a different observed failure pattern: it moves 650 steps,
its longest low-speed interval is only 1.6 s, and it still times out after the
cap-only change. The scalar cap alone therefore does not resolve that cell.
This is a moving-budget timeout, not proof that the route is physically
infeasible or that every historical station timeout has the same cause.
Robot crowding is also moving rather than a sustained stationary stall; the
cap-only episode reaches completion. Pedestrian overtaking succeeds in both
variants for this seed, so the issue's 60-seed aggregate regression is not
reproduced in that particular cell/seed.

## One-factor diagnostic

The ablation replaces only `_v4_human_speed_cap` with the existing legacy
center-distance `_human_speed_cap`. It preserves all other resolved planner
fields, the 2.0 m/s ceiling, scorer, observed radii, native plant, map,
constant-velocity pedestrian predictions, rollout, contact checks, braking-tail
rejections, and authored episode budgets. The hook runs outside production
source. It is not the production successor in PR #10294.

**Safety boundary:** removing the radial inverse-braking cap changes a v4
safety guarantee. Retaining other guards and observing zero contacts does not
prove equivalent safety. The group runs approach about 0.051 m surface gap;
acceptance must evaluate that proximity as well as progress.

| Scenario | Planner | Before → cap-only | Steps before/after | Low-speed interval before/after (s) | Min surface gap before/after (m) | Classification |
| --- | --- | --- | ---: | ---: | ---: | --- |
| classic_station_platform_medium | continuous | timeout → timeout | 650/650 | 1.6/1.4 | 0.279293/0.294402 | moving_budget_timeout_persists |
| francis2023_join_group | static | timeout → success | 400/99 | 31.5/0.2 | 0.136815/0.051294 | stop_band_stall_rescued |
| francis2023_join_group | adaptive bottleneck | timeout → success | 400/99 | 31.4/0.2 | 0.143289/0.051294 | stop_band_stall_rescued |
| francis2023_join_group | adaptive collision | timeout → success | 400/99 | 31.4/0.2 | 0.143289/0.051294 | stop_band_stall_rescued |
| francis2023_leave_group | static | timeout → success | 400/149 | 30.8/0.2 | 0.146028/0.051148 | stop_band_stall_rescued |
| francis2023_pedestrian_overtaking | continuous | success → success | 465/468 | 0.6/0.6 | 1.122345/1.178766 | success_with_changed_trajectory |
| francis2023_robot_crowding | static | timeout → success | 400/188 | 0.3/0.2 | 0.144236/0.146174 | moving_crowding_timeout_rescued |

All seven trajectories changed and every changed episode is classified above.
The four stationary group timeouts and one moving crowding timeout are rescued;
station remains a moving-budget timeout; overtaking remains successful with a
changed trajectory. Zero contacts occur in both cohorts. Low-speed intervals
use observed post-step robot speed below 0.05 m/s at 0.1-second intervals.
The compact [episode evidence](dev1001.json) binds raw-row hashes, reset/trajectory
digests, source revision, execution status and all per-episode classifications.

## Reproduction

Use the canonical `robot_sf_bench run` entrypoint, the release-0.0.8 scenario
matrix, the appropriate `*_s30_h600_release_0_0_8_frozen.yaml` candidate, and
explicit `seeds: [1001]`. Select one named scenario in an include manifest;
never allow an inherited release seed list to resolve. Use `--base-seed 1001
--repeats 1 --dt 0.1 --workers 1 --benchmark-profile experimental
--socnav-missing-prereq-policy fail-fast --record-forces --no-video
--video-renderer none --no-resume` and
`--telemetry-config configs/benchmarks/analysis_ready_full_campaign.yaml`.
Set `--horizon 650` for station, 600 for pedestrian overtaking, and 400 for the
other named cells. The per-scenario manifest overrides seeds before environment
construction. The preserved diagnostic hook is precisely:

```python
from robot_sf.planner.hybrid_rule_local_planner import HybridRuleLocalPlannerAdapter

def cap_only(self, state):
    cap = self._human_speed_cap(
        self._nearest_ped_distance(state["robot_pos"], state["ped_pos"])
    )
    gaps = self._v4_surface_clearances(state)
    self._last_v4_speed_safety = {
        "model": "diagnostic_v3_cap_only",
        "speed_cap": cap,
        "min_surface_clearance": float(min(gaps)) if gaps.size else None,
    }
    return cap

HybridRuleLocalPlannerAdapter._v4_human_speed_cap = cap_only
```

The public compact evidence excludes infrastructure identifiers and storage
locations. Full inputs, commands, raw traces and logs are preserved privately;
their raw-row hashes are in the committed JSON. No publication or sealed-evidence
admission is claimed.

## Implementation handoff and gates

[PR #10294](https://github.com/ll7/robot_sf_ll7/pull/10294) already implements an
opt-in guarded-progress successor for the same mechanism, with regression tests
and matched empty-world evidence. Its inspected head is
`04013a43c8f15e51c87a392b90fa5a4ac4a9d4d9`. It changes the same planner/test paths
and is stacked on PR #10166. This diagnostic PR avoids a competing implementation
and leaves [#10319](https://github.com/ll7/robot_sf_ll7/issues/10319) open.

This lane makes no planner, benchmark or test change. A new failing-on-main,
passing-on-fix regression test and a production before/after empty-world sweep
were therefore **not run**. PR #10294's prior test and sweep claims were read;
they are not presented as independently rerun proof. Its reported empty-world
comparison covers 480 pairs, with 463 successes and 17 timeouts per arm, identical
recorded trajectories and zero contacts. Its reported crowded comparison covers
720 pairs, including seven introduced moving-budget timeouts. Those results are
on its recorded sources, not this current-main source.

The existing PR's crowded table omits `francis2023_robot_crowding` and
`francis2023_pedestrian_overtaking`, so it cannot discharge this issue's full
requested production comparison. Independent review must reconcile its base and
current main, classify those missing successor comparisons, inspect the seven
introduced timeouts and proximity tradeoff, and decide whether to adopt the
opt-in. Do not substitute this cap-only ablation for the production gate.

Owner: maintainer and PR #10294 implementation owner. Next action: reconcile the
same-mechanism issues and complete the missing comparisons on the actual selected
successor. Revival condition: an ownership handoff or a selected successor with
current-main validation; preserve this diagnostic evidence rather than repeating
successful runs without a changed question.
