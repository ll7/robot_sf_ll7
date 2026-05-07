# Issue #1034 Continuous Corridor Maneuver

Date: 2026-05-06

Related issues:

- #884: <https://github.com/ll7/robot_sf_ll7/issues/884>
- #1028: <https://github.com/ll7/robot_sf_ll7/issues/1028>
- #1029: <https://github.com/ll7/robot_sf_ll7/issues/1029>
- #1034: <https://github.com/ll7/robot_sf_ll7/issues/1034>

## Goal

Implement the #1034 follow-up to the disabled #1028 `corridor_subgoal` slice: verify short-horizon
corridor recovery motion against the environment's continuous static-obstacle surface instead of
relying only on occupancy-grid clearance, then prove whether it can safely recover #884 target
timeouts.

## Implementation

The implementation keeps the base planner deterministic and config-gated:

- `HybridRuleLocalPlannerAdapter.bind_env()` stores map bounds and simulator obstacle line segments
  for continuous static collision checks.
- `robot_sf.benchmark.map_runner._build_adapter_policy()` exposes planner environment binding via
  `_planner_bind_env`, so diagnostic and benchmark runners bind the real environment geometry.
- `HybridRuleCandidate` now supports an optional `rollout_sequence` for safety/scoring. The selected
  command remains the first immediate `(linear, angular)` action, but evaluation can check a
  turn-then-forward sequence before allowing the candidate.
- `corridor_subgoal` turn candidates now evaluate a planned turn followed by slow route-tangent
  forward motion when the rollout horizon allows it.
- `corridor_subgoal_use_continuous_static_check` keeps exact checks on the subgoal/strict path.
- `continuous_static_clearance_enabled` optionally lets exact environment geometry replace
  conservative occupancy-grid clearance-band rejection for all candidate sources. Occupied grid
  cells and continuous line collisions still fail closed.

The promoted candidate is:

- `configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape_continuous.yaml`
- registry key: `hybrid_rule_v3_fast_progress_static_escape_continuous`

The expensive h500 evidence was generated while the config still lived under
`output/ai/autoresearch/issue_1034_continuous_static_probe/candidate_enabled.yaml`; that probe
config is byte-for-byte equivalent to the tracked config except for the `name` field. The tracked
registry/config path was then smoke-tested successfully.

## Target H500 Evidence

Output root:
`output/ai/autoresearch/issue_1034_sequence_all_static_margin_probe/`

Final target command shape:

```bash
rtk timeout 180s env LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape_continuous_probe \
  --candidate-registry output/ai/autoresearch/issue_1034_continuous_static_probe/candidate_registry.yaml \
  --stage full_matrix \
  --scenario-name <scenario> \
  --seed <seed> \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_1034_sequence_all_static_margin_probe/<scenario>_<seed>_h500
```

| Scenario | Seed | Outcome | Collision flags | `corridor_subgoal` selected |
|---|---:|---|---|---:|
| `classic_merging_low` | 111 | route-complete success at step `481` | none | 0 |
| `classic_merging_low` | 113 | route-complete success at step `315` | none | 5 |
| `classic_merging_medium` | 111 | route-complete success at step `480` | none | 7 |
| `classic_merging_medium` | 112 | route-complete success at step `418` | none | 0 |
| `classic_merging_medium` | 113 | timeout at horizon `500` | none | 2 |

Interpretation:

- The #1034 mechanism recovers three target seeds that were not recovered by #1028:
  `classic_merging_low` seed `111`, `classic_merging_medium` seed `111`, and
  `classic_merging_medium` seed `112`.
- It preserves the previously recovered `classic_merging_low` seed `113`.
- It does not recover `classic_merging_medium` seed `113`, but keeps it as a safe timeout with no
  obstacle, pedestrian, or robot collision.
- No fallback or degraded execution was counted as success.

## Gate Evidence

Nominal h500:

```bash
rtk timeout 900s env LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape_continuous_probe \
  --candidate-registry output/ai/autoresearch/issue_1034_continuous_static_probe/candidate_registry.yaml \
  --stage nominal_sanity \
  --horizon 500 \
  --workers 2 \
  --output-dir output/ai/autoresearch/issue_1034_sequence_all_static_margin_probe/nominal_sanity_h500_rerun
```

- Decision: `pass`
- Episodes: `18`
- Success rate: `1.0`
- Collision rate: `0.0`
- Near-miss rate: `0.2778`
- Execution mode: adapter, benchmark availability `available`

Stress h500:

```bash
rtk timeout 1200s env LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape_continuous_probe \
  --candidate-registry output/ai/autoresearch/issue_1034_continuous_static_probe/candidate_registry.yaml \
  --stage stress_slice \
  --horizon 500 \
  --workers 2 \
  --output-dir output/ai/autoresearch/issue_1034_sequence_all_static_margin_probe/stress_slice_h500
```

- Decision: `tracked`
- Episodes: `24`
- Success rate: `1.0`
- Collision rate: `0.0`
- Near-miss rate: `0.3333`
- Execution mode: adapter, benchmark availability `available`

Tracked config smoke:

```bash
rtk timeout 180s env LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape_continuous \
  --stage smoke \
  --horizon 80 \
  --workers 1 \
  --output-dir output/ai/autoresearch/issue_1034_sequence_all_static_margin_probe/tracked_smoke \
  --docs-root output/ai/autoresearch/issue_1034_sequence_all_static_margin_probe/docs_handoff
```

- Decision: `pass`

## Current Conclusion

#1034 satisfies the scoped success metric from #884/#1034: at least one remaining timeout is
recovered without target obstacle collisions, `classic_merging_low` seed `113` remains successful,
and retained h500 nominal/stress gates show zero collisions.

The raw limitation is still important: `classic_merging_medium` seed `113` remains a safe h500
timeout. Treat this branch as a validated targeted recovery candidate, not proof that every
classic-merging target seed is route-complete and not a full-matrix benchmark-strengthening claim.
If a future PR claims broad benchmark improvement beyond this targeted #884/#1034 boundary, it
should run the full matrix or document why the retained targeted boundary is sufficient.
