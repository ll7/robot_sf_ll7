# Issue #9751: hybrid v4 replaces the four hybrid slots of the 0.0.8 roster

This note prepares #9751 under the author amendment on #9668 (2026-09-28, "hybrid arm naming").
It chooses the v4-named keys, checks each slot for an honest v4 twin, and wires the 0.0.8
templates behind placeholders that cannot launch before #9748 freezes the v4 parameters. It
reports no run, tuning result or benchmark claim.

## Rules this implements

1. A key that names a version never runs a different version.
2. The four hybrid slots are replaced by hybrid v4 arms under new, v4-named keys. The roster stays
   at 14 slots; no arm is added (14 × 48 × 30 = 20,160 main rows).
3. A slot without an honest v4 twin keeps its unchanged 0.0.7 key and implementation.
4. The 0.0.7 comparison reports each replaced slot as `implementation replaced`, not as a paired
   correction. The v3 rows stay frozen and citable in 0.0.7.
5. v4 parameters are frozen only after #9748. Nothing here tunes, and nothing runs on seeds 111–140
   or on the #9748 development split.

## Slot-by-slot finding

The two "scenario-adaptive ORCA v2" arms are not a separate planner family. Their 0.0.7 release
configs are `hybrid_rule_local_planner` candidates on the v3 base
`configs/algos/hybrid_rule_v3_teb_like_rollout.yaml`. "v2" names their scenario-override set, and
"orca" names one scenario (`francis2023_leave_group`) that they hand to tuned ORCA. The #9747 fixes
(surface-clearance speed levels, drive-limited rollouts, braking check) live in the hybrid core, so
they apply to these arms exactly as they apply to the fast-progress arms.

#9747 already added the v4 twins. For all four slots the twin differs from its v3 predecessor only
in these points, and the new test checks that each twin keeps the same set of override scenarios:

- the base config is `configs/algos/hybrid_rule_v4_clearance_braking.yaml`
  (`planner_variant: hybrid_rule_v4_clearance_braking`);
- the v3 raise to 3.0 m/s (speed and acceleration) is dropped; the speed cap is 2.0 m/s;
- centre-distance yielding overrides (3.5 m / 4.5 m) are restated as surface clearances
  (2.1 m / 3.1 m for the 1.0 m robot and the 0.4 m pedestrian).

The other scenario overrides and the `francis2023_leave_group` ORCA override are carried over
byte for byte.

| 0.0.7 slot key | 0.0.8 key | v4 twin (unfrozen candidate) | Honest twin? | Recommendation |
|---|---|---|---|---|
| `scenario_adaptive_hybrid_orca_v2_bottleneck_yield` | `scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4` | `configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4_s30_h600_release.yaml` | yes: same override scenarios, v4 core | replace |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | `scenario_adaptive_hybrid_orca_v2_collision_guard_v4` | `configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard_v4_s30_h600_release.yaml` | yes: same override scenarios, v4 core | replace |
| `hybrid_rule_v3_fast_progress_static_escape` | `hybrid_rule_v4_fast_progress_static_escape` | `configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_s30_h600_release.yaml` | yes | replace |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | `hybrid_rule_v4_fast_progress_static_escape_continuous` | `configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release.yaml` | yes | replace |

None of the four twins is a relabel, so no slot falls back to its 0.0.7 implementation under
rule 3. That fallback stays available per slot. If #9748 cannot freeze a twin, restore that
slot's 0.0.7 row and key in the template, set its mapping entry back to `paired`, and re-pin the
hashes.

Two caveats for the scenario-adaptive slots:

- In `francis2023_leave_group` (one of the 48 release scenarios), both the v2 arm and its v4 twin
  run the same tuned ORCA. These cells are expected to be identical within the replaced slot, so
  they are a useful sanity check in the comparison. A difference there needs its own cause.
- The per-scenario overrides are keyed on release scenario IDs (`classic_bottleneck_high`,
  `classic_merging_low`, `francis2023_blind_corner`, `francis2023_perpendicular_traffic`,
  `francis2023_leave_group`). The #9748 development identities (`issue_9748_dev_*`) never trigger
  them. The overrides therefore stay as they were carried over from v3 and cannot be tuned. This
  is correct under the held-out rule, but it means the #9761 finding (`very_slow_speed: 0.6` against
  the 0.15 m v4 stop band in `francis2023_perpendicular_traffic`) is not exercised by the
  development split and must be settled before the freeze.

### Key names

Each new key says `v4`. The scenario-adaptive keys keep `v2` because it names the unchanged
override set; the `_v4` suffix names the hybrid core. This matches the twin config names from
#9747 and the key suggestion in that PR's Downstream Propagation section. It does not clash with
the unrelated historical `hybrid_rule_v4_recovery_aware` candidate (a rejected v3-scorer ablation).

## Manifest mapping

`robot_sf/benchmark/release_parameter_freeze.py::ARM_SLOTS_0_0_7_TO_0_0_8` holds the mapping in
code. It follows the order of the 0.0.7 manifest roster, and a test pins it against the frozen
0.0.7 manifest keys:

| 0.0.7 key | 0.0.8 key | comparison |
|---|---|---|
| 10 unchanged keys (`prediction_planner`, `goal`, `social_force`, `orca`, `ppo`, `socnav_sampling`, `sacadrl`, `guarded_ppo`, `predictive_mppi`, `risk_dwa`) | same | `paired` (every difference still needs a named cause) |
| `scenario_adaptive_hybrid_orca_v2_bottleneck_yield` | `scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4` | `implementation replaced` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | `scenario_adaptive_hybrid_orca_v2_collision_guard_v4` | `implementation replaced` |
| `hybrid_rule_v3_fast_progress_static_escape` | `hybrid_rule_v4_fast_progress_static_escape` | `implementation replaced` |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | `hybrid_rule_v4_fast_progress_static_escape_continuous` | `implementation replaced` |

`social_force` and `socnav_sampling` stay `paired` at slot level, even though they bind corrected,
versioned configs (rule 5 of the amendment). The release manifest must still disclose those configs
(open on #9668).

## Wiring behind placeholders (this PR)

- `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml`: the four
  hybrid rows now carry the v4 keys, and each points at
  `configs/policy_search/release_0_0_8_placeholders/<key>.unfrozen.yaml`. There are still 14 rows.
- Each placeholder declares `release_parameter_freeze: {status: unfrozen, required_gate:
  ll7/robot_sf_ll7#9748, replaces_0_0_7_slot, implementation_family, unfrozen_candidate}`. It has
  no base config and no parameters, so nothing about the eventual v4 parameters is fixed here.
- Three layers fail closed on any `release_parameter_freeze` block whose status is not `frozen`.
  Loading and inspecting the template still work.
  - The map runner's `_parse_algo_config` and the shared `resolve_candidate_manifest_runtime`
    raise `UnfrozenReleaseParametersError`.
  - `prepare_campaign_preflight` raises before it creates any output directory.
  - `validate_release_planner_roster`, used by manifest validation and the no-campaign
    rehearsal, reports one blocker per unfrozen arm.
- `configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml`: the planner keys
  and groups use the new keys. `campaign_config_sha256` is re-pinned
  `7dc9a2dd…` → `f453b7c8…`, because the campaign template bytes changed.
- Runtime smoke v0_4 has to match the template row for row (#9803), so its config gets the same
  four rows. Its `derived_from` pin and manifest pins are re-pinned: the config goes
  `43683e99…` → `698b4bc4…`. Its manifest validation now reports exactly the four freeze
  blockers. So v0_4 cannot run before #9748 either, which is correct: a smoke of placeholders
  would test nothing. v0_4 is edited in place for the same reason as in #9854: it has never been
  run. The canonical runtime-smoke admission roster (`RUNTIME_SMOKE_PLANNER_KEYS`, bound to the
  historical v0_2 smoke) is unchanged.

**Freeze procedure (after #9748 and after #9817 lands).** For each slot, add a new frozen config
next to its candidate: a new file, with the #9748-selected parameters and the #9761 alignment.
Point the template row at it and re-pin the template, the release template and v0_4. Keep the
placeholder files for history, or remove them in that PR. Never flip a placeholder to
`status: frozen`, because it has no parameters to freeze. The guard admits a config with no freeze
block. A frozen config may also carry `release_parameter_freeze: {status: frozen, ...}` with its
#9748 receipt.

## What changes in calibration (#9667)

`configs/benchmarks/snqi_v2/calibration.dev101_102.yaml` still lists the four 0.0.7 hybrid keys
with their v3 configs. It also still runs legacy socnav and `social_force_terminal_goal_v1`
(#9850). The failed 1,152/1,344 calibration is diagnostic only. For 0.0.8 anchors, the
calibration roster must be the frozen 0.0.8 roster: the four v4 keys with their frozen configs, and
the corrected `social_force` and `socnav_sampling` configs. So the calibration can only be regenerated
after the v4 freeze. It uses the same 14 slots, so its size does not change (14 × 48 × 2 = 1,344).
This PR does not touch that file. It is a #9667 / #9850 follow-up, blocked by #9748.

## What changes in the 0.0.7 comparison tooling

`scripts/analysis/compare_issue_9431_release.py` is the 0.0.6 → 0.0.7 comparator. It pairs rows
by identical arm key against one frozen 14-key set. A 0.0.7 → 0.0.8 comparator (#9668 stage 3)
must:

1. read predecessor and successor rows with their own key sets
   (`ARM_SLOTS_0_0_7_TO_0_0_8` gives both);
2. pair rows on (slot, scenario, seed);
3. for `paired` slots, keep the tolerance-1e-12 per-field comparison and require a named cause
   for every difference;
4. for `implementation replaced` slots, emit descriptive per-field deltas and rate and ranking
   impact under that label, and never count them as corrections or equivalences. The exception is
   the `francis2023_leave_group` cells of the two scenario-adaptive slots, which should match
   (ORCA in both);
5. write the label into the comparison report and the release manifest disclosure.

That is a new comparator, not a small patch, so it is a follow-up. This PR adds only the shared
mapping it will use.

## Follow-ups

- #9748: include both scenario-adaptive v4 twins in the development tuning, or freeze them
  untuned on the record. The current dev-split config on `codex/issue-9748-dev-split-20260928`
  lists only the two `hybrid_rule_v4_fast_progress_*` candidates.
- #9761 / #9817: land the v4 surface-clearance and angular-window fix, and settle the
  `francis2023_perpendicular_traffic` `very_slow_speed: 0.6` override before the freeze.
- Freeze PR (after #9748): add the frozen configs, point the template rows at them, re-pin the
  hashes, and see the procedure above.
- #9667 / #9850: regenerate the SNQI-v2 calibration roster from the frozen 0.0.8 roster.
- #9668 stage 3: build the slot-aware 0.0.7 → 0.0.8 comparator described above.
- #9668: disclose the four replaced slots in the release manifest, together with the corrected
  `social_force` and `socnav_sampling` configs.
