# Hybrid defaults for release 0.1.0

Diagnostic development evidence only. This change does not admit release results
or assert collision safety. Retired and sealed seeds are excluded from new runs.

The three current defaults are `physical_static_exclusion_enabled: true`,
`goal_next_validity_enabled: true`, and `include_goal_next_valid: true`. Every
explicit value still wins. The sensor flag adds structured `goal.next_valid`
(and flat `goal_next_valid`) as a float32 vector of length one. It distinguishes
an absent successor from a valid waypoint at world origin.

## Released configuration compatibility

`robot_sf/common/legacy_hybrid_defaults.json` explicitly registers released
source paths, their exact file hashes, hybrid base dependencies, and canonical
effective-config digests. The registry covers the four frozen 0.0.8 hybrid
candidates, five older released hybrids, the release scenarios/campaigns, and
release algorithm inputs including PPO and guarded PPO. A name that looks
frozen does not confer legacy defaults. Registered inputs whose bytes drift are
refused before execution rather than silently assigned a new contract.

Compatibility lives in typed constructor fill-in. The episode runner selects a
scoped default set using its original algorithm source path; config-less release
arms use the registered scenario source. Standalone scenario construction uses
its scenario source. Direct planner builders can pass `source_path`; direct
constructors can use `defaults_for_source`. Unknown inputs receive current
defaults. No key is added to the resolved algorithm mapping, and no frozen YAML
or canonical config digest changes. The recorded/current digest table is in
[compatibility_audit.json](compatibility_audit.json); all nine full planner and
environment dumps match their immutable base snapshots. The scoped policy is restored on exit;
policy caching also distinguishes the selected default sets.

Episode `algorithm_metadata.hybrid_default_policy` records `legacy-0.0.8` or
`current` separately from the config identity. Explicit overrides can differ
from the selected fill-in, so the development runner additionally records the
three effective flag values.

Released PPO and guarded PPO model metadata describes observation keys without
successor validity. The structured environment space changes under the current
default. Their registered algorithm inputs retain the exact old space. The cached guarded
PPO checkpoint was inspected without inference: its checksum matches the model
registry and its saved structured keys omit successor validity; see
[learned_observation_check.json](learned_observation_check.json). The Dict PPO
adapter selects saved model keys, so an extra runtime key need not change its
model input tensors. Box policies flatten the runtime observation and therefore
need the legacy flag or an explicit false override to preserve their dimension.
The plain 0.0.8 PPO checkpoint was not hydrated locally; its registered observation
metadata and actual environment space were checked. This is
space/contract preservation, not a checkpoint performance evaluation or retrain.
Unregistered learned-policy configurations must deliberately choose the sensor
flag matching their training contract.

## Tests and base proof

`tests/planner/test_hybrid_default_compatibility.py` contains twenty cases. The
same file reports twenty failures on base `303ddac871557ba4350933de54306b83288b437f`,
all at old-default assertions, then twenty passes after implementation. Commands:

```sh
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/planner/test_hybrid_default_compatibility.py -n 8 -q
```

The base failure is `assert False is True` for the missing-field current-default
witness, rather than an import or fixture error. Base snapshots contain full
built dataclass dumps, compressed losslessly: nine released planner configs,
the full environment constructor config, and all 48 real release-scenario
environment configs. The latter were captured by executing immutable base
module bytes in an isolated capture process. Maps and all inherited fields are
compared; only the repository root in path fields is normalized to `<repo>`.
No environment was reset or stepped during snapshot capture.

| Test | Behavior and credible regression | Existing coverage gap | Real path, determinism, base failure |
| --- | --- | --- | --- |
| `test_current_defaults_enable_all_three_switches` | Protect missing-field activation; a restored false default fails. | Earlier tests exercise opt-in repairs. | Typed constructors; no random inputs; false-on-base assertion. |
| `test_each_explicit_switch_overrides_the_selected_defaults` | Protect independent explicit on/off values; a coupled toggle or ignored value fails. | Earlier parsing tests omit the other current defaults. | Real parser/constructor, six deterministic cases; omitted companion remains false on base. |
| `test_registered_release_full_dataclasses_and_mapping_match_base` | Protect current activation alongside release identity and full typed parity; applying current defaults to releases fails. | Freeze tests hash raw maps and omit typed fill-in. | Real resolver/builder, fixed base dumps, nine cases; current control fails on base. |
| `test_registry_requires_known_source_and_matching_bytes` | Reject filename-only compatibility; a guessed legacy assignment fails. | No prior registry boundary. | Real scoped constructors, fixed temporary input; current control fails on base. |
| `test_release_registry_covers_learned_observation_contract` | Protect released spaces while exposing the current validity field; a missing learned source binding fails. | Prior optional-field tests do not cover released model paths. | Actual space builder, fixed map and bounds; current control fails on base. |
| `test_release_scenario_environment_dataclasses_match_full_base_dumps` | Protect every environment field on the 48 release scenarios; an env migration leaking into legacy fails. | No full before/after constructor comparison. | Actual scenario/env builders, complete fixed dumps; current control fails on base. |
| `test_native_episode_records_legacy_and_current_builder_default_sets` | Protect runtime propagation and provenance; lost context/source binding fails. | Constructor tests cannot catch missing runner wiring. | Two sequential native one-step episodes on dev1001; current control fails on base. |

The diagnostic observer regression
`test_missing_candidate_probe_skips_initial_goal_stop_without_speed_cap` catches
a first goal-stop decision before speed-cap diagnostics exist. It fails on base
with `TypeError`, then passes after a reporting-only guard. The inputs are fixed
and it calls the real planner and counterfactual diagnostic; no simulation is
reset or stepped. Its base-failure receipt is tracked separately.

No production seam was added solely for tests. Preservation assertions naturally
also hold on base; the same tests include the changed current behavior so the
migration contract fails on base. Existing evaluator characterizations now pin
the legacy switches explicitly, and the flat-observation frame test includes
the new declared key. Their original witnesses remain intact.

## Behavior comparison

```sh
scripts/dev/run_worktree_shared_venv.sh -- uv run python -m scripts.validation.run_hybrid_default_comparison --output <comparison-folder> --workers 2
```

The native diagnostic runner uses the standard 48 release scenarios. The
empty-world sweep uses the canonical pedestrian-removal transformation on those
48 plus the three doorway widths at dev1001–1002. Crowded comparisons use only
dev1001–1030. Both arms use authored scenario horizons and dt 0.1 s. The old
arm explicitly pins the three false values; the current arm omits all three
and verifies they resolve true in execution. No release seed policy executes.

The manifest binds source revision, runtime file hashes, environment versions,
scenario inputs and expected 3,084 episodes. Full traces preserve contacts,
feasible-moving candidates and displacement. Failure labels describe executed
contacts, goal stops before route completion, forced stops with actual evaluated
candidates, low progress/livelock, or horizon exhaustion; a timeout
is not proof of physical infeasibility. Success/collision deltas and time deltas
on common successes are descriptive development results. Fallback or degraded
execution raises an error and cannot count as success. The hybrid implementation
runs directly; command-space metadata saying `adapter` is distinct from a
fallback planner.

Full comparison results are pending until `summary.json` is complete.
