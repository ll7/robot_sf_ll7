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
arms use 25 explicit, reviewed scenario/algorithm pairs. An unrecorded algorithm on the same assets receives current defaults. Standalone scenario construction uses the current default unless the caller
explicitly scopes the execution source through the same default selector as the planner. Direct planner builders can pass `source_path`; direct
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

`tests/planner/test_hybrid_default_compatibility.py` contains twenty-three cases. The
same file reports twenty-three failures on base `303ddac871557ba4350933de54306b83288b437f`,
all at old-default assertions, then twenty-three passes after implementation. Commands:

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
| `test_scenario_validity_override_wins_or_rejects_non_boolean` | Protect source-scoped scenario overrides: ignored booleans or accepted strings break the sensor contract. | Constructor override tests do not exercise the scenario loader. | Real native environment builder on a fixed released scenario; three fixed values, no reset/step; current control fails on base. |
| `test_registered_release_full_dataclasses_and_mapping_match_base` | Protect current activation alongside release identity and full typed parity; applying current defaults to releases fails. | Freeze tests hash raw maps and omit typed fill-in. | Real resolver/builder, fixed base dumps, nine cases; current control fails on base. |
| `test_registry_requires_known_source_and_matching_bytes` | Reject filename-only compatibility; a guessed legacy assignment fails. | No prior registry boundary. | Real scoped constructors, fixed temporary input; current control fails on base. |
| `test_release_registry_covers_learned_observation_contract` | Protect released spaces while exposing the current validity field; a missing learned source binding fails. | Prior optional-field tests do not cover released model paths. | Actual space builder, fixed map and bounds; current control fails on base. |
| `test_release_scenario_environment_dataclasses_match_full_base_dumps` | Protect every environment field on the 48 release scenarios; an env migration leaking into legacy fails. | No full before/after constructor comparison. | Actual scenario/env builders, complete fixed dumps; current control fails on base. |
| `test_native_episode_records_legacy_and_current_builder_default_sets` | Protect runtime propagation and provenance; lost context/source binding fails. | Constructor tests cannot catch missing runner wiring. | Three sequential native one-step episodes on dev1001; current control fails on base. |

The diagnostic observer regression
`test_missing_candidate_probe_skips_initial_goal_stop_without_speed_cap` catches
a first goal-stop decision before speed-cap diagnostics exist. It fails on base
with `TypeError`, then passes after a reporting-only guard. The inputs are fixed
and it calls the real planner and counterfactual diagnostic; no simulation is
reset or stepped. Existing probe tests start with speed diagnostics already
available and miss this initial goal-stop path. No test-only production seam is
needed. Its base-failure receipt is tracked separately.

The native provenance test also exercises an explicitly empty inline mapping on
a registered release scenario. Before the final dispatch correction it fails
with `assert 'legacy-0.0.8' == 'current'`; the corrected code treats `{}` as
current and reserves scenario fallback for missing or `None` inline config.
Existing source-based tests skipped the explicit-empty case. This uses the same
real runner at dev1001, with no additional test-only seam. All twenty-three migration
cases were rerun on immutable base and still fail their intended default witness.

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

The complete 3,084-episode comparison is recorded in
[comparison_summary.json](comparison_summary.json). All 294 failed episodes are
classified there; [comparison_episodes.json.gz](comparison_episodes.json.gz)
preserves every episode result. The empty-world gate has no newly failing cells
and no contacts. The larger crowded matrix exposes a contact regression and
35 newly failing cells, despite 67 recovered failures and a net gain of 32
successes. These findings require review before adopting the change.

| Set | Episodes per arm | Successes old → current | Collisions old → current | Timeouts old → current | Paired successful time delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Empty world | 102 | 88 → 100 (+11.765 percentage points) | 0 → 0 | 14 → 2 | +0.276 s (88 pairs) |
| Crowded standard scenarios | 1,440 | 1,285 → 1,317 (+2.222 percentage points) | 0 → 1 (+0.069 percentage points) | 155 → 122 | +0.677 s (1,250 pairs) |

Mean time to goal conditional on each arm's own successes changes from 17.181
to 17.642 s in empty worlds and 22.138 to 22.812 s in crowds. Those populations
differ; the paired column compares only cells succeeding in both arms. Positive
deltas mean slower completion. No performance or collision noninferiority claim
is made.

The sole executed contact occurs with current defaults in
`francis2023_robot_crowding`, dev1013, after 5.8 s. The environment reports
`is_pedestrian_collision`; the last native mode is `PROTECTIVE_STOP`, with no
feasible moving candidates. This does not establish the cause of the contact.
All fallback and degraded execution counters are zero.

| Failure classification | Old defaults | Current defaults |
| --- | ---: | ---: |
| Horizon exhausted with moving candidates | 104 | 75 |
| Forced stop timeout | 43 | 32 |
| Low progress or livelock timeout | 22 | 10 |
| Goal stop before route completion | 0 | 7 |
| Executed contact | 0 | 1 |

The summary lists all 35 newly failing crowded cells individually. Shared
narrow-doorway empty failures at dev1001 and dev1002 remain forced-stop timeouts.
Neither the empty gate nor the net crowded success gain establishes release
admission or collision safety.

[comparison_manifest.json](comparison_manifest.json) binds 1,428 runtime/input
files and dependencies. Original producer revisions are preserved even after
branch rebase. [main_integration_check.json](main_integration_check.json) verifies
all bound bytes still match after integrating main; upstream changes affect
analysis and test tooling only. The initial run encountered a diagnostic
observer error before stepping the environment. After the tested reporting-only
guard, 1,738 completed episodes were retained with verified unchanged native
control/input bytes, then the remaining episodes completed. Exact source
amendment and AST-equivalence evidence are in
[diagnostic_amendment.json](diagnostic_amendment.json).

[comparison_trace_identities.json.gz](comparison_trace_identities.json.gz)
records SHA-256 identities and sizes for every raw record and full step trace.
The raw traces remain under the owned lane's comparison directory; preserve
that directory during handoff. The tracked complete episode results, failure
classifications, source manifest and trace identities are durable review
artifacts; the command above reproduces the complete development matrix.

Final source review: [post_comparison_source_review.json](post_comparison_source_review.json)
records a subsequent change to the episode decorator's source-dispatch
condition, with strict AST comparison. The direct diagnostic comparator never
calls that decorator and scopes its current defaults explicitly. All constructor
fill-in, registry checks and native diagnostic control are unchanged. The ignored
generated `_version.py` also changes when the editable package rebuilds after
rebase; it contains informational version constants. The other 1,426 bound
files match. The complete measured results remain applicable; the original
producer identities and hashes are preserved.

The existing pedestrian/static-gate characterizations
`test_v4_continuous_static_acceptance_still_checks_pedestrian_collision` and
`test_v4_route_guide_candidate_hits_static_collision_gate` pin
`physical_static_exclusion_enabled=False`: their empty geometry stubs deliberately
isolates the older continuous gate from occupancy-grid conservatism. With the
new default the unrelated physical gate reads geometry absent from those stubs.
The pins preserve the original dynamic/static collision witnesses and catch an
erroneous static-clearance early return or route-guide bypass. Physical-wall tests use
real geometry instead. No additional production seam or new test is introduced.

The complete planner directory passes after these fixture pins (2,321 passed,
13 skipped); both isolated witnesses also pass with immutable base planner bytes.
[latest_main_integration_check.json](latest_main_integration_check.json) records
the subsequent main integration and verifies no additional comparison-bound
runtime/input change.

## Review fixes and per-switch reopening evidence

The author choice remains all three on. The review fixes make enabled terminal
tracking wait for the bound navigator's actual completion (radius or goal zone),
and require the declared validity sensor. Unbound planners track the terminal
center. Validity-off inputs retain the legacy tolerance rule. The route guide
tracks the terminal center with its waypoint tolerance restored after each call.

Scenario assets alone no longer choose a different default for a new planner.
Both typed builders use the common selector. Released callers explicitly scope
the registered algorithm source; absent-algorithm batch payloads retain `None`
so the runner can scope the registered scenario. Explicit empty inline mappings
remain new inputs. The registry now covers 183 arms in 15 release campaigns,
including the old PPO source and arms without algorithm files. Four historical
unfrozen placeholders remain blocked by their existing execution guard; their
pre-admission typed environment dumps also match. No frozen YAML changed.

[Review regression proof](review_fix_proof.md), [full released-arm comparison](review_fix_proof.json),
and [release-arm inventory](released_arm_inventory.json) provide the audit trail.
The older paired evidence above is retained with its original producer identity;
it predates the terminal tracking fix and cannot substitute for the new measurements.

The per-switch runner measures all-off, static-only, sensor-only, validity with
its required sensor, and all-on. Literal validity-only is invalid: it fails closed
because `next_valid` is absent. Compare validity+sensor to sensor-only to isolate
validity. Every executable arm is checked against its exact flag tuple.

```sh
scripts/dev/run_worktree_shared_venv.sh -- uv run python -m scripts.validation.run_hybrid_default_comparison --per-switch --output <comparison-folder> --workers 2
```

This runs 7,200 crowded episodes (48 scenarios × dev1001–1030 × five arms)
and repeats the 204-episode empty-world gate after the fixes. Full step traces
support every failure/contact classification; stationary pedestrian contacts
remain collisions. No released or sealed seed is reset or stepped.

## Evidence after the review fixes

The author choice remains all on. The [one-page decision brief](per_switch_decision.md)
reports the complete 7,404-episode sweep and all 1,440 literal goal-only contract
errors, with an author recommendation under the reopen clause.
[All 48 scenario rows and times](per_switch_scenarios.md),
[full classifications and paired deltas](per_switch_summary.json),
[lossless episode records](per_switch_episodes.json.gz), and
[all raw trace identities](per_switch_trace_identities.json.gz) are tracked.
The measured producer is `a24bf6e04f293cce4722da4b34f365606684e0fe`;
[applicability proof](measurement_applicability.json) explains the subsequent
episode-selection correction without relabelling the measurements.
Raw per-step traces remain preserved outside the worktree.

All ten review bug tests fail on reviewed head and the 33 focused cases pass:
[proof and four test-value questions](review_fix_proof.md). Full release-arm
constructor/environment dumps now compare canonical JSON bytes after only root
redaction. Config-less release identities cover 59 arms; unknown hybrid inputs
remain current. No released learned observation space changes. The two new
validity-arm seed-1026 scorer livelocks remain explicitly reported for the author;
the hard terminal-stop fix does not claim to resolve every preferred-stop decision.
