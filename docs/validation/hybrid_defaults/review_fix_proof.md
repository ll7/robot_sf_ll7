# Review regression proof

Diagnostic-only. Before-fix head: `fcadacd13f71902635fc7008960aa17c2d95e75c`.
The all-on author choice remains unchanged. The baseline snapshots were captured
from immutable main `9d8dac2a140f00dadf2c8c2621977bcee96c83e0`, without reset/step.
At snapshot capture, the then-newer main changed only the release runbook and its test; this is a historical capture statement, not a claim about later main heads.

```sh
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/planner/test_hybrid_default_review_regressions.py -n 2 -q
```

With the changed production files restored to the reviewed head, the same ten
cases fail for their intended bugs: **10 failed in 24.11s**. Full output:
[review_fix_fail_before.txt](review_fix_fail_before.txt). With fixed production
bytes: **33 passed in 24.79s**, including all ten new cases and all 23 preservation/override cases. No test-only production seam was added.
All inputs are fixed. The worker-provenance case runs one native step at dev1001;
other new cases do not reset or step an environment.

| Test (cases) | Behavior / credible bug | Why existing coverage misses it | Before-fix witness; deterministic real path |
| --- | --- | --- | --- |
| `test_terminal_goal_at_022_m_keeps_tracking_before_environment_success` (3) | Keep tracking at 0.22 m until navigator success; reintroducing the 0.25 m stop deadlocks. Also assert actual completion stops, route-guide tolerance is restored, and validity-off retains its stop. | Prior validity tests only select a successor far from terminal completion. | `GOAL_STOP` at 0.22 m; real planner and route guide, real radius/zone navigators or unbound mode; fixed geometry, no RNG. |
| `test_enabled_validity_rejects_a_missing_sensor_field` (2) | Fail closed on absent nested/flat validity; an implicit valid default silently ignores the contract. | Prior sensor tests exercise present fields or opt-out. | `DID NOT RAISE`; real planner observation extraction, fixed fields. |
| `test_new_env_and_planner_share_defaults_on_registered_release_scenario` | New inputs use one current default source; inferring env defaults from scenario assets produces incompatible pairs. | Prior runtime test scopes both inside the runner. | Env false versus planner true; actual separate builders on the released matrix, no reset/step. |
| `test_released_002_ppo_constructor_uses_legacy_sensor_defaults` | Preserve the omitted sensor in the older PPO config; missing registry entry changes its space. | Prior learned-space coverage checks only 0.0.8 learned sources. | True versus false; real scoped constructor and exact released source hash. |
| `test_worker_preserves_absent_algorithm_config_for_release_default_selection` | Preserve missing config provenance through batch payloads; serializing absence as explicit `{}` loses legacy selection. | Prior one-episode provenance test passes `None` directly and skips batch serialization. | `{}` is not `None`; real worker payload builder, then real worker/episode dispatch, dev1001 horizon 1. |
| `test_every_release_arm_keeps_full_base_environment_and_mapping_dumps` | Preserve all fields and identities for every catalogued arm, including config-less arms; a registry omission changes released env fill-in. | Earlier fixture covers only hybrid/learned sources and one scenario suite. | Missing old PPO changes constructor dump; real resolver, env builder and policy overrides for all 183 arms. Full dumps, not only hashes; four existing unfrozen guards remain blocking. |

The last test loops all arms as one preservation contract: already-correct arms
are negative controls and naturally also pass before the fix. Its actual failure
is the omitted release source, without a separate current-default witness.
[review_fix_proof.json](review_fix_proof.json) lists every arm, recorded/current
mapping digest and complete dataclass comparison result. Full old dumps are in
`tests/fixtures/hybrid_defaults/base_released_arms.json.gz` (lossless, fixed gzip
timestamp); only root path fields are normalized to `<repo>`.

Changed existing tests: the shared planner observation fixture and the historical
v3 flat-frame fixture declare a valid successor field; their original evaluator
and velocity-frame assertions remain. Without this field the new fail-closed
guard raises before those unrelated assertions. The 48-scenario preservation
test explicitly scopes its release source, as a caller of released behavior must;
otherwise it intentionally describes new-input construction. No new seam or RNG.
The related three planner/default files pass **95 cases in 20.36s**; the final
review regression run additionally checks restored guide tolerance and actual goal completion.

Additional existing fixture corrections: the v4 speed/closed-loop fixtures,
proxemic-costmap and decomposition characterizations, and two map-runner selector
routing tests declare a valid successor field. The bridge's explicit sensor-less
case removes that key before flattening; it still tests both omission and retained
validity. Their speed limits, contact criteria, golden commands, frame conversion,
routing diagnostics and preservation assertions are unchanged.

| Changed fixture/test family | Behavior / credible regression | Existing gap | Seam / determinism / real path |
| --- | --- | --- | --- |
| Speed/braking and closed-loop v3/v4 tests | Keep their physical speed/contact/frame witnesses; weakening braking or converting v3 flat velocities must still fail. | Their historical fixtures predate the required sensor and would raise before the witness. | No production seam; fixed valid field, original deterministic real planners and drive integration. |
| Proxemic and decomposition characterizations | Preserve gate ordering, cost fields and golden commands; changed gates or scores still fail. | Sparse observations skip the new sensor contract. | No seam; fixed sensor value, unchanged real planner assertions. |
| Two selector-routing tests | Preserve selected planner and diagnostics; wrong dispatch still fails. | Inline sparse observations omit validity. | No seam; fixed field, real policy/selector/hybrid path. |
| Optional bridge test | Preserve omission and validity value through flattening; dropping the optional key still fails. | The shared fixture now includes the current sensor, so omission must be explicit. | No seam; fixed payload, real flatten/normalize functions. |

Before declaration corrections, the speed/characterization/proxemic files fail
23 cases at the new missing-sensor guard; after correction all 62 pass. The
dispatch group initially exposes two selector-fixture omissions and the bridge
fixture assumption; after correction all 193 pass. These existing tests preserve
old witnesses and are not presented as new fail-on-base bug tests. The ten new
bug cases still fail on the reviewed head for the defects described above.

The config-less release registry now binds 25 exact scenario/algorithm pairs
covering all 59 catalogued arms without an algorithm file. A new hybrid using
those scenario assets remains current. The existing 60 source entries are
unchanged; their exact bytes/dependencies are still verified.

| Additional test | Behavior / credible bug | Existing gap | Base failure; determinism; real path |
| --- | --- | --- | --- |
| `test_unknown_configless_hybrid_on_released_assets_uses_current_defaults` | An unrecorded hybrid arm must use current defaults even on registered scenario assets; a scenario-only fallback disables them. | Prior config-less coverage checks the recorded goal arm and skips a new algorithm on the same assets. | The actual episode records legacy rather than current on reviewed head; fixed dev1001, two sequential native one-step episodes, first direct then through the actual serialized worker. No production seam. |

Full preservation assertions now compare canonical sorted JSON text after only
repo-root redaction. They reject representation changes such as integer versus
float or boolean versus integer, in addition to field/value changes. The base
fixtures are unchanged. Negative controls that were already correct still pass
on base; the all-arm test fails for the actual omitted PPO source.

Full validation after the review fixes is recorded in [review_validation.json](review_validation.json):
**one full invocation, 42,973 passed / 28 failed / 69 skipped / 7 xfailed**.
The 28 failures are individually classified there. Nineteen sparse planner
fixtures lacked validity, five separate benchmark proxy/bridge inputs dropped
or omitted it, and three identity test doubles expected `{}` rather than `None`.
Those input/contract corrections retain their old regression witnesses.
The remaining failure is a real current-default static-exclusion mirror risk:
the historical diagnostic fixture now explicitly pins its original three false
controls; current-default symmetry is still reported separately to the author.

The affected 12 files pass 210 cases with one expected failure in 123.43 s.
After narrowing the compact bridge to retain its old current-goal target while
carrying only the observed validity bit, its final two files pass 22 cases in
61.93 s. No second full invocation or final-head full-suite pass is claimed.

| Additional corrected family | Bug/witness retained | Gap and before proof | Determinism / real path |
| --- | --- | --- | --- |
| Adaptive selector, topology and reverse fixtures (19 failures) | Keep profile selection, topology refusal/fallback/selection and rear-wall rejection assertions. | Sparse inputs omit the enabled sensor; full run raises before those witnesses. | Fixed declared successor; real adapters/evaluator; no production seam. |
| Goal-posterior proxy, compact native bridge and native-command reconstruction (5 failures) | Keep posterior consumption, native command/geometry response and real runner witnesses. | Proxy input omits declared validity; compact bridge loses actual navigator validity; native reconstruction drops supplied validity. | Proxy declares its successor; real bridge reads navigator only when the sensor is enabled; RPC copies only supplied validity. Missing input still fails closed. |
| Resume identity doubles (3 failures) | Preserve algorithm/dt/track identity separation and written-row assertions. | `dict(None)` fails before identity assertions after absence is preserved. | Deterministic real batch dispatch and identity encoder; optional mapping normalized only in test doubles. |
| Historical diagnostic mirror fixture | Preserve original reflected trace/outcome assertions at 0.1 mm, with original controls now explicit. | Missing control pins let new defaults change its diagnostic premise. The full run records a real new-default failure, retained as author evidence. | Real simulator, drive and planner; original test seed unchanged. Separate 15-episode dev1001 probe checks every switch in native typed configs. |

The separate [mirror probe](mirror_switch_probe.json.gz) checks static attribution:
static-only and all-on fail both reflections (maximum 0.374 m); all-off, sensor-only
and validity with sensor meet 0.1 mm. It is a synthetic diagnostic scene, not an
additional standard-scenario outcome denominator.
