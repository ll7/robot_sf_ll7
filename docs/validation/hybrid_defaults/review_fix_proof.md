# Review regression proof

Diagnostic-only. Before-fix head: `fcadacd13f71902635fc7008960aa17c2d95e75c`.
The all-on author choice remains unchanged. The baseline snapshots were captured
from immutable main `9d8dac2a140f00dadf2c8c2621977bcee96c83e0`, without reset/step.
The newer main changes only the release runbook and its test, not these builders.

```sh
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/planner/test_hybrid_default_review_regressions.py -n 2 -q
```

With the changed production files restored to the reviewed head, the same nine
cases fail for their intended bugs: **9 failed in 9.45s**. Full output:
[review_fix_fail_before.txt](review_fix_fail_before.txt). With fixed production
bytes: **9 passed in 19.81s**. No test-only production seam was added.
All inputs are fixed; builder checks do not reset or step an environment.

| Test (cases) | Behavior / credible bug | Why existing coverage misses it | Before-fix witness; deterministic real path |
| --- | --- | --- | --- |
| `test_terminal_goal_at_022_m_keeps_tracking_before_environment_success` (3) | Keep tracking at 0.22 m until navigator success; reintroducing the 0.25 m stop deadlocks. Also assert actual completion stops, route-guide tolerance is restored, and validity-off retains its stop. | Prior validity tests only select a successor far from terminal completion. | `GOAL_STOP` at 0.22 m; real planner and route guide, real radius/zone navigators or unbound mode; fixed geometry, no RNG. |
| `test_enabled_validity_rejects_a_missing_sensor_field` (2) | Fail closed on absent nested/flat validity; an implicit valid default silently ignores the contract. | Prior sensor tests exercise present fields or opt-out. | `DID NOT RAISE`; real planner observation extraction, fixed fields. |
| `test_new_env_and_planner_share_defaults_on_registered_release_scenario` | New inputs use one current default source; inferring env defaults from scenario assets produces incompatible pairs. | Prior runtime test scopes both inside the runner. | Env false versus planner true; actual separate builders on the released matrix, no reset/step. |
| `test_released_002_ppo_constructor_uses_legacy_sensor_defaults` | Preserve the omitted sensor in the older PPO config; missing registry entry changes its space. | Prior learned-space coverage checks only 0.0.8 learned sources. | True versus false; real scoped constructor and exact released source hash. |
| `test_worker_preserves_absent_algorithm_config_for_release_default_selection` | Preserve missing config provenance through batch payloads; serializing absence as explicit `{}` loses legacy selection. | Prior one-episode provenance test passes `None` directly and skips batch serialization. | `{}` is not `None`; real worker payload builder with absent input. |
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
nine-case run additionally checks restored guide tolerance and actual goal completion.
