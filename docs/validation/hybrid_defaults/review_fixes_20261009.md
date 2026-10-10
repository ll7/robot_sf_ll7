# Three review fixes, 2026-10-09

Base: `0ca61efee4bd92d3cb9a408fbffb4e90b76d7796` (PR #10166).
The 2026-10-08 defaults remain validity plus its sensor on, static exclusion opt-in.

Resume identity now includes `hybrid_default_set: current` for current-default runs.
Legacy runs omit that term and retain their historical IDs. Resume filtering selects
its policy from the same registered algorithm source or config-less release arm as
execution. Neither the resolved mapping nor its effective-config digest changes.

The PPO training factory now honors `env_overrides.include_goal_next_valid`, allowing
an old checkpoint's structured observation space to exclude the new sensor. The
planner checks whether it is v4 before requiring that sensor; v3 keeps its previous
observation contract and v4 still rejects a missing sensor when validity is enabled.

## Regression proof

Before any production edits, this command failed on the base above with three
behavioral failures. Run through `scripts/dev/run_worktree_shared_venv.sh --`:

```sh
uv run pytest \
  tests/benchmark/test_map_runner_resume_identity.py::test_resume_runs_current_defaults_with_a_legacy_result_present \
  tests/training/test_train_expert_ppo_contract.py::test_training_factory_honors_goal_validity_sensor_override \
  tests/planner/test_hybrid_default_review_regressions.py::test_v3_planner_does_not_require_the_v4_goal_validity_sensor \
  -n 2 -q
```

| Test | Failure on base | Protection and existing coverage gap | Determinism and real path |
| --- | --- | --- | --- |
| Resume | `assert [] == [(scenario, 1001)]` | Prevents skipping a current run because a legacy row exists. Existing algorithm/config-hash tests never changed the source-bound default set. | Fixed scenario and seed 1001; actual JSONL indexing, policy resolution, identity hashing and resume filter. Also pins the base legacy ID and compares write-time and resume-time payloads. |
| PPO factory | `next_valid` remained in the space with explicit `False` | Prevents an incompatible observation space when resuming an old checkpoint. Existing factory tests checked pickling and vector modes, not this override. | Actual factory and environment construction at seed 1001; checks the structured observation space for both false and true; closes each environment. |
| v3 | `ValueError: goal_next_validity_enabled requires observation next_valid` | Prevents imposing the v4 sensor contract on older planners. Existing missing-sensor tests exercise v4 only. | Fixed observation; actual v3 `plan` produces a positive, finite command without the field. |

Each test fails if its corresponding fix is reverted. None needs a test-only seam in
production code. These are deterministic contract checks, without training or a
benchmark campaign.

## Validation and legacy identity proof

The three touched test files and `tests/docs/test_release_decision_ledgers.py` passed:
**88 tests**. This includes all **183** released constructor/environment dumps,
**179** resolved-mapping digests and four unchanged rejection guards. A subsequent
run of the three regressions and `test_hybrid_default_compatibility.py` passed:
**26 tests**, including all nine released planner dumps and recorded mapping digests.
Ruff lint, formatting and `git diff --check` passed.

The separate identity audit executed the base identity module and the changed module
against every scenario in the 183-arm inventory, using development seeds 1001–1030.
All **117,990** legacy payload byte strings and episode IDs were identical. The
comparison supplies the same explicit empty algorithm mapping on both sides to
isolate the default-policy term; the independent released-arm test verifies real
resolved mappings and complete constructor/environment values. Both identity
streams hash to:

`e6c85a081b09250f4714ea2c84555e0d36922bf51225accba516c168b1f66a41`

The base literal pinned by the resume regression is
`resume-identity-smoke--1001--26366e490e464c9d`. All **62** registered source and
dependency files remain byte-identical to the base. The machine-readable result is
[review_fixes_20261009_identity_audit.json](review_fixes_20261009_identity_audit.json).
Historical campaign data, frozen configs and snapshots were not changed. No new
scientific result or hosted CI result is claimed by this repair.
