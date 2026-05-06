# Issue #1015 Multi-Ped Adversarial Family Smoke

Related issues:

- Parent: <https://github.com/ll7/robot_sf_ll7/issues/870>
- Child: <https://github.com/ll7/robot_sf_ll7/issues/1015>

## Goal

This slice follows the #870 runtime path with concrete scripted family fixtures and metadata proof
for development smoke use. It remains deliberately short of benchmark-frozen certification.

## What Changed

- `configs/adversarial/group_squeeze_multi_ped_example.yaml` is now documented as a development
  smoke fixture and carries per-pedestrian role/lane metadata.
- `configs/adversarial/doorway_blocker_multi_ped_example.yaml` adds a second concrete scripted
  family fixture with deterministic seed 53.
- `materialize_multi_ped_scenario_payload(...)` now adds
  `metadata.adversarial_multi_ped_runtime`, including:
  - family,
  - schema version,
  - scenario seed,
  - pedestrian ids,
  - `evaluation_scope: development_stress_test`,
  - `certification_status: uncertified_development_smoke`,
  - `benchmark_frozen: false`.
- Policy-analysis episode records preserve this metadata through `scenario_params` because the
  scenario payload is copied into the record.

## Proof

TDD red proof:

```bash
uv run pytest \
  tests/adversarial/test_adversarial_search.py::test_multi_ped_adversarial_family_fixtures_reset_step_deterministically \
  tests/tools/test_policy_analysis_run.py::test_build_episode_record_preserves_multi_ped_adversarial_metadata -q
```

Before implementation, this failed because `adversarial_multi_ped_runtime` did not exist and the
doorway blocker fixture was missing.

After implementation, the same targeted command passed:

```text
3 passed in 36.58s
```

## Boundary

These fixtures are development stress tests only. They are not benchmark-frozen cases and should
not be used as benchmark evidence until they pass the repository scenario-certification and replay
proof path.
