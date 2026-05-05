# Issue 936 Multi-Ped Adversarial Overrides

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/936>

Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/870>

Depends on issue #923 / PR #924.

## Goal

Bridge the `adversarial-multi-ped.v1` schema toward existing Robot-SF scenario runtime surfaces
without running environments or claiming benchmark readiness. The new materializer converts a
validated `MultiPedAdversarialConfig` into deterministic `single_pedestrians` override dictionaries.

## Public Surface

- `robot_sf.adversarial.materialize.materialize_multi_ped_single_pedestrian_overrides(config)`
- Re-exported from `robot_sf.adversarial`.

Each materialized override records:

- pedestrian `id`,
- `start` and `goal` points,
- `speed_m_s`,
- `start_delay_s` as `spawn_time_s + delay_s`,
- a human-readable adversarial note,
- metadata with family, schema version, scenario seed, original per-ped metadata, spawn time, and
  delay.

## Scope Boundary

This is pure data conversion. It does not mutate scenario manifests, load maps, apply overrides,
step an environment, or certify generated adversarial cases. The output is a development stress-test
bridge and must not be treated as frozen benchmark evidence without later scenario certification and
runtime proof.

## Validation

Targeted TDD evidence:

```bash
uv run pytest tests/adversarial/test_adversarial_search.py -q
```

The RED run failed during collection with `ModuleNotFoundError: No module named
'robot_sf.adversarial.materialize'`, proving the tests covered the missing public materializer
surface. The GREEN run passed:

```text
26 passed in 22.90s
```

Before PR handoff, run the stacked readiness gate against the #923 branch:

```bash
git diff --check
BASE_REF=origin/923-multi-ped-adversarial-schema scripts/dev/pr_ready_check.sh
```

## Follow-Up Boundary

Future #870 children can wire these overrides into a scenario-loader or policy-analysis smoke path,
then prove reset/step determinism. Learned adversaries and frozen benchmark promotion remain out of
scope until certification and replay evidence exist.
