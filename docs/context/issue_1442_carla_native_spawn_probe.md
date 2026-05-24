# Issue #1442 CARLA Native Spawn Probe (2026-05-24)

Issue: [#1442](https://github.com/ll7/robot_sf_ll7/issues/1442)

Evidence:
[`docs/context/evidence/issue_1442_carla_native_spawn_probe_2026-05-24/`](evidence/issue_1442_carla_native_spawn_probe_2026-05-24/)

Related contract:
[`issue_1444_carla_coordinate_alignment_contract.md`](issue_1444_carla_coordinate_alignment_contract.md)

## Outcome

The #1442 live replay attempt found a narrower transfer boundary than #1440.

The existing certified #1111 payload still requires CARLA map projection:

```text
status: oracle-replay
mode: oracle-replay-adapted
robot spawn projection: 18.190854674447806 m
```

That result remains diagnostic only. It is not native or aligned parity evidence
under the #1444 contract.

An exploratory native-spawn probe then used the inverse Robot-SF coordinates of
the CARLA spawn that #1440 projected onto the road surface:

```text
Robot-SF start: x=2.449455738067627, y=-13.190784454345703
CARLA spawn:    x=2.449455738067627, y=13.190784454345703
```

That generated T0 payload reached:

```text
status: oracle-replay
mode: oracle-replay
robot_spawn.adapted: false
robot_spawn.strategy: exact
```

## Interpretation

Native CARLA replay is possible when the T0 payload is generated from
CARLA-aligned coordinates. The remaining blocker is not Docker, CARLA startup,
or robot actor spawning; it is producing a certified scenario/export path whose
coordinate semantics are intentionally CARLA-aligned and whose replay output
contains trajectory-level metrics for parity comparison.

The parity adapter output for the native probe is still `unavailable` because the
current T1 live replay JSON does not emit the metric fields needed by
`compare_oracle_replay_metrics.py`. This keeps the paper-facing claim boundary
intact: #1442 has native runtime evidence and a narrower next blocker, but not a
positive metric-parity claim.

## Validation

CARLA-free contract tests:

```bash
uv run pytest -q \
  tests/carla_bridge/test_t1_live_replay.py \
  tests/carla_bridge/test_parity.py \
  tests/carla_bridge/test_parity_cli.py \
  --no-cov
```

Result: `18 passed in 3.15s`.

CARLA runtime preflight:

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime preflight --json
```

Result: Docker daemon, GPU, NVIDIA container toolkit, ports, CARLA image, and
CARLA Python API all available.

Live replay checks:

- Certified #1111 payload: `oracle-replay-adapted`, `18.190854674447806 m`
  spawn projection.
- Native-spawn probe: `oracle-replay`, exact robot spawn, 10 replay steps,
  no adaptations.

Parity report:

```bash
uv run python scripts/carla_bridge/compare_oracle_replay_metrics.py \
  --robot-sf docs/context/evidence/issue_1442_carla_native_spawn_probe_2026-05-24/robot_sf_metrics_reference.json \
  --carla docs/context/evidence/issue_1442_carla_native_spawn_probe_2026-05-24/live_replay_native_probe.json \
  --output docs/context/evidence/issue_1442_carla_native_spawn_probe_2026-05-24/parity_report_native_probe.json
```

Result: `status=unavailable`, because no comparable CARLA metric fields were
available.

## Follow-Up Boundary

The next implementation should convert this probe into a durable aligned/native
scenario path by:

- defining a CARLA-aligned scenario/export fixture instead of deriving from an
  adapted replay result,
- recording the coordinate transform or native map contract before replay,
- emitting trajectory-level CARLA metrics from T1 replay,
- and only then comparing Robot-SF and CARLA metrics under the #1444 parity
  rules.
