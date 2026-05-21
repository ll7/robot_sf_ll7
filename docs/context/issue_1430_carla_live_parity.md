# Issue #1430 CARLA Live Replay Parity

Issue: [#1430](https://github.com/ll7/robot_sf_ll7/issues/1430)
Parent epic: [#872](https://github.com/ll7/robot_sf_ll7/issues/872)
Evidence bundle:
[`docs/context/evidence/issue_1430_carla_live_parity_2026-05-21/`](evidence/issue_1430_carla_live_parity_2026-05-21/)

## Outcome

The post-#1329 live replay path was attempted against the certified #1111 payload. The preferred
`imech036` route was not usable in this non-interactive run because SSH stopped at host-key
verification, so the documented fallback host `imech156-u` was used.

On `imech156-u`, the pinned Docker runtime reached a live CARLA server:

- Docker daemon: available.
- NVIDIA GPU/container runtime: available.
- CARLA image: `carlasim/carla:0.9.16`, digest
  `sha256:aaf1df22702780ece072069e23d03c4879b002ae028c79744b09c4c7ddbae953`.
- Client/server: CARLA `0.9.16` connected to `Carla/Maps/Town10HD_Opt`.
- Scenario: `pr_promoted_planner_smoke` from the certified #1111 manifest.

The run then failed closed before oracle replay metrics were produced:

```text
status: failed
mode: failed
reason: CARLA failed to spawn robot
```

## Interpretation

This is real live CARLA runtime evidence, but not Robot-SF/CARLA metric parity and not simulator
transfer evidence. It does show that the pre-#1329 static-geometry failure was not the observed
post-#1329 blocker: the live replay progressed past the earlier blanket static-obstacle rejection
and failed at robot actor spawning.

The #1110 parity adapter was run against the failed CARLA record and produced
`status: unavailable`, with every metric row unavailable because CARLA replay status/mode is
`failed`. That is the correct conservative outcome for #1430 unless a later run reaches
`oracle-replay`.

## Remaining Work

To satisfy the parent epic's positive metric-parity criterion, the next CARLA issue should diagnose
why CARLA failed to spawn the robot actor in the certified payload, then rerun the same live replay
until it either reaches `oracle-replay` or fails closed with a narrower actor-spawn condition.
Follow-up [#1437](https://github.com/ll7/robot_sf_ll7/issues/1437) tracks that work.

Do not treat this failed replay, setup-only smoke, image availability, or server/client
connectivity as benchmark success.

## Validation

Evidence commands:

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime live-replay \
  --manifest docs/context/evidence/issue_1111_carla_setup_smoke_2026-05-18/manifest.json \
  --max-steps 10 \
  --pull \
  --json
uv run python scripts/carla_bridge/compare_oracle_replay_metrics.py \
  --robot-sf docs/context/evidence/issue_1430_carla_live_parity_2026-05-21/robot_sf_metrics_reference.json \
  --carla docs/context/evidence/issue_1430_carla_live_parity_2026-05-21/live_replay_imech156.json \
  --output docs/context/evidence/issue_1430_carla_live_parity_2026-05-21/parity_report.json
```
