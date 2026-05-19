# Issue #1169 CARLA Live Replay Evidence

Issue: [#1169](https://github.com/ll7/robot_sf_ll7/issues/1169)
Context note: [docs/context/issue_1169_carla_live_replay.md](../../issue_1169_carla_live_replay.md)

This bundle preserves compact JSON summaries from the first Docker-backed CARLA T1 live replay
attempt on May 18, 2026.

## Files

* `preflight_skip_api_before_pull.json` - CARLA Docker preflight before the pinned CARLA image was
  present locally. Docker, Linux x86_64, NVIDIA GPU, NVIDIA Container Toolkit, and ports were
  available; the missing capability was `carla-image`.
* `live_replay_no_pull_before_image.json` - `live-replay` without `--pull`, also failing closed on
  the missing pinned CARLA image.
* `live_replay_static_boundary_fixed.json` - `live-replay` after pulling `carlasim/carla:0.9.16`;
  the Python client connected to CARLA `0.9.16` on `Town10HD_Opt`, then replay failed closed
  because the certified T0 payload includes four static obstacles and static-geometry replay is
  not implemented.

## Checksums

```text
99a88fc176820693879a6df1abc0a4a2b43516d6707eb7a7ce78aebe0624f857  live_replay_no_pull_before_image.json
66e6d9ff1331b9eec7f752399c45e65c01bee0651a3ae454b121e531a4b6bdbe  live_replay_static_boundary_fixed.json
074f9c04211bfc76fd172910ef1a7453341153860c6f75011e2d7b9c3fa5d6c9  preflight_skip_api_before_pull.json
```
