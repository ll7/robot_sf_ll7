# Issue #5578 robot speed-tier sweep — disjoint-seed activation preflight

> **NOT BENCHMARK EVIDENCE — DISJOINT-SEED ACTIVATION CHECK ONLY**

This directory holds the compact, explicitly non-evidence preflight record produced
by issue #6101. It is a manipulation-activation check: it proves the robot speed-cap
intervention reaches the real Robot SF runtime and measurably activates across the
three frozen speed tiers (2.0 / 3.0 / 4.0 m/s) **before** the 2,160 registered
episodes are committed by the downstream campaign lane (#6102).

## What it proves

- The frozen speed-cap variant flows to the real `BicycleDriveRobot` drive model
  through the canonical runner surfaces (`apply_variant`, `_robot_speed_cap`,
  `_robot_angular_cap`, `_env_action`, `make_robot_env`).
- The resolved cap matches the planned cap for every tier (no silent fallback).
- The non-nominal tiers (3.0 and 4.0 m/s) satisfy the activation rule frozen by
  #6100 (`fraction_above_2_0_mps >= 0.05 OR realized_speed_peak_m_s > 2.2`).

## What it is NOT

- It is not benchmark evidence and must not be used to tune harm thresholds, choose
  favourable scenarios/planners, or preview the registered primary-outcome verdict.
- It uses a goal-saturating command probe (the canonical activation probe for a
  speed-cap intervention), not a planner-behaviour measurement.
- It uses only disjoint seeds (211-214) **outside** the registered 111-140 block;
  no registered seed is executed or modified.

## How to regenerate

```bash
uv run python scripts/benchmark/run_issue_5578_speed_tier_campaign.py --preflight
```

The artifact (`issue_5578_activation_preflight.json`) records the git SHA, the exact
command/environment manifest, the disjoint pilot seeds, planned versus resolved
cap/acceleration/deceleration values, commanded and realized speed summaries,
cap-saturation / fraction-above-2.0 summaries, native execution status, and the
binary activation-gate result.

## Provenance

- **Seeds**: disjoint pilot seeds 211-214 (outside the registered 111-140 block).
- **Config**: no tunable algorithm config is used — the speed-tier manifest is
  compiled by `scripts/benchmark/run_issue_5578_speed_tier_campaign.py`, and the
  probe reads/writes through the fixed
  `command_environment_manifest` recorded in the JSON artifact (env/runner/cap
  reader function paths).
- **Hash**: the JSON artifact's `git_provenance.git_head` field records the exact
  commit SHA the preflight ran against, for reproducibility.
