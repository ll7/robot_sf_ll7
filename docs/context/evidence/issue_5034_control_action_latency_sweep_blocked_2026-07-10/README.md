<!-- AI-GENERATED (robot_sf#5061, 2026-07-10) - NEEDS-REVIEW -->

# Issue #5034 — Control-action-latency fidelity sweep: fail-closed blocker (2026-07-10)

Fail-closed decision packet for issue #5034 ("execute the control-action-latency
fidelity sweep"). This is a **launch/readiness artifact only** — it is not
benchmark evidence, not simulator-realism evidence, not sim-to-real evidence, and
not paper-facing evidence. No benchmark episodes were run to produce it.

- `status`: `blocked`
- `result_classification`: `blocked_not_benchmark_evidence`
- `evidence_tier`: `diagnostic_only`
- `paper_facing`: `no`
- `schema_version`: `control-action-latency-sweep-preflight.v1`

## What this records

`control_action_latency_sweep_preflight.json` is the deterministic preflight
packet built from the shipped study config
`configs/research/fidelity_sensitivity_v1.yaml` at base commit `e128d3fa2`.

The sweep this issue asks to execute requires a `control_action_latency` axis in
the fidelity-sensitivity study config whose variants exercise the 0/100/300
ms-equivalent delays (action-latency steps 0, 1, 3). That axis is wired by
**PR #5026** (parent issue #4977), which is **not yet merged**. On the current
`origin/main` config the axis is absent, so the preflight fails closed:

- `decision`: `blocked`
- `axis_present`: `false`
- `missing_latency_steps`: `[0, 1, 3]`
- exact unmet prerequisite: merge PR #5026 (adds the axis and the
  `sim_config.action_latency_steps` field).

## Why not just run the campaign runner

The existing fixed-scope runner **is** launchable on this host, but only over the
five fidelity axes already on `main` — it has nothing to do with control-action
latency. Observed on `imech156-u` at base commit `e128d3fa2`:

```text
uv run python scripts/benchmark/run_fidelity_sensitivity_campaign.py \
  --fixed-scope-plan-only --require-launchable --plan-out output/fidelity_latency_plan
# preflight_decision=preflight_ready executable=True launched=False run_cells=126
# axes: clearance_radius, integration_timestep, observation_noise,
#       pedestrian_integration_scheme, social_force_speed_archetypes
```

Executing that 126-cell plan would produce a fidelity sweep with **no latency
axis** and could be mistaken for the sweep issue #5034 requires. The new
preflight (`scripts/benchmark/preflight_control_action_latency_sweep.py`) is the
guard that prevents that: it fails closed until the latency axis exists, and
flips to `ready` automatically once PR #5026 lands the same config.

## Regenerate

```bash
uv run python scripts/benchmark/preflight_control_action_latency_sweep.py \
  --config configs/research/fidelity_sensitivity_v1.yaml \
  --out output/issue_5034_preflight --require-ready
# decision: blocked  (exit 2) while PR #5026 is unmerged
```

The `git_head` and `date` fields vary by checkout/run; all other fields are
deterministic for a given config.

## Next empirical action (post-#5026)

1. Merge PR #5026 so the `control_action_latency` axis and
   `sim_config.action_latency_steps` land on `main`.
2. Re-run the preflight above; expect `decision: ready`,
   `observed_latency_steps: [0, 1, 3]`.
3. Preflight the fixed-scope plan
   (`--fixed-scope-plan-only --require-launchable`) and, if launchable, execute
   (`--fixed-scope-execute`) — a campaign run, out of scope for the
   implementation lane.
4. Promote the compact latency-cell evidence (success / collision /
   min-clearance per latency step) and update parent issue #4977 with the
   result classification.
