# Issue #3637 — Paper-grade reactivity-vs-replay rank study: preflight control layer

**Status:** evidence-control layer (no campaign run). **Evidence grade:** plan-preflight only — no
benchmark evidence, no rank-stability interpretation, no paper-facing claim.

Split from #3573 (closed diagnostic-complete). The diagnostic landed the quantifier (#3594) and the
paired-run + open-loop-replay pedestrian mode (#3612) on a small matrix (2 scenarios, goal + orca,
4 seeds). #3637 carries the **paper-grade** extension: the same ablation across ≥3 planners at a seed
budget sufficient for rank stability, with the replay limitation stated explicitly. See
[issue_3573_reactivity_ablation.md](issue_3573_reactivity_ablation.md) for the quantifier.

## What this slice adds

The **next evidence-control layer**: a plan-level preflight that gates the (not-yet-run) paper-grade
campaign so under-powered or mislabeled runs are caught **before** any compute is spent. It does
**not** run the benchmark, measure/interpret rank stability, or edit any paper/dissertation claim.

- `robot_sf/benchmark/reactivity_replay_preflight.py` — pure, side-effect-free checker
  (`reactivity_replay_rank_study_preflight.v1`). `build_preflight_manifest(plan)` returns a manifest
  with `status: ready|blocked`. Checks (all plan-level **preconditions**, not run output):
  - `planner_count` — ≥ `MIN_PLANNERS` (3) distinct planners;
  - `arms_present` — exactly the `reactive`/`replay` arms;
  - `paired_seeds` — both arms use the identical seed set (common random numbers);
  - `seed_budget` — ≥ `MIN_RANK_STABILITY_SEEDS` (20, the S20 floor) and > the #3573 diagnostic
    matrix (4 seeds);
  - `horizon` — ≥ 150 steps (the contrast is near-null below this on the diagnostic family);
  - `scenario_set_sha256` — when supplied, the named scenario-set file must match the packet digest;
  - `replay_limitation` — present, and `is_trajectory_playback` is `False`.
- `scripts/benchmark/preflight_reactivity_replay_rank_study_issue_3637.py` — thin CPU-only CLI over
  the checker. Exit `0` ready / `1` blocked / `2` usage error.
- `configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml` — the proposed run
  plan (planners, paired S20 seeds, scenario set + sha256, horizon, replay limitation) + the
  out-of-scope run/post-run commands.

## Replay limitation (canonical, must travel with every artifact)

`REPLAY_LIMITATION` / `REPLAY_IS_TRAJECTORY_PLAYBACK` now live in the canonical owner
`robot_sf/benchmark/reactivity_ablation.py` and are imported by the preflight (no second copy):

> "replay" = robot→pedestrian social-force term disabled in a **live** social-force sim
> (`peds_have_robot_repulsion=false`); **NOT** pre-recorded trajectory playback. Pedestrians still
> follow social-force dynamics among themselves but do not yield to the robot, so disabling
> reactivity tends to *reveal* intrusion hazard rather than hide it.

## Boundary

The preflight checks the **plan**, not sufficiency. Actual seed *sufficiency* (CI half-width /
rank-flip) is decided **post-run** by `scripts/tools/seed_sufficiency_gate.py`; no reactivity-rank
claim may be promoted until that gate plus claim-card review classify the executed bundle.

## Tests

`tests/benchmark/test_reactivity_replay_preflight_issue_3637.py` (18 tests): each precondition
pass/block path, the packet loader (shared vs per-arm seeds, malformed rejection), the shipped
packet (ready + scenario-set sha256 drift guard), and the CLI exit codes.
