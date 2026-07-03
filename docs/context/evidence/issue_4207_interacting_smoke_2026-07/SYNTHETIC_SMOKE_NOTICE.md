# Synthetic Smoke Provenance Notice

This evidence packet is **not** the product of a robot/pedestrian physics simulation. It is built
in report-only mode (`--episodes-jsonl`) from a hand-authored, deterministic episode fixture:

- Fixture: `tests/benchmark/fixtures/issue_4207_interacting_smoke/interacting_smoke_episodes.jsonl`
- Probe config: `configs/benchmarks/issue_4207_interacting_smoke_probe.yaml`
- Gate spec: `configs/benchmarks/release_gates/issue_4207_interacting_smoke_gates.yaml`
- Scenario descriptor (proposed geometry, unverified): `configs/scenarios/single/issue_4207_interacting_smoke.yaml`

## What it proves

- The certification-transfer interaction-validity guard added in PR #4308 produces
  `model_sensitivity_exercised = true` end-to-end when fed an interacting scenario family, with all
  16 transfer cells classified `interacting` and a genuine interacting pass/fail flip on the `ppo`
  arm.

## What it does NOT prove

- It does not show that any real planner behaves this way, nor that the
  `configs/scenarios/single/issue_4207_interacting_smoke.yaml` geometry actually produces near-field
  contact under simulation. The metric values are fixture design choices, not measurements.
- No deployment, real-world safety, benchmark-strength, or paper/dissertation claim follows.

## Deferred follow-up

The physics-verified geometry/spawn CPU re-run — running the proposed interacting scenario matrix
through the real 4-arm probe and confirming `robot_ped_within_5m_frac > 0` from simulation — remains
the tracked next step for issue #4207. This slice unblocks it by proving the guard's positive path
and providing a reusable interacting family template.
