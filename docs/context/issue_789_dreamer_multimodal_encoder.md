# Issue 789: DreamerV3 multimodal encoder stop note

Date: 2026-04-28
Related notes:
- `docs/context/dreamerv3_program_full_handoff_2026_04_28.md`
- `docs/context/issue_578_608_609_dreamerv3_parity.md`
- `docs/context/issue_782_dreamerv3_pretraining_design.md`

## Goal

Determine whether the current RLlib DreamerV3 stack can accept a Dict observation of the form
`{"image": Box(3, 32, 32), "state": Box(...)}` without a substantive fork.

## Local investigation result

The stop condition is met on the current dependency stack (`ray==2.53.0`).

Evidence from the installed RLlib source after `uv sync --extra rllib`:

- `ray/rllib/algorithms/dreamerv3/dreamerv3_catalog.py`
  - `DreamerV3Catalog.build_encoder()` branches only on `observation_space.shape`.
  - 2D/3D spaces route to the Atari-style CNN encoder.
  - all other spaces route to a single MLP using `np.prod(self.observation_space.shape)`.
  - there is no Dict or multi-branch encoder path.
- `ray/rllib/algorithms/dreamerv3/utils/__init__.py`
  - `do_symlog_obs()` includes the explicit comment: `TODO (sven): Support mixed observation spaces.`

This means a Robot SF wrapper that emits `{"image": ..., "state": ...}` would not be enough.
The current DreamerV3 catalog, decoder selection, and world-model preprocessing expect a single
`Box` observation space and would need a Dreamer-specific extension.

## Outcome

- Do **not** add multimodal wrapper changes in `robot_sf/training/rllib_env_wrappers.py` on this
  branch.
- Do **not** add `benchmark_socnav_grid_br08_gate_multimodal.yaml` or
  `benchmark_socnav_grid_br08_full_multimodal.yaml` on this branch.
- Do **not** spend a dedicated #789 gate SLURM job on a configuration that the current RLlib
  catalog cannot consume.

This is a fail-closed decision, not a claim that multimodal Dreamer is impossible in principle.
It only means the current path is larger than the issue allows.

## Follow-up boundary

If maintainers still want multimodal DreamerV3, open a separate follow-up issue with one of these
explicit scopes:

1. Upstream contribution or local fork of `DreamerV3Catalog` and the related decoder/world-model
   plumbing for mixed observation spaces.
2. A separate representation-learning path outside RLlib, tracked independently from #578.

Do not fold either path back into #578, #608, or #609 without a new issue and proof plan.
