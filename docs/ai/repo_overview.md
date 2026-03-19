# AI Repo Overview

This document is the shortest reliable orientation for Codex-style agents working in
`robot_sf_ll7`.

## Mission

`robot_sf_ll7` is a social-navigation simulation and benchmark repository. It combines:

- Gymnasium-compatible robot and pedestrian environments,
- benchmark runners and metric tooling,
- baseline planners and learned-policy evaluation,
- reproducible training and artifact workflows,
- paper-facing benchmark documentation and provenance notes.

The repository is optimized for reproducible experiments, benchmark comparability, and issue-driven
delivery rather than generic robotics breadth.

## First Files To Read

Start here before expanding:

1. `AGENTS.md`
2. `.specify/memory/constitution.md`
3. `docs/dev_guide.md`
4. `code_review.md`
5. `.agent/PLANS.md` for non-trivial work

Then branch by task type:

- benchmark semantics: `docs/benchmark_spec.md`, `docs/benchmark.md`
- planner family support: `docs/benchmark_planner_family_coverage.md`
- training/eval workflow: `docs/AGENT_INDEX.md`, `docs/training/`
- issue execution history: `docs/context/`

## Key Repository Areas

### Core library

- `robot_sf/gym_env/`: factory-based environment entrypoints and unified configs
- `robot_sf/benchmark/`: episode schema, runner, aggregation, metrics, reporting
- `robot_sf/sim/`: simulator/backend glue
- `robot_sf/nav/`: planners, occupancy, map/planning helpers
- `robot_sf/render/`: playback and visual tooling

### Reproducibility surfaces

- `configs/`: canonical YAML configs for training, benchmarks, scenarios, and baselines
- `scripts/`: runnable entrypoints and validation helpers
- `output/`: git-ignored canonical artifact root
- `docs/context/`: execution notes and issue-specific evidence logs

### Existing agent workflow surfaces

- `.codex/skills/`: execution-oriented skills plus repo-local context-pack skills for benchmark,
  planner, experiment, and documentation work

## Working Style Defaults

- Prefer config-first commands over long CLI override chains.
- Use `scripts/dev/` entrypoints where available.
- Preserve benchmark claims conservatively.
- Treat planner provenance as part of correctness, not just documentation polish.
- Distinguish observed evidence from hypothesis in findings and reports.

## Common Failure Modes

- changing observation normalization without checking learned-policy contract,
- broadening scenario defaults without documenting the comparison shift,
- presenting experimental planners as paper-grade baselines,
- adding ad-hoc scripts instead of extending reusable library/config surfaces,
- documenting results without a committed command/config path.

## Minimum Output For High-Risk Tasks

When the task touches benchmarks, planners, or paper-facing docs, include:

- what changed,
- why it matters,
- what validation ran,
- what limitation or follow-up remains.
