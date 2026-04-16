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
2. `CLAUDE.md` when the agent runtime supports it
3. `memory/MEMORY.md` for stable repo-local memory
4. `.specify/memory/constitution.md`
5. `docs/dev_guide.md`
6. `docs/code_review.md`
7. `.agent/PLANS.md` for non-trivial work

Then branch by task type:

- benchmark semantics: `docs/benchmark_spec.md`, `docs/benchmark.md`
- planner family support: `docs/benchmark_planner_family_coverage.md`
- training/eval workflow: `docs/AGENT_INDEX.md`, `docs/training/`
- issue execution history and handoff knowledge base: `docs/context/`
- context note workflow: `docs/context/README.md`

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
- `docs/context/`: linked execution notes, issue-specific evidence logs, and agent handoff context
- `memory/`: repo-local Markdown memory tree for stable cross-session facts and concise retrieval

### Existing agent workflow surfaces

- `.agents/skills/`: execution-oriented skills plus repo-local context-pack skills for benchmark,
  planner, experiment, and documentation work
- `.agents/skills/autoresearch/` and `.agents/skills/auto-improvement/`: repo-local skills for
  measurable improvement loops and smaller refinement passes
- `.agents/skills/context-map/` and `.agents/skills/what-context-needed/`: context-gathering skills
  for locating the right files or requesting the minimum missing context
- `.agents/skills/context-note-maintainer/`: note-creation and stale-note-maintenance workflow for
  durable Markdown handoff context
- `CLAUDE.md` + `memory/MEMORY.md`: startup-oriented memory/index pair for agent runtimes that
  support imported project instructions
- `.agents/skills/quality-playbook/`, `.agents/skills/agentic-eval/`,
  `.agents/skills/review-and-refactor/`, and `.agents/skills/update-docs-on-code-change/`: quality
  and maintenance skills for proof-first changes, AI-output evaluation, narrow refactors, and
  doc-sync

## Working Style Defaults

- Prefer config-first commands over long CLI override chains.
- Use `scripts/dev/` entrypoints where available.
- Preserve benchmark claims conservatively.
- Treat planner provenance as part of correctness, not just documentation polish.
- Distinguish observed evidence from hypothesis in findings and reports.
- Prefer `autoresearch` for measurable improvement loops and `auto-improvement` for targeted
  refinement passes.
- Prefer `context-map` before multi-file work, `what-context-needed` when context is missing,
  `quality-playbook` for proof-first planning, `agentic-eval` for AI workflow outputs,
  `review-and-refactor` for surgical improvements, and `update-docs-on-code-change` when a code
  change would stale docs.

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
