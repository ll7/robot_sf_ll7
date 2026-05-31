# Quickstart Map

Use this page as the docs-site entry point for first-run and local development paths. The canonical
guides remain in their existing Markdown locations.

## First Local Checks

- [Development Guide](dev_guide.md) covers setup, test commands, quality gates, and local workflow.
- [Runtime Requirements](dev_runtime_requirements.md) lists non-`uv` host tools and optional Docker
  support.
- [Benchmark Runner And Metrics](benchmark.md) includes the local smoke benchmark demo:
  `uv run python scripts/demo/run_robot_sf_smoke.py`.
- The social-navigation benchmark quickstart lives outside the Sphinx source tree at
  `specs/120-social-navigation-benchmark-plan/quickstart.md`.

## Discovery Paths

- [Scenario Zoo](scenario_zoo/index.md) summarizes maintained and emerging scenario families.
- [Planner Zoo](planner_zoo/index.md) summarizes runnable, diagnostic, learned-policy, and blocked
  planner rows.
- [Benchmark Suites Map](benchmark_suites.md) points to the benchmark suite and campaign surfaces.
- [Evidence Bundles](context/evidence/README.md) explains which generated artifacts are durable
  enough to preserve in git.

## Agent And Contributor Orientation

- [AI Coding Workflow](ai/ai-workflow.md) describes the issue-to-PR workflow used by agents.
- [Agent Index](AGENT_INDEX.md) collects agent-facing training, benchmarking, observation, and
  artifact entry points.
- [Maintainer Values](maintainer_values.md) is the compact source of truth for validation depth and
  research-progress priorities.
