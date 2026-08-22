# Robot SF – Development Guide

[← Back to Documentation Index](./README.md)

This is the concise contributor landing page. Use it to set up a worktree, choose the right
validation tier, and find the canonical procedure. The [developer guide index](./developer-guide.md)
organizes architecture, planner, scenario, tooling, and research-facing topics; the
[detailed compatibility reference](./dev_guide_reference.md) preserves older deep links while
procedures migrate to focused guides.

## First-use path

1. Read [`AGENTS.md`](../AGENTS.md) and [`docs/maintainer_values.md`](./maintainer_values.md) for
   precedence, safety, evidence, and worktree rules.
2. Check host capabilities with
   [`docs/dev_runtime_requirements.md`](./dev_runtime_requirements.md), then run the setup below.
3. Create a linked worktree before editing, pushing, or running PR validation. Follow the
   [worktree lifecycle guide](./dev/worktree_lifecycle.md).
4. Pick the cheapest validation tier that proves the change. Use the
   [local-CI guide](./dev/local_ci.md) for shared helpers and PR readiness.

## Setup

<a id="setup"></a>

```bash
scripts/dev/check_runtime_requirements.sh
uv sync --all-extras
source .venv/bin/activate
uv run pre-commit install
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; print('Import successful')"
```

Use `uv sync` for the core environment or a named extra when the touched path needs optional
dependencies. `uv sync --all-extras` is the canonical complete local setup; CARLA remains an
explicit opt-in group for CARLA-capable worktrees.

## Choose validation

| Change | Minimum proof | Canonical entry point |
| --- | --- | --- |
| Docs or instructions | Diff, changed links/paths, lightweight docs checks | `scripts/dev/check_docs_evidence_integrity.py` |
| Workflow, helper, or skill | Focused tests plus schema/sync checks | [`docs/dev/local_ci.md`](./dev/local_ci.md) |
| Runtime code | Focused tests, Ruff, and format | `scripts/dev/run_tests_parallel.sh` |
| Benchmark, metric, schema, provenance, or paper claim | Executable proof with provenance and caveats | [`docs/code_review.md`](./code_review.md) |

Escalate to `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` for scripts, schemas, generated
indexes, routing, runtime, benchmark, metric, provenance, or paper-facing changes. Fallback or
degraded execution is diagnostic only, never success evidence.

Coverage remains opt-in locally. The `coverage-gate` and `changed-coverage-gate` CI jobs combine
their shard data and enforce the workflow's configured thresholds; the local wrapper is configured
to measure only the `robot_sf/` package. Auxiliary `fast-pysf` sources are not included in wrapper reports.
CI publishes the `changed-coverage.v1` result for changed-file enforcement, while `coverage-gate`
enforces the configured 85.0% total threshold. The coverage configuration omits `fast-pysf/tests/*`
and `fast-pysf/examples/*` from the measured source scope. See the [coverage guide](./coverage_guide.md)
for source scope, reports, and explicit coverage commands.

## Worktree and local-CI procedures

- [Worktree lifecycle](./dev/worktree_lifecycle.md) — create, bootstrap, validate, preserve, and
  retire linked worktrees safely.
- [Local CI and PR readiness](./dev/local_ci.md) — dependency profiles, shared environments,
  scratch capacity, focused tests, and final readiness.
- [Agent workflow entrypoints](./ai/agent_workflow_entrypoints.md) — canonical `uv run` and
  validation commands.
- [Batch-first issue workflow](./context/issue_713_batch_first_issue_workflow.md) — issue and
  Project #5 batching rules. Canonical path: `docs/context/issue_713_batch_first_issue_workflow.md`.
- [Coding-agents compatibility note](./context/issue_728_coding_agents_compatibility.md) —
  retrieval, planning, execution, and verification across providers.

## Canonical topic owners

- [Developer guide index](./developer-guide.md) — architecture and contribution navigation.
- [Runtime requirements](./dev_runtime_requirements.md) and [environment setup](./ENVIRONMENT.md).
- [Benchmark governance](./benchmark_governance.md), [research guide](./research-guide.md), and
  [code review guide](./code_review.md).
- [Coverage guide](./coverage_guide.md), [quality report guide](./quality_report_guide.md), and
  [dev scripts](../scripts/dev/).
- Canonical skills live in `.agents/skills/`; use the skill's [`SKILL.md`](../.agents/skills/README.md)
  for task-specific execution contracts.
- The current Issue #5303 promotion entrypoint is
  `scripts/tools/check_issue_5303_search_promotion_contract_v2.py --identities`; its identity mode is
  side-effect-free and the historical v1 path cannot authorize promotion.

## Compatibility anchors

These short stubs preserve common inbound links from examples, docs, and older checkouts. The
procedure itself lives in the detailed reference or the linked canonical guide. The complete
source-to-target inventory is [`dev_guide_anchor_migration.yaml`](./dev_guide_anchor_migration.yaml).

<a id="quickstart"></a>
### Quickstart

See the [examples quickstart walkthrough](./dev_guide_reference.md#examples-quickstart-walkthrough).

<a id="environment-factory"></a>
### Environment factory

See the [environment factory procedure](./dev_guide_reference.md#environment-factory-pattern-critical).

<a id="baseline-policies"></a>
### Baseline policies

See the [advanced feature and policy examples](./dev_guide_reference.md#advanced-feature-demos).

<a id="feature-extractors"></a>
### Feature extractors

See the [advanced feature and policy examples](./dev_guide_reference.md#advanced-feature-demos).

<a id="pedestrian-environments"></a>
### Pedestrian environments

See the [environment factory procedure](./dev_guide_reference.md#environment-factory-pattern-critical).

<a id="advanced-feature-demos"></a>
### Advanced feature demos

See the [advanced feature demos](./dev_guide_reference.md#advanced-feature-demos).

<a id="planner-selection-visibility-vs-classic-grid"></a>
### Planner selection

See the [planner selection procedure](./dev_guide_reference.md#planner-selection-visibility-vs-classic-grid).

<a id="coverage-workflow"></a>
### Coverage workflow

See the [coverage workflow](./coverage_guide.md) and the detailed [coverage section](./dev_guide_reference.md#coverage-workflow-explicit-opt-in).

<a id="testing-strategy-unified-test-suite"></a>
### Testing strategy

See the [unified test-suite section](./dev_guide_reference.md#testing-strategy-unified-test-suite).

<a id="cicd-expectations"></a>
### CI/CD expectations

See the [CI/CD expectations](./dev_guide_reference.md#cicd-expectations) and [local-CI guide](./dev/local_ci.md).

<a id="run-tracker--history-cli"></a>
### Run tracker and history CLI

See the [run tracker procedure](./dev_guide_reference.md#run-tracker--history-cli).

<a id="per-test-performance-budget"></a>
### Per-test performance budget

See the [performance budget](./dev_guide_reference.md#per-test-performance-budget).

<a id="section"></a>
### Generic section compatibility anchor

This placeholder preserves the example fragment used by documentation templates; link to a named
section in the [developer guide index](./developer-guide.md) instead.

## Maintenance rule

Keep this landing page focused on first-use decisions and compatibility stubs. Add detailed
procedures to an existing canonical topic owner or a new task guide only when no owner exists, then
link it from [`docs/developer-guide.md`](./developer-guide.md). Do not copy a second full workflow
back into this file.
