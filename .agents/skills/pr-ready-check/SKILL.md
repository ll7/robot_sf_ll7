---
name: pr-ready-check
description: Run the repository PR readiness pipeline using shared scripts/dev entry points (ruff fix/format,
  parallel tests, coverage, and docstring checks).
category: validation
kind: atomic
phase: verification
requires_write: false
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# PR Ready Check

## Purpose

Run the canonical local readiness gates that mirror repo PR policy before handoff or review.

## Workflow

1. Confirm repo guidance (`AGENTS.md`, `docs/dev_guide.md`, `.specify/memory/constitution.md`) when uncertain.
2. Classify the change using the `AGENTS.md` readiness matrix:
   - docs-only or instruction-only: inspect the diff, verify changed links or paths where practical,
     and run available lightweight markdown, index, skill, or sync checks.
   - workflow/tooling docs or skills: also run relevant checks such as
     `uv run python scripts/dev/check_skills.py --preflight <skill-name>` and
     `uv run python scripts/tools/sync_ai_config.py --check`.
   - runtime, benchmark, metric, schema, model-provenance, or paper-facing work: run executable
     proof appropriate to the claim.
3. Escalate to the full readiness pipeline when scripts, schemas, generated indexes, routing
   behavior, automation, runtime behavior, benchmark/metric/schema semantics, model provenance, or
   paper-facing claims are touched.
4. Ensure environment has `.venv` loaded or `source .venv/bin/activate` is available.
5. Run: `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`.
6. If needed, use overrides, e.g. `MIN_COVERAGE`, `GOAL_COVERAGE`.
7. Fix failures and rerun the same command until stable green.
8. If benchmark outputs or model artifacts are involved, check artifact persistence policy before handoff.

## Guardrails

- This skill runs validation; it does not perform PR creation or issue edits on its own.
- Do not require full PR readiness for low-risk docs/instruction-only changes unless the matrix
  escalation rules apply; record the cheap validation commands instead.
- When full readiness is required, do not stop at lint green: ensure parallel tests and diff gates
  are also clean.
- Treat benchmark fallback execution as unresolved unless explicitly scoped.

## Output

- Readiness command and result summary.
- Which checks failed/succeeded (ruff, parallel tests, changed coverage, docstring TODO diff gate).
- Any residual risks and required artifact persistence actions.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
