---
name: experiment-context
description: "Find the canonical config-first training or evaluation path, artifact lineage, and validation gates for a concrete experiment task in robot_sf_ll7."
---

# Experiment Context

## When to use

Use this skill when a specific training/evaluation task needs the canonical config-first command, artifact lineage, and validation gates before execution or interpretation.

## Read First

- `docs/AGENT_INDEX.md`
- `docs/dev_guide.md`
- relevant runbooks under `docs/training/`
- relevant configs under `configs/training/` or `configs/benchmarks/`
- `docs/context/issue_691_benchmark_fallback_policy.md` when benchmark runs are involved

## Workflow

1. Resolve the exact command or config entrypoint.
2. Verify expected output roots and artifact layout.
3. Capture host/runtime assumptions and dependencies.
4. Identify required validation or promotion gates.
5. Summarize the primary risk of deviating from canonical execution.

## Proof and Guardrails

- Prefer tracked config and command paths over ad-hoc overrides.
- If artifacts are missing or reproducibility assumptions are violated, classify the result as incomplete.
- Keep benchmark conclusions conditional on provenance and fallback mode status.

## Output

Return:

- exact config/command path,
- expected artifact root,
- required checks,
- the noncompliance risk if the canonical path is not used.
