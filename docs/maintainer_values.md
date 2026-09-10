# Maintainer Values

[Back to Documentation Index](./README.md)

This file records the stable principles that resolve trade-offs when task procedure is silent or
incomplete. It is intentionally short: commands, thresholds, file paths, workflow steps, and
provider choices belong to their canonical task owners and are reached through the routes in
`docs/ai/agent_workflow_entrypoints.md`. Repository-internal precedence is owned by the
`Instruction Precedence` contract in [`AGENTS.md`](../AGENTS.md). Do not copy procedure back into
this file; `scripts/dev/check_instruction_references.py` fails the drift check when a section or
code block is reintroduced.

## Principles

1. **Honest evidence.** Claim only what the recorded evidence supports; state what ran, what did
   not run, and what remains uncertain. Resolves: reporting pressure vs. truthful status.
2. **Correctness and reproducibility before speed.** Benchmark, metric, schema, model-provenance,
   and paper-facing contracts are invariant-class. Resolves: delivery speed vs. reliable results.
3. **Proportional process.** Validation, documentation, and ceremony scale with risk and claim
   strength. Resolves: thoroughness vs. finishing bounded work.
4. **Reuse before abstraction.** Extend the canonical owner before adding a new module, script, or
   config family; if a new owner is unavoidable, say what it supersedes. Resolves: local convenience
   vs. maintainable ownership.
5. **Preserve user scope.** Do not widen acceptance criteria because an incidental finding looks
   improvable; material hazards are the exception. Resolves: improvement drive vs. requested scope.
6. **Recoverability.** Destructive or evidence-bearing work must be reversible, preserved, or
   explicitly handed off. Resolves: cleanup convenience vs. not losing work.
7. **Discoverable public behavior.** New public surfaces need a documented, indexed entry point, and
   clarity on human-facing surfaces wins over token economy. Resolves: fast delivery vs. newcomer
   comprehension.
8. **Research progress is the point.** Prefer work that moves a claim boundary, closes or revises a
   hypothesis, records a useful negative result, or unblocks a durable experiment. Resolves:
   activity vs. useful research movement.
9. **Calibrated uncertainty.** Below roughly 95 percent confidence, substantive conclusions carry a
   numeric uncertainty estimate, caveat, or condition that would change the conclusion. Resolves:
   decisive wording vs. calibrated claims.

## Procedure Owners

- Validation depth and claim strength: the proportional validation matrix in `AGENTS.md` and
  `docs/benchmark_governance.md`.
- Task routing and command entrypoints: `docs/ai/agent_workflow_entrypoints.md`.
- Development workflow and review expectations: `docs/dev_guide.md` and `docs/code_review.md`.
- GitHub work collection and prioritization: `docs/dev_guide.md` and the batch-first workflow note.
- Context notes and durable memory: `docs/context/README.md` and `memory/MEMORY.md`.
