# Code Review Guide

Use this review guide for benchmark, planner, training, and documentation changes that can affect
published claims, reproducibility, or agent workflows.

## Review Priorities

Check these in order:

1. Correctness of user-visible behavior and public contracts.
2. Benchmark credibility and provenance.
3. Reproducibility of commands, configs, and artifacts.
4. Scope discipline: no hidden workflow churn or speculative infrastructure.
5. Test and documentation coverage proportional to the risk of the change.

## Benchmark-Credibility Review

For benchmark-facing changes, explicitly verify:

### Evaluation semantics
- Does the change alter success, collision, timeout, or metric semantics?
- Are fallback, skip, and fail-fast paths still explicit and testable?
- Are benchmark categories (`diagnostic`, `classical`, `learning`) still accurate?
- Does any report wording overstate what the benchmark actually measures?

### Observation normalization
- Are observation keys, bounds, clipping, and dtype contracts preserved or versioned?
- If a learned policy is involved, does the runtime observation contract still match the training contract?
- Are adapter transformations documented when planner inputs differ from env-native observations?

### Scenario distributions
- Does the scenario set, seed policy, or map pool change what is being compared?
- Are scenario family changes reflected in configs, docs, and interpretation guidance?
- Does any new default silently bias benchmark outputs toward a subset of scenarios?

### Reproducibility
- Is there a committed config or canonical command for the new behavior?
- Are generated outputs still rooted under `output/`?
- Do docs and PR text identify the exact command path needed to reproduce the result?
- If results depend on optional extras, hardware, or third-party assets, is that dependency explicit?

### Upstream provenance
- For vendored, wrapped, or adapter-backed planners: is the upstream source still identifiable?
- Are licenses, checkpoints, model origins, and wrapper boundaries documented?
- Does the change preserve the claim that a result is original-code-backed rather than a local reimplementation?

## Planner Integration Review

For planner additions or modifications, review:

- input contract: required observation/state fields are explicit,
- output contract: action space and kinematics adaptation are explicit,
- fallback policy: missing dependencies or artifacts fail clearly,
- provenance: upstream repo/model references remain auditable,
- benchmark readiness: paper-facing vs experimental status is still correctly labeled.

## Documentation Review

Docs changes should be rejected if they:

- introduce benchmark claims without a reproducible command path,
- blur the line between observed evidence and future intent,
- describe experimental planners as baseline-ready,
- omit caveats around optional dependencies or upstream source limits.

Prefer docs that point to canonical config files, scripts, and issue execution notes instead of freehand operational prose.

## Tests And Validation

Apply Principle XIII from `.specify/memory/constitution.md`:

- fix high-signal tests immediately,
- challenge flaky or low-value tests before investing in them,
- add tests when public contracts or benchmark semantics change,
- keep validation commands in PR text explicit and reproducible.

Recommended validation gate for broad changes:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

For docs-only or context-stack changes, still verify the modified markdown paths and any repo-local skill references.
