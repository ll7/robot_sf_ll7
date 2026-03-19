# experiment-context

Use this skill when a task needs the current training/evaluation run context rather than generic
benchmark theory.

## Read First

- `docs/AGENT_INDEX.md`
- `docs/dev_guide.md`
- relevant runbooks under `docs/training/`
- relevant configs under `configs/training/` or `configs/benchmarks/`

## Focus

- canonical command path,
- config-first workflow,
- artifact lineage,
- host/runtime assumptions,
- validation or promotion gates.

## Output Expectations

Return:

- the exact config or command path to use,
- the artifact root that should contain outputs,
- the main risk if the user deviates from the canonical path.
