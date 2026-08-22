# Copilot Instructions

`AGENTS.md` is the canonical repository instruction source. Treat this file as the Copilot-facing
entrypoint and keep it limited to Copilot-specific pointers that are not already covered there.
For current maintainer values and hard contracts, read
[`docs/maintainer_values.md`](../docs/maintainer_values.md), and for development workflow use the
official [dev_guide](../docs/dev_guide.md) as the primary reference.

## Copilot-Specific Instructions

- Use scriptable interfaces instead of CLI interfaces when possible.
- Source the environment before using python or uv: `source .venv/bin/activate`.
- For GitHub issue batches and Project #5 writes, follow the batch-first workflow in
  `docs/context/issue_713_batch_first_issue_workflow.md`.
- For interactive issue, PR, and project work, prefer GitHub MCP / GitHub app tools; keep `gh`
  for deterministic batch automation, score sync, and auth/debugging fallback.
- Follow the [coding-agent compatibility note](../docs/context/issue_728_coding_agents_compatibility.md).
- For any changes that affect users, update the `CHANGELOG.md` file.
- Link new documentation (sub-)pages in the appropriate section of `docs/README.md`.

All other workflow, validation, evidence, and publication rules live in `AGENTS.md`,
`docs/maintainer_values.md`, and `docs/dev_guide.md` — do not duplicate them here.
