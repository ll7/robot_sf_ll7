# Question-First Experiment Registry

This directory records planned or active exploratory experiments before they run. It complements
GitHub issues, W&B artifacts, model registries, and publication bundles; it does not make local
`output/` files durable by itself.

Each record must state:

- `experiment_id`
- `issue` and `issue_url`
- `question`
- `hypothesis`
- `config`
- `command`
- `inputs`
- `outputs`
- `expected_artifacts`
- `evidence_grade`
- `paper_relevance`
- `status`

Use `paper_relevance: exploratory` for local pilots and early research runs. Use
`paper_relevance: paper_facing` only when every local `output/` artifact listed in `outputs` or
`expected_artifacts` has a durable `durable_reference`, such as a W&B artifact, model registry
entry, release asset, or tracked evidence manifest.

Validate the registry with:

```bash
uv run python scripts/tools/validate_experiment_registry.py experiments/registry.yaml
```
