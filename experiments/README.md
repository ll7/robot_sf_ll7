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

## Create / Review / Validate / Update Flow

### Create a draft experiment card

```bash
uv run python scripts/tools/create_experiment_card.py \
  --issue 2103 \
  --experiment-id issue_2103_example \
  --template benchmark-analysis \
  --output-root output/experiments/issue_2103_example
```

Available templates: `benchmark-analysis`, `planner-ablation`, `figure-table-pack`.

This writes:

- `output/experiments/<experiment-id>/<experiment-id>.yaml` - the experiment record,
- `output/experiments/<experiment-id>/CHECKLIST.md` - validation and promotion checklist.

Generated records contain `TODO` placeholders that must be filled before the card is actionable.

### Review

Edit the generated YAML to replace all `TODO` placeholders with concrete config paths,
commands, and hypothesis details.

### Validate

```bash
# Validate the full registry after registering the card.
uv run python scripts/tools/validate_experiment_registry.py experiments/registry.yaml
```

### Update

1. Edit the record YAML when values change.
2. Re-validate with `validate_experiment_registry.py`.
3. Register the card in `experiments/registry.yaml` (add the filename to the `records` list).
