# Comparing resolved configs and explaining semantic drift

Raw YAML diffs overstate and understate config drift at the same time: they show
formatting and provenance noise as changes, and they hide values that arrive
through `base_config` inheritance and deep merge. This walkthrough compares two
training configs **after canonical resolution** and labels every changed leaf.

## Run it

```bash
uv run python examples/advanced/38_compare_resolved_configs.py \
  --left examples/fixtures/config_drift/left.yaml \
  --right examples/fixtures/config_drift/right.yaml --json
```

The fixture pair shares `base.yaml`. `right.yaml` overrides one declared
semantic key (`seed`), replaces a scenario list, changes the training hidden
sizes, and edits provenance, presentation, and output-location fields.

## How to read the report

- `verdict`: `identical`, `comparable`, or `not_comparable`.
- `changes[*].change_class`: `semantic`, `provenance`, `execution_environment`,
  `presentation_only`, or `unknown`.
- `changes[*].contributes_to_identity`: whether the field participates in the
  comparability digest.
- `changes[*].left_origin` / `right_origin`: the config file that declared the
  value (or `absent`), so inherited values are visible.
- `identity_digest`: digest over identity-contributing leaves only. Equal
  identity digests mean presentation, provenance, and execution-environment
  edits do not change comparability identity.

The classifier is conservative by construction. Only keys the benchmark
governance policy declares identity-bearing (schema/version markers, seed and
seed-policy keys, action semantics, and the observation/metric/model contract
markers) are labeled `semantic`. Every changed leaf the current owners do not
classify is labeled `unknown`, and any `semantic` or `unknown` difference makes
the pair `not_comparable`. The example never guesses a field policy and never
claims scientific comparability beyond the declared policy.

## Canonical owners reused

- `base_config` inheritance and deep merge:
  `scripts.training.train_ppo._load_expert_training_config_mapping`.
- Identity-bearing policy categories: `docs/benchmark_governance.md`.

## Limitations

- The `unknown` class is intentionally incomplete; it marks fields that the
  repository does not classify rather than inventing semantics.
- The example compares configuration identity only. Two configs with equal
  identity digests are not automatically valid benchmark evidence.
- Unresolved interpolation, duplicate YAML keys, parent-directory escapes, and
  malformed schemas fail closed instead of being compared.
