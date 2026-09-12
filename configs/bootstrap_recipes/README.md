# Bootstrap Recipes

Frozen, sanitized bootstrap recipes for the active execution classes. Each recipe records the exact
safe sequence that created or activated one environment: prerequisites, source/lock identity,
ordered argument-vector steps (`setup`, `probe`, `cleanup`), expected probes, outputs, private
substitutions, and verification status.

| Recipe | Execution class | Verification |
| --- | --- | --- |
| [cpu_batch.v1.json](cpu_batch.v1.json) | `cpu_batch` | verified |
| [gpu_training.v1.json](gpu_training.v1.json) | `gpu_training` | verified |
| [carla_platform.v1.json](carla_platform.v1.json) | `carla_platform` | verified |
| [local_analysis.v1.json](local_analysis.v1.json) | `local_analysis` | verified |

```bash
uv run python scripts/tools/bootstrap_recipe_check.py --check \
  --recipes configs/bootstrap_recipes --execute-safe-checks --require-verified --format markdown
```

Private host, account, path, module, and storage details never appear here. A recipe that needs them
declares a `private_substitutions` placeholder with a capability class;
[private_overlay.example.json](private_overlay.example.json) shows the overlay shape.

The checker is report-only by default; `--execute-safe-checks` runs only `probe` steps marked
`safe_check: true` inside an isolated temporary root and never mutates the host. Output is
deterministic (recipes sorted by `recipe_id`, JSON keys sorted). Blocking findings include
credential or private-path leaks, source-host access, stale absolute paths, unresolved placeholders,
mutable container tags without a digest, unpinned module or package aliases, destructive cleanup
targets, and shell-string steps that hide ordering. Recipes with `verification_status: unavailable`
are valid but must name an `unavailable_reason`; `--require-verified` fails when an execution class
has no verified recipe.
