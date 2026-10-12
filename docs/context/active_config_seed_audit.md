# Development config seed audit

Active training and selection configs use replacements in 1031–1060; active
development evaluation configs use replacements in 1101–1130. The replacements
preserve list order and size. They change experiment identity: historical results,
checkpoint selections and outcome descriptions do not transfer to a replacement
seed. No training or evaluation result is established by this maintenance change.

`configs/benchmarks/dev_seed_sets.yaml` and `dev_seed_list.yaml` are development
copies of the immutable legacy transports. Development tier names such as `dev_eval_s30` explicitly distinguish
them from protected paper schedules; they confer no paper or release authority. Active consumers now reference these copies. Frozen consumers
retain their original source bytes and transports.

`tests/validation/test_active_config_seeds.py` scans every non-exempt YAML config
for restricted seed literals, quoted lists, aliases, and embedded CLI arguments.
It also rejects references to retired transports. It distinguishes seed fields
from batch sizes and budgets and keeps held-out-band declarations and provenance
metadata intact. It checks the entire sealed numeric band, not just its sampled
release inventory. Diagnostics show file, line, field and `<seed>`.

`tests/fixtures/active_config_seed_exclusions.json` records exact paths, reasons
and original hashes for frozen/versioned releases, their transitive dependencies,
frozen preregistrations, and historical analysis/replay identity consumers. The
regression verifies these files retain their original bytes. New active files
are scanned automatically; a newly excluded input requires explicit review.

Run the pure configuration audit without stepping a planner:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/validation/test_active_config_seeds.py -q
```

The audit is static. Dynamic generation, environment overrides and runtime
training pool admission remain the responsibility of the separate seed-policy
lane. This change does not edit `scripts/training/train_ppo.py`.
