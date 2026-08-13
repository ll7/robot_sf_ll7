# Chapter 7 case-dossier fixtures

Artifact classification: `tracked-compact-evidence`.

These packages are synthetic, source-bound renderer proofs. They preserve the
planner/scenario/seed shapes of the first requested templates, but every input
sets `mode=synthetic_fixture` and `scientific_admission=false`. They are not
release evidence and must not be cited as scientific results.

- `matched_seed118/` covers the matched-start `goal` versus `ppo` grammar.
- `doorway_seeds113_114/` covers the same-cell, no-shared-prefix `ppo` grammar.
- Each directory has one valid `input.json` and one known-bad input. The bad
  inputs pin the source-hash and forbidden-difference-curve stop conditions.

Generated SVG, PDF, caption, dossier manifest, sidecar, and artifact catalog
files are disposable validation outputs. They are intentionally not committed
or cited. Render a fixture without running a simulation:

```bash
uv run python scripts/analysis/render_case_dossier.py \
  --check-determinism \
  --input tests/fixtures/benchmark/case_dossier_v1/matched_seed118/input.json \
  --out-dir /tmp/case-dossier-matched
```

After a portfolio or provenance-contract change, refresh the tracked compact packages with the
canonical test builder and then run its no-simulation integrity check:

```bash
scripts/dev/run_worktree_shared_venv.sh -- python scripts/dev/refresh_case_dossier_fixtures.py --write
scripts/dev/run_worktree_shared_venv.sh -- python scripts/dev/refresh_case_dossier_fixtures.py
```

Use the non-mutating drift gate in validation or continuous integration to regenerate both
packages under a temporary directory and compare every JSON file with the committed tree:

```bash
scripts/dev/run_worktree_shared_venv.sh -- python scripts/dev/refresh_case_dossier_fixtures.py --check
```

The check reports the first differing path and the source-bound digests, then exits non-zero
without changing tracked fixtures.
