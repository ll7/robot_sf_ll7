# Issue #2035 Path-Reference Audit (2026-06-01)

Date: 2026-06-01

Related issue:

- <https://github.com/ll7/robot_sf_ll7/issues/2035>

Related context:

- [root_layout_structured_migration_2026-06-01.md](root_layout_structured_migration_2026-06-01.md)
- [issue_1690_root_layout_inventory.md](archive/issue_1690_root_layout_inventory.md)

## Scope

This audit checked active instruction, workflow, prompt, command, docs, and script surfaces for
stale references left by the root-layout restructuring:

- `.agent/PLANS.md`
- root-level `contracts/`
- `model_ped/`
- `test_pygame/`
- `test_scenarios/`

## Changes Made

- Clarified Speckit Codex/GitHub/Gemini prompt and command text so generated API contracts are
  described as feature-local `FEATURE_DIR/contracts/` or `$SPECS_DIR/contracts/`, not the removed
  repository-root `contracts/` directory.
- Clarified `configs/scenarios/README.md` so scenario-intent contracts point at
  `configs/scenarios/contracts/`.

## Intentional Residual Matches

- `docs/context/root_layout_structured_migration_2026-06-01.md` keeps the old path names in its
  migration mapping table as historical evidence.
- `docs/context/policy_search/contracts/*`, `configs/scenarios/contracts/*`,
  `configs/benchmarks/odd_contracts/*`, and `specs/*/contracts/*` are scoped contract directories,
  not the removed root-level `contracts/` directory.
- `.specify` templates and scripts still use `contracts/` for feature-local generated Speckit
  artifacts under the current feature directory. That is an active workflow convention, not a
  root-layout stale path.

## Validation

Commands run from the issue #2035 worktree:

```bash
rg -n --hidden --glob '!/.git/**' --glob '!output/**' --glob '!results/**' \
  --glob '!.venv/**' --glob '!.understand-anything/**' --glob '!specs/**' \
  '\.agent/PLANS\.md|model_ped/|test_pygame/|test_scenarios/' \
  AGENTS.md docs .agents .github scripts README.md pyproject.toml .specify memory configs

rg -n --fixed-strings "contracts/" \
  .agents/commands .agents/prompts .agents/agents configs/scenarios/README.md

scripts/dev/run_worktree_shared_venv.sh -- \
  python scripts/validation/check_docs_proof_consistency.py \
    --path configs/scenarios/README.md

git diff --check
```

The first search returns only the historical migration mapping note plus this audit note for the
non-contract moved paths. The second search returns only explicitly scoped
`FEATURE_DIR/contracts/`, `$SPECS_DIR/contracts/`, and `configs/scenarios/contracts/` references in
the touched active surfaces.
