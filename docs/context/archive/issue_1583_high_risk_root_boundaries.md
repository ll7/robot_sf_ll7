# Issue #1583 High-Risk Root Path Boundaries - 2026-05-28

Date: 2026-05-28

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1583>

Status: Superseded for root paths moved by
[root_layout_structured_migration_2026-06-01.md](../root_layout_structured_migration_2026-06-01.md).
Keep this note as provenance for the earlier conservative decision.

## Decision

No high-risk root paths should move in the current root-layout cleanup stream. Four paths are
intentional root contracts and should stay where they are. Two paths may be revisited, but only in
dedicated follow-up issues with compatibility plans and targeted validation.

| Path | Active references | Decision | Validation gate | Compatibility burden | Rollback plan |
| --- | --- | --- | --- | --- | --- |
| `.agents/PLANS.md` | `AGENTS.md`, `docs/ai/repo_overview.md`, repo-local skills, and compatibility notes point to `.agents/PLANS.md`. | Keep. | `rg -nF ".agents/PLANS.md" AGENTS.md docs .github .vscode .agents` | Low, but it is part of the agent workflow contract. | Restore `.agents/PLANS.md` and revert docs/instruction links. |
| `specs/` | Docs, examples, schemas, visual contract tests, helper docs, and runtime comments deep-link into `specs/...`. | Defer to Issue #1599. | `rg -nF "specs/" AGENTS.md docs tests scripts robot_sf examples .github .vscode` plus targeted schema/docs tests from the follow-up inventory. | Very high: broad docs/schema/test path churn and potential broken contract links. | Keep root path or provide compatibility links until every consumer is migrated. |
| `model/pedestrian/` | Pedestrian PPO training scripts, collision benchmark script, adversarial pedestrian demo, examples manifest, and `tests/test_training_ped_ppo.py` encode checkpoint paths. | Keep. | `rg -nF "model/pedestrian/" AGENTS.md docs tests scripts examples robot_sf`; if behavior changes, `uv run pytest tests/test_training_ped_ppo.py -q`. | High: checkpoint paths are user-facing and test-visible. | Restore root `model/pedestrian/` and checkpoint path defaults. |
| `tests/pygame/` | `AGENTS.md`, `docs/dev_guide.md`, `pyproject.toml`, demo scripts, and JSONL/recording tests reference GUI tests and recordings there. | Keep. | `rg -nF "tests/pygame/" AGENTS.md docs tests scripts examples pyproject.toml`; for GUI changes, run the headless pygame test command from `AGENTS.md`. | Medium: active test surface and fixture path assumptions. | Restore root `tests/pygame/` and revert config/test path edits. |
| `tests/fixtures/scenarios/` | OSM examples and tests reference `tests/fixtures/scenarios/osm_fixtures/sample_block.pbf` directly. | Defer to Issue #1598. | `rg -nF "tests/fixtures/scenarios/" tests examples docs scripts robot_sf`; targeted OSM tests from Issue #1598. | Medium-high: fixture-relative paths and example defaults need lockstep migration. | Keep root fixtures or provide compatibility path until examples/tests are migrated. |
| `CITATION.cff` | Release docs, release manifests, and release-protocol tests require a root citation file. | Keep. | `rg -nF "CITATION.cff" AGENTS.md docs tests configs .github`; for release changes, `uv run pytest tests/benchmark/test_release_protocol.py -q`. | High: publication metadata and release manifests expect the root path. | Restore root `CITATION.cff` and manifest `citation_path` values. |

## Follow-Up Split

- Issue #1599 owns the `specs/` compatibility plan and is resolved by
  [issue_1598_1599_root_compatibility_decisions.md](issue_1598_1599_root_compatibility_decisions.md).
- Issue #1598 owns the `tests/fixtures/scenarios/` fixture relocation boundary and is resolved by
  [issue_1598_1599_root_compatibility_decisions.md](issue_1598_1599_root_compatibility_decisions.md).
- No follow-up issue is needed for `.agents/PLANS.md`, `model/pedestrian/`, `tests/pygame/`, or `CITATION.cff`
  unless a maintainer later requests a migration despite the current keep decision.

## Evidence Checked

The decisions above combine the read-only Spark sidecar inventory with local `rg` checks run on this
branch. The sidecar did not edit files; final decisions and GitHub writes were handled locally.

Reference commands:

```bash
rg -nF ".agents/PLANS.md" AGENTS.md docs .github .vscode .agents --glob '!docs/context/evidence/**' --glob '!output/**'
rg -nF "specs/" AGENTS.md docs tests scripts robot_sf examples .github .vscode --glob '!docs/context/evidence/**' --glob '!output/**'
rg -nF "model/pedestrian/" AGENTS.md docs tests scripts examples robot_sf --glob '!docs/context/evidence/**' --glob '!output/**'
rg -nF "tests/pygame/" AGENTS.md docs tests scripts examples pyproject.toml --glob '!docs/context/evidence/**' --glob '!output/**'
rg -nF "tests/fixtures/scenarios/" AGENTS.md docs tests scripts examples robot_sf --glob '!docs/context/evidence/**' --glob '!output/**'
rg -nF "CITATION.cff" AGENTS.md docs tests configs .github --glob '!docs/context/evidence/**' --glob '!output/**'
```

No files were moved and no generated artifacts were promoted.
