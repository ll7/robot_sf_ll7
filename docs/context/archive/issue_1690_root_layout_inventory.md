# Issue #1690 Root Layout Inventory

Date: 2026-05-30

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1690>

Status: Superseded for the moved root paths by
[root_layout_structured_migration_2026-06-01.md](../root_layout_structured_migration_2026-06-01.md).
Keep this note as historical inventory evidence.

Predecessors:

- [Issue #1573 root-layout inventory](../issue_1573_root_layout_inventory.md)
- [Issue #1583 high-risk root path boundaries](issue_1583_high_risk_root_boundaries.md)
- [Issues #1598 and #1599 root compatibility decisions](issue_1598_1599_root_compatibility_decisions.md)

## Scope

This note refreshes the root-layout cleanup inventory without moving, deleting, or renaming files.
It is a preflight artifact for future structural cleanup, not a cleanup PR by itself.

The issue body referenced
`docs/context/open_issue_execution_improvement_plan_2026-05-30.md`; that file is not present on
`origin/main` at `8e845fd05d5b8cc68039a53ecabece05b7a96d74`. The inventory therefore uses the
current repository context stack and the earlier root-layout notes listed above.

## Recommendation Table

| Path | Current evidence | Recommendation | Follow-up shape |
| --- | --- | --- | --- |
| `.agents/PLANS.md` | `AGENTS.md`, `docs/ai/repo_overview.md`, and repo-local skills point to `.agents/PLANS.md`. | Keep. | No cleanup unless the agent planning contract is replaced everywhere. |
| `.agents/`, `.codex/`, `.opencode/`, `.gemini/` | Tracked compatibility mirrors and provider surfaces for currently supported agent workflows. | Keep. | Treat mirror changes as agent-workflow compatibility work, not root cleanup. |
| `.cursorrules` | Pointer file content is `Repository instructions are canonical in AGENTS.md.`; `scripts/tools/sync_ai_config.py` and tests verify that pointer. | Keep. | Keep root-visible for editor compatibility. |
| `.coverage` | No tracked root `.coverage` file is present; `.gitignore` ignores `/.coverage`; coverage config writes to `output/coverage/.coverage`. | Generated/ignored. | No deletion PR needed. If it reappears locally, discard it. |
| `.pre-commit-config.yaml` | Root pre-commit entrypoint invokes `hooks/prevent_schema_duplicates.py`. | Keep. | Moving it would be a tooling migration, not cleanup. |
| `.git-blame-ignore-revs` | Standard root Git metadata file. | Keep. | No follow-up. |
| `CITATION.cff` | Release docs, benchmark release configs, and release-protocol tests require this root path. | Keep. | Any change needs release-protocol validation. |
| `ACKNOWLEDGMENTS.md`, `CHANGELOG.md`, `LICENSE`, `README.md`, `AGENTS.md` | Standard repository metadata and AI entrypoints. | Keep. | No cleanup. |
| `opencode.json` | Root OpenCode configuration. | Keep. | Treat as provider tooling config. |
| `specs/` | Broad docs, schema tests, visual contract tests, examples, scripts, and runtime comments deep-link into `specs/...`. | Keep for now. | Defer any move to a dedicated compatibility migration; see the #1598/#1599 note. |
| `model/pedestrian/` | Pedestrian PPO scripts, collision benchmark, adversarial-pedestrian demo, examples manifest, changelog, and `tests/test_training_ped_ppo.py` reference `model/pedestrian/`. | Keep. | A future artifact migration would need checkpoint provenance and path-compatibility handling. |
| `tests/pygame/` | `AGENTS.md`, `docs/dev_guide.md`, `pyproject.toml`, tests, examples, and scripts reference this GUI test path. | Keep. | Only move with a dedicated GUI-test path migration and headless pygame validation. |
| `tests/fixtures/scenarios/` | OSM examples and tests reference `tests/fixtures/scenarios/osm_fixtures/sample_block.pbf`. | Keep for now. | Use the #1598/#1599 compatibility plan before any fixture move. |
| `hooks/` | `.pre-commit-config.yaml` invokes `hooks/prevent_schema_duplicates.py`; `pyproject.toml` has path-specific lint rules. | Defer. | A future tooling PR may relocate it only if pre-commit and tests are updated together. |
| `utilities/` | Refactoring docs and commands reference `utilities/migrate_environments.py`; `pyproject.toml` has path-specific lint rules. | Defer. | Possible docs/tooling relocation, but not without updating exact command references. |
| `experiments/` | `docs/dev_guide.md`, `docs/README.md`, and registry validation tests describe `experiments/registry.yaml` as the question-first experiment registry. | Keep. | Do not collapse into `configs/` unless the registry workflow is replaced. |
| `output/` | `AGENTS.md`, `.gitignore`, docs, workflows, and coverage config define `output/` as the canonical ignored artifact root; `output/repos/README.md` is the only tracked root artifact inside it. | Keep artifact root; split follow-up for `output/repos/README.md` only if maintainers want a cleaner ignored tree. | Any change must preserve artifact-root policy and avoid committing raw generated outputs. |
| `SLURM/` | Tracked SLURM workflow surface; many issues use SLURM/Auxme lanes. | Keep. | Treat as workflow documentation/infrastructure, not incidental clutter. |
| `contracts/` | Small tracked contract surface. A worker suggested it may be a low-risk relocation candidate, but this issue did not inspect contract consumers deeply enough to approve a move. | Defer. | Only move under a dedicated contract-location issue with reference and schema checks. |
| `.specify/` | Repo-local specification scaffolding and constitution memory. | Defer. | Treat as agent/specify workflow compatibility, not incidental clutter. |
| `memory/` | Stable repo-local memory layer referenced by `AGENTS.md` and the context stack. | Keep. | Do not move unless the memory contract is replaced across supported agent entrypoints. |
| `configs/`, `docker/`, `docs/`, `examples/`, `fast-pysf/`, `maps/`, `model/`, `robot_sf/`, `robot_sf_carla_bridge/`, `scripts/`, `tests/`, `third_party/` | Canonical source, docs, fixture, dependency, or workflow roots. | Keep. | Not cleanup candidates in this issue. |

## Current Deltas From Earlier Notes

- Root `.coverage` is no longer a tracked deletion candidate on this branch; it is absent and
  explicitly ignored.
- `specs/` and `tests/fixtures/scenarios/` have since been resolved by the #1598/#1599 compatibility note:
  keep both at root unless a dedicated migration preserves compatibility.
- `output/repos/README.md` is the only tracked file under the otherwise ignored `output/` root.
  This does not block current workflows, but it is the cleanest small follow-up candidate if the
  project wants `output/` to be purely ignored.
- The issue-referenced `open_issue_execution_improvement_plan_2026-05-30.md` was not present on the
  checked base, so it should not be treated as a dependency for future cleanup.

## Evidence

Local commands used:

```bash
git ls-files | awk -F/ 'NF>1{print $1}' | sort -u
git ls-files | awk 'index($0,"/")==0{print $0}' | sort
git status --ignored --short -uall
git ls-files .coverage
git check-ignore -v .coverage output/coverage/.coverage
rg -nF ".agents/PLANS.md" AGENTS.md docs .github .vscode .agents
rg -nF "model/pedestrian/" AGENTS.md docs tests scripts examples robot_sf
rg -nF "tests/pygame/" AGENTS.md docs tests scripts examples pyproject.toml
rg -nF "tests/fixtures/scenarios/" AGENTS.md docs tests scripts examples robot_sf pyproject.toml
rg -nF "CITATION.cff" AGENTS.md docs tests configs .github
rg -nF "hooks/" .pre-commit-config.yaml pyproject.toml AGENTS.md docs tests scripts .github
rg -nF "utilities/" AGENTS.md docs tests scripts pyproject.toml .github
rg -nF "experiments/" AGENTS.md docs tests scripts pyproject.toml .github README.md
rg -nF "output/" AGENTS.md docs tests scripts pyproject.toml .github README.md
```

Delegated read-only checks:

- Gemini `auto` completed but used semantic tools instead of shell checks and recommended moves
  contradicted local exact-path evidence; its output was treated as low confidence.
- OpenCode Go `opencode-go/deepseek-v4-pro` completed a token-bounded review and agreed on the
  main keep/defer shape, while flagging `output/repos/README.md` as the only tracked file inside
  the ignored artifact root.
- Qwen `Qwen3.6-27B` completed a scout and confirmed the missing plan doc plus the main
  high-risk keeps. It also proposed `contracts/`, `.specify/`, and `memory/` as possible
  relocation candidates; this note classifies those as defer/keep because their workflow contracts
  need a dedicated compatibility check before any move.
- Qwen `qwen3-coder-plus` timed out with no edits or captured stdout.

## Validation

This is a docs-only inventory. Validation should include:

```bash
git diff --check origin/main...HEAD
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```
