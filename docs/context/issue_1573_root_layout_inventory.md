# Issue #1573 Root-Layout Inventory - 2026-05-27

Issue: [#1573](https://github.com/ll7/robot_sf_ll7/issues/1573)

This note is the conservative foundation slice for #1573. It inventories the candidate first-level
paths, records the local evidence checked on this branch, and makes an explicit action call for
each path without performing a bulk directory move.

High-risk root path migration boundaries are now split to
[issue_1583_high_risk_root_boundaries.md](archive/issue_1583_high_risk_root_boundaries.md). That note keeps
`.agents/PLANS.md`, `model/pedestrian/`, `tests/pygame/`, and `CITATION.cff` at root, and delegates any future
`specs/` or `tests/fixtures/scenarios/` migration to dedicated follow-up issues.

Scope boundary for this PR:

- no first-level directory moves,
- no deletion of publication or contributor metadata,
- no movement of `specs/`, `model/pedestrian/`, `tests/pygame/`, `tests/fixtures/scenarios/`, `output/`,
  `experiments/`, `.pre-commit-config.yaml`, `.cursorrules`, or `CITATION.cff`,
- no claim based only on prior-agent output; the table below is based on local `git`/`rg` checks.

Action meanings:

- `keep`: current root location is intentional or already part of an explicit repo contract.
- `low-risk move candidate`: limited references; a future dedicated PR could relocate it with small
  path churn.
- `delete candidate`: looks like stale/generated root clutter and already conflicts with current
  repo policy.
- `moved`: relocated to a new path (e.g. `docs/tooling/` subtree); cross-reference the PR number.
- `deferred follow-up`: plausible cleanup target, but current references or workflow coupling make a
  move too risky for this foundation PR.

| Path | Local evidence checked | Action | Why this is the conservative call |
| --- | --- | --- | --- |
| `.agents/PLANS.md` | `AGENTS.md` points to `.agents/PLANS.md` as the repo plan-writing convention. | `keep` | This is already part of the agent-context contract, so moving it would ripple through agent instructions. |
| `class_diagram/` | No repo-wide references found outside the directory itself; contents are generated SVGs plus `class_diagram/generate_uml.sh`. | `moved` (#1579) | Relocated to `docs/tooling/class_diagram/`. |
| `experiments/` | `docs/dev_guide.md` and `docs/README.md` explicitly describe `experiments/registry.yaml` and `experiments/README.md` as the question-first experiment registry. | `keep` | This path is already documented as a canonical workflow surface. |
| `hooks/` | `.pre-commit-config.yaml` invokes `hooks/prevent_schema_duplicates.py`; tests also cover the hook contract/API surface under `tests/contract/` and related integration checks. | `deferred follow-up` | Cleanup is possible later, but this path is wired into pre-commit and test hook contracts, so any relocation still needs coordinated updates if the entrypoint or module path changes. |
| `model/pedestrian/` | Multiple scripts and tests resolve checkpoints from `model/pedestrian/`. | `keep` | Direct runtime and test references make this a high-blast-radius path; keep unchanged in the foundation PR. |
| `output/` | `AGENTS.md`, `docs/README.md`, `docs/dev_guide.md`, `.gitignore`, and coverage docs all treat `output/` as the canonical artifact root. | `keep` | Root-level artifact policy is intentional and already documented. |
| `specs/` | `docs/README.md`, `docs/dev_guide.md`, and multiple docs deep-link into `specs/...` quickstarts, plans, tasks, and contracts. | `deferred follow-up` | A future relocation would require broad doc-link migration; that is outside a conservative foundation PR. |
| `svg_conv/` | Only lightweight config/docs references were found; `svg_conv/README.md` itself says to prefer `robot_sf/nav/svg_map_parser.py`. | `moved` (#1579) | Relocated to `docs/tooling/svg_conv/`. |
| `tests/pygame/` | `docs/dev_guide.md`, `pyproject.toml`, tests, and helper scripts all reference `tests/pygame/` explicitly. | `keep` | This is a documented, active test surface with exact path references. |
| `tests/fixtures/scenarios/` | Tests load fixtures from `tests/fixtures/scenarios/osm_fixtures/...`; root fixtures are also part of current local test inputs. | `deferred follow-up` | Possible future consolidation into a test-fixture subtree, but not without updating multiple tests and docs. |
| `utilities/` | Refactoring docs and example commands explicitly reference `utilities/migrate_environments.py`; `pyproject.toml` also has per-path lint rules for `utilities/**/*.py`. No confirmed doc references to `utilities/n.py` were found in this check. | `deferred follow-up` | This is not a core runtime package, but current docs/tooling still point to a root `utilities/` path, so moving it needs a conservative dedicated cleanup PR. |
| `.coverage` | Root `.coverage` is a tracked SQLite database, while `pyproject.toml` and `docs/coverage_guide.md` now set the canonical coverage data file to `output/coverage/.coverage`. | `delete candidate` | This root artifact no longer matches the current coverage contract and is the clearest stale-root cleanup candidate. |
| `.cursorrules` | Root editor/agent metadata file; current content is a simple pointer back to `AGENTS.md`. | `keep` | User boundary forbids moving it here, and keeping editor metadata at repo root is conventional. |
| `.git-blame-ignore-revs` | Standard root-level Git metadata file with formatter/import-sort SHAs. | `keep` | Root placement matches normal Git tooling expectations. |
| `.pre-commit-config.yaml` | Canonical pre-commit entrypoint; directly references `hooks/prevent_schema_duplicates.py`. | `keep` | User boundary forbids moving it here, and repo tooling expects it at root. |
| `CITATION.cff` | `docs/RELEASE.md` and `docs/benchmark_release_reproducibility.md` explicitly require `CITATION.cff`. | `keep` | Publication/citation metadata must remain root-visible and untouched in this PR. |

## Validation Path Used For This Inventory

- `git status --short`
- `git ls-files --error-unmatch <path>` / `git check-ignore <path>` for candidate-path state
- `rg` lookups across `AGENTS.md`, `docs/`, `scripts/`, `tests/`, `pyproject.toml`, and
  `.pre-commit-config.yaml`
- direct file inspection for `class_diagram/generate_uml.sh`, `hooks/prevent_schema_duplicates.py`,
  `experiments/README.md`, `utilities/migrate_environments.py`, `docs/refactoring/README.md`,
  `svg_conv/README.md`, `tests/pygame/README.md`, `output/repos/README.md`, `.cursorrules`,
  `.git-blame-ignore-revs`, and `CITATION.cff`

## Follow-Up Boundary

If #1573 later needs actual path changes, the safest order is:

1. remove the stale root `.coverage` artifact first and add a `.gitignore` rule for root
   `.coverage` so it does not reappear,
2. decide whether `class_diagram/` and `svg_conv/` should move under docs/tooling: completed in
   PR #1579 (now at `docs/tooling/class_diagram/` and `docs/tooling/svg_conv/`),
3. handle any `hooks/`, `specs/`, `tests/fixtures/scenarios/`, or `utilities/` relocation only in dedicated
   PRs that update every exact-path reference together.
