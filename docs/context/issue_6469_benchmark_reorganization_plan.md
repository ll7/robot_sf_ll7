# Issue #6469 Benchmark Module Reorganization Plan

Status: proposal (planning only; not benchmark evidence)

This note records the bounded repository-hygiene outcome for
[Issue #6469](https://github.com/ll7/robot_sf_ll7/issues/6469). It confirms that the
ghost utility directories carry no tracked files, and it proposes a domain-subdirectory
reorganization for the flat `robot_sf/benchmark/` namespace. Per the Issue #6469 domain-aware
approval boundary, this note does **not** move or rename any benchmark module and does **not**
alter any import. It produces only the reorganization plan and the follow-up contract.

## Ghost Directory Confirmation

`robot_sf/util/` and `robot_sf/utils/` are untracked husks that only ever contained stale
`__pycache__/` bytecode from previously deleted modules. Confirmation on this branch:

- `git ls-files robot_sf/util robot_sf/utils` returns no tracked files.
- `git status --ignored robot_sf/util robot_sf/utils` reports a clean tree; neither directory is
  present in a fresh checkout because `__pycache__/` is ignored and never committed.

Because no tracked file lives under either path, there is nothing to `git rm` in this issue. A
fresh clone already satisfies the "removed" acceptance criterion; the directories reappear only as
ignored bytecode caches when old import paths are exercised, and they disappear on any clean
checkout. The follow-up contract below includes a guard so the husks are not reintroduced.

## Current Benchmark State

`robot_sf/benchmark/` has grown to **279 top-level non-`__init__.py` modules** (**280 top-level
`.py` files including `__init__.py`**) plus existing subdirectories.
The flat namespace mixes domain modules, issue-specific modules, cross-cutting preflight/readiness
helpers, and schema/data directories. Existing subdirectories show that domain grouping is viable:

| Existing subdirectory | Top-level `.py` count | Notes |
| --- | --- | --- |
| `camera_ready/` | 14 | release campaign surface |
| `full_classic/` | 18 | classic benchmark variant |
| `map_runner_policies/` | 12 | policy pack for the map runner |
| `scenario_generation/` | 14 | scenario synthesis |
| `snqi/` | 12 | Social Navigation Quality Index surface |
| `figures/` | 6 | figure export |
| `schemas/` | 4 | versioned JSON schema package (has `__init__.py`) |
| `identity/` | 2 | identity/equivalence checks |
| `metrics/` | 2 | metric package |
| `validation/` | 1 | validation helpers |
| `schema/` | 0 | holds only `scenarios.schema.json` (data, not a package) |

Two structural smells are already visible and should be resolved by the follow-up:

- `schema/` (a single JSON data file) and `schemas/` (the versioned schema package) are a naming
  collision. The follow-up should decide whether `schema/scenarios.schema.json` moves under
  `schemas/` or a dedicated data directory.
- `map_runner_policies/` exists as a subdirectory while 22 `map_runner*` modules remain top-level,
  splitting one domain across two locations.

## Proposed Domain Subdirectory Groupings

The grouping below is illustrative and conservative. Counts are filename-prefix counts over the
279 top-level non-`__init__.py` modules; some modules are cross-cutting and final assignment is a
follow-up decision, not a claim made here. The issue-specific modules named in Issue #6469 (for
example
`issue_5302_oracle_gap.py`, `issue_4142_dpcbf_dense_runner.py`) are assigned to the domain they
serve rather than to an `issue_*` directory, so the namespace stays domain-oriented.

| Proposed subdirectory | Candidate members (prefix) | Approx. count | Example modules |
| --- | --- | --- | --- |
| `benchmark/adversarial/` | `adversarial_*` | 4 | `adversarial_package_b_report.py` |
| `benchmark/forecast/` | `forecast_*` | 11 | `forecast_baseline_comparison.py`, `forecast_metrics.py` |
| `benchmark/predictive/` | `predictive_*` | 4 | `predictive_v2_comparison_readiness.py` |
| `benchmark/collision/` | `collision_*` | 5 | `collision_causal_report.py`, `collision_cause_analyser.py` |
| `benchmark/safety/` | `safety_*`, `cbf_safety_*` | 7 | `safety_wrapper_runtime.py`, `cbf_safety_filter_runtime.py` |
| `benchmark/latency/` | `*latency*` | 5 | `control_action_latency_snqi.py`, `latency_stress.py` |
| `benchmark/map_runner/` | `map_runner*` | 22 | `map_runner.py`, `map_runner_batch_runner.py` (merge with `map_runner_policies/`) |
| `benchmark/scenario/` | `scenario_*` | 16 | `scenario_contract.py`, `scenario_coverage.py` (relate to `scenario_generation/`) |
| `benchmark/campaign/` | `campaign_*`, `camera_ready*` | 7 | `campaign_runtime_preflight.py` (relate to `camera_ready/`) |
| `benchmark/constraint/` | `issue_4142_dpcbf_*` | 3 | `issue_4142_dpcbf_dense_runner.py` |

Modules that do not match a domain prefix (shared runners, preflight/readiness helpers, policy
builders, schema loaders, metric layers) stay top-level or move into the existing `metrics/`,
`schemas/`, or `validation/` packages during the follow-up. The goal is to shrink the flat
top-level set substantially, not to force every file into a domain.

## Migration Contract (for the follow-up, not this issue)

The follow-up must treat the move as an import-contract change, because 158 top-level
non-`__init__.py` benchmark modules import from `robot_sf.benchmark`. Required discipline:

1. Move files with `git mv` so history is preserved; never delete-and-recreate.
2. Update every `robot_sf.benchmark.<module>` import across `robot_sf/`, `scripts/`, `tests/`, and
   `examples/` in the same change that moves the module.
3. Prefer direct moves. If a module is part of a public or cross-package surface, keep a temporary
   re-export shim in the old location for one release and mark it deprecated, then remove it in a
   named successor issue.
4. Resolve the `schema/` vs `schemas/` collision and the `map_runner` split as explicit, recorded
   decisions in the follow-up note.
5. Keep each domain move independently reviewable: one domain (or one tightly coupled domain
   cluster) per PR, not one 279-file rewrite.

### Proof plan for each move PR

A move PR should be accepted only after:

- exact-path searches show the moved modules no longer exist at the old top-level location;
- `uv run ruff check .` and `uv run ruff format --check .` pass;
- `uv run pytest tests` passes (no import breakage);
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` passes;
- a grep for the old import path returns only intentional deprecation shims.

## Follow-Up Issue Contract

Issue #6469 acceptance is satisfied by this plan. This PR does not create the follow-up issue;
the contract below gives the coordinator an actionable shape for that next step:

- Title: `benchmark: reorganize flat module namespace into domain subdirectories`.
- Parent/umbrella issue with one child issue per domain cluster in the table above, each child
  independently movable and validatable under the proof plan.
- First child should be the lowest-risk, highest-signal cluster (for example `adversarial/`, four
  modules) to prove the migration contract before larger clusters such as `map_runner/` (22) or
  `scenario/` (16).
- Include a guard task that adds a test or lint rule preventing `robot_sf/util/` and
  `robot_sf/utils/` from being reintroduced as tracked paths.
- Label the follow-up `technical-debt`; it is structural cleanup, not benchmark evidence, and must
  not promote fallback or degraded execution as success.

## Boundaries And Caveats

- This note is a proposal. It establishes no benchmark, metric, schema, or paper-facing claim.
- No benchmark module is moved, renamed, or imported differently by this issue.
- Domain assignments and counts are filename-derived estimates; the follow-up owns final placement.
- Ghost-directory removal is already satisfied on any clean checkout because the directories hold
  only ignored `__pycache__/` bytecode and no tracked files.

## Links

- Motivating issue: [Issue #6469](https://github.com/ll7/robot_sf_ll7/issues/6469) (`repo-hygiene:
  remove ghost util/utils dirs and plan benchmark module reorganization`).
- Prior structured-migration precedent and proof-plan style:
  [root_layout_structured_migration_2026-06-01.md](root_layout_structured_migration_2026-06-01.md).
- Repository execution rules: [AGENTS.md](../../AGENTS.md).
