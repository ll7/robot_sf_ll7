# Issue #1550 Predictive Same-Seed Row Summary Schema

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1550>

## Purpose

Define a small tracked schema for future predictive same-seed comparison outcome rows so bounded,
durable row summaries can live in git without depending on local `output/` artifacts. This schema is
for per-row comparison bookkeeping, not for benchmark synthesis or paper-facing claims.

## Scope Boundary

- **In scope**: a machine-readable row contract, validator, checked-in template, and docs guidance
  for future predictive same-seed campaigns.
- **Out of scope**: reconstructing or inventing missing historical #1427 per-row results, wiring new
  campaign emitters, or upgrading these rows into benchmark-strength evidence.

Issue #1427 already has high-level same-seed conclusions in
`docs/context/issue_1167_predictive_obstacle_pipeline.md`. If a historical row was not preserved in
a durable artifact at the time, this schema must **not** be used to backfill or infer that missing
row later.

## Row Contract

Each row represents one variant/scenario/seed/planner-grid outcome within a same-seed comparison
campaign. Store rows as YAML or JSON mappings and validate them with the dedicated CLI before
committing them. The tuple `(variant, scenario, seed, planner_grid_key)` is the canonical row
identity and must be unique within a summary file.

### Required fields

| Field | Type | Meaning |
| --- | --- | --- |
| `source_issue` | string | Issue number that owns the row summary, e.g. `#1550`. |
| `campaign` | string | Stable campaign label or run grouping identifier. |
| `variant` | string | Variant/checkpoint label for the row. |
| `scenario` | string | Scenario identifier used for the evaluation row. |
| `seed` | integer | Exact deterministic seed for the row. |
| `planner_grid_row` | string | Human-readable planner-grid row name. |
| `planner_grid_key` | string | Stable planner-grid key when present; duplicate `planner_grid_row` if no separate key exists. |
| `status` | enum | One of `ok`, `failed`, `degraded`, `unavailable`, `unknown`. |
| `success` | boolean or null | Episode success outcome. Must be null for `unavailable`/`unknown`. |
| `collision_event` | boolean or null | Whether a collision-like terminal event was observed. |
| `near_miss` | boolean or null | Whether a near-miss event was observed or flagged. |
| `low_progress` | boolean or null | Whether the row ended in low-progress failure. |
| `timeout` | boolean or null | Whether the row ended by timeout. |
| `min_distance` | number or null | Minimum observed clearance for the row. |
| `artifact_pointer` | string | One durable pointer for the row summary, such as a tracked note/table path or durable remote artifact URI. |
| `commit_artifact` | string | Git SHA plus one or more provenance tokens for the row source, using the same comma/newline token style as the hybrid-evidence matrix validator. |

### Optional fields

| Field | Type | Meaning |
| --- | --- | --- |
| `comparison_group` | string | Stable label tying baseline and variant rows together. |
| `scenario_matrix` | string | Repository-relative path to the scenario matrix used for the row. |
| `seed_manifest` | string | Repository-relative path to the committed seed manifest. |
| `source_note` | string | Repository-relative path to the note or evidence summary that interprets the row. |

## Status Semantics

| `status` | Expected outcome fields |
| --- | --- |
| `ok` | All booleans plus `min_distance` must be populated. |
| `failed` | Known outcome fields may be populated; unknown values may remain null. |
| `degraded` | Partial outcome information may be recorded, but the row remains an explicit caveat. |
| `unavailable` | All outcome fields must be null. Use this when the planner/dependency/source artifact could not satisfy the row contract. |
| `unknown` | All outcome fields must be null. Use this when the row should exist but no durable outcome was preserved. |

These rows are **diagnostic-only**. They are not substitutes for the benchmark-facing evidence matrix
from `docs/context/issue_1499_hybrid_evidence_matrix_schema.md`, and they must not be treated as
synthesis-grade evidence for issue #1489.

## Tracked Template And Validator

- Template:
  `docs/context/evidence/predictive_same_seed_row_summary_template.yaml`
- Python validator:
  `robot_sf/benchmark/predictive_same_seed_row_summary.py`
- CLI:
  `scripts/validation/validate_predictive_same_seed_row_summary.py`

Default validation is format-only so future row drafts can be checked before the final campaign SHA
is available locally. For stricter provenance checks, rerun the same file with `--check-git-history`
from a checkout whose git history contains the cited commit.

Canonical command:

```bash
uv run python scripts/validation/validate_predictive_same_seed_row_summary.py \
  --input docs/context/evidence/predictive_same_seed_row_summary_template.yaml
```

## Usage Guidance

Use this schema when a predictive same-seed campaign needs a small tracked table of row outcomes that
can survive worktree cleanup. Typical cases:

1. preserving a planner-grid row outcome that would otherwise only exist inside ignored JSONL output,
2. carrying one unavailable/unknown row forward without inventing metrics,
3. linking a future predictive comparison note to a compact tracked YAML summary.

Do **not** use this schema to restate old #1427 rows from memory, from screenshots, or from deleted
local output trees. If a durable source was not preserved, keep the row as `unknown` or leave it out
until a real source exists.
