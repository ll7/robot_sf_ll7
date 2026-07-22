# Issue #6054: Regulation-to-Scenario Compilation Prototype

## Overview

This note documents a deterministic prototype that compiles a textual
regulation/requirement excerpt into a parameterized `robot_sf_ll7` scenario
config. The prototype is a **compiler**, not a benchmark: it reports
**compilation** and **schema** validity on separate axes and explicitly does
**not** assess **scenario** validity. Every output is a **hypothesis** until the
scenario is executed through the benchmark runner and reviewed.

Status: exploratory prototype. No evidence claim is made or implied.

## Goals And Non-Goals

Goals:

- Turn a textual regulation/requirement excerpt into a `robot_sf_ll7` scenario
  config deterministically (no LLM, no network).
- Report three distinct validity signals so callers cannot conflate "the
  compiler ran" with "the schema is valid" with "the scenario is valid":
  1. **Compilation validity** — did the compiler extract parameters, and what
     could it not interpret?
  2. **Schema validity** — does the generated config satisfy
     `robot_sf.scenario_matrix.v1` via `robot_sf.benchmark.scenario_schema`
     (reused, not rewritten)?
  3. **Scenario validity** — `not_assessed`. Only execution + review establishes
     this.
- Mark outputs as hypotheses (`required_manual_review: true`,
  `benchmark_evidence: false`, explicit claim boundary).

Non-goals:

- No LLM generation, inference, or fuzzy interpretation. Ambiguous clauses are
  recorded as unmatched, not guessed.
- No map reconstruction. The selected SVG is a structural approximation of the
  regulation's setting, not a replica (same convention as the failure-record
  converter, `docs/context/issue_4760_failure_record_to_scenario.md`).
- No runtime/clearance enforcement. A clearance requirement is recorded as
  metadata only; the simulator does not enforce it from this field.
- No claim that the compiled scenario is realistic, runnable as-is, or suitable
  for planner ranking.

## Why Three Validity Axes

A regulation excerpt and a benchmark scenario live at different epistemic
levels. Collapsing them hides failure modes:

- A compiler can succeed (compilation valid) while emitting a schema-invalid
  config (e.g. a bad enum), or vice-versa.
- A schema-valid config can still reference POIs that do not exist in the map,
  or encode parameters the runner ignores — that is a scenario-validity gap the
  schema cannot catch.
- Only executing the scenario through the benchmark runner and reviewing the
  results can establish scenario validity. Reporting it as `not_assessed` keeps
  reviewers from treating a clean compile as evidence.

This separation follows the repository's claim-strength discipline:
compilation and schema checks are cheap structural proofs; scenario validity is
a benchmark/evidence-grade claim that this prototype deliberately does not make.

## Input Schema (`regulation-record.v1`)

The input is a YAML envelope wrapping the textual excerpt plus minimal
structured hints. The excerpt is authoritative; hints only broaden keyword
matching (e.g. `context: shared space`).

```yaml
schema_version: regulation-record.v1
regulation:
  id: "<unique-identifier>"
  source: "<source-description>"                 # optional
  issuing_body: "<body>"                          # optional
  date: "<ISO-8601-date>"                         # optional
  context: "<zone-hint>"                          # optional, broadens zone match
  setting: "<zone-hint>"                          # optional, broadens zone match
  regulation_text: |
    <the textual regulation/requirement excerpt>
  required_manual_review: true                    # must be true
  claim_boundary: "...not evidence"               # must state 'not evidence'
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Must be `regulation-record.v1` |
| `regulation.id` | string | Unique identifier for the regulation excerpt |
| `regulation.regulation_text` | string | The textual excerpt to compile (non-empty) |
| `regulation.required_manual_review` | bool | Always `true` for compiled hypotheses |
| `regulation.claim_boundary` | string | Must state `not evidence` |

### Optional Hint Fields

`source`, `issuing_body`, `date`, `context`, `setting` are preserved as
provenance metadata only and never override an explicit extraction. A
`max_episode_steps` hint (positive int), if present, sets the episode horizon.

See `configs/scenarios/contracts/issue_6054_regulation_source_example.yaml`
for a worked example.

## Compilation Rules

The compiler is deterministic and auditable. Every extracted parameter is
recorded in `metadata.compiled_parameters`; every clause it could not interpret
is recorded in `metadata.unmatched_clauses`.

### Zone / Template Mapping

The excerpt and `context`/`setting` hints are scanned (case-insensitive) for
zone keywords. Longer/more-specific keys win (e.g. `shared space` beats a
generic match). The template selects an approximate SVG map.

| Keyword(s) in excerpt/hints | Template | Map (approximation) |
|-----------------------------|----------|---------------------|
| `shared space`, `plaza` | `shared_space` | `classic_merging.svg` |
| `pedestrian zone` | `pedestrian_zone` | `classic_urban_crossing.svg` |
| `crossing`, `intersection` | `crossing` | `classic_urban_crossing.svg` |
| `corridor`, `hallway`, `narrow passage` | `corridor` | `classic_head_on_corridor.svg` |
| `station`, `platform` | `station_platform` | `classic_station_platform.svg` |
| `sidewalk`, `pavement` | `sidewalk` | `classic_crossing.svg` |
| `room`, `elevator` | `room` | `francis2023/francis2023_entering_room.svg` |
| (none) | `shared_space` (default) | `classic_merging.svg` |

### Parameter Extraction

- **Max linear speed (m/s):** matched after phrasings like `max speed`,
  `maximum speed`, `speed limit`, `shall not exceed`, `must not exceed`,
  `no faster than`, `at most`, `not exceed`; or a number immediately preceding
  a unit (`m/s`, `mps`, `meters per second`). Feeds `robot_config.max_linear_speed`.
  If multiple distinct speeds appear, the first is used and a warning is emitted.
- **Clearance / separation (m):** matched after `clearance`, `distance`,
  `separation`, `keep at least`, `maintain at least`, `minimum distance`,
  `buffer`, `gap of`. Recorded as metadata (`clearance_requirement_m`) only; the
  simulator does not enforce it from this field (`clearance_enforcement:
  metadata_only_not_runtime_enforced`).
- **Pedestrian density (peds/m^2):** an explicit numeric density (0.0–1.0) wins;
  otherwise density wording (`low`/`sparse`→0.02, `medium`/`moderate`→0.05,
  `high`/`dense`→0.08, `crowded`/`very high`→0.12); otherwise the default 0.05.
  Feeds `simulation_config.ped_density`.

Numeric extraction tolerates up to 20 non-digit characters between a keyword and
its value (e.g. `clearance of at least 1.0 m`). European decimals (`1,5`) are
accepted.

### Unmatched Clauses

Clauses that contain no speed/clearance/density keyword are recorded verbatim in
`metadata.unmatched_clauses`. Examples the compiler intentionally does **not**
guess: "the robot shall yield at a marked crossing", "the robot shall provide an
audible alert before moving". A reviewer must decide how (or whether) to encode
these; the prototype refuses to invent parameters for them.

## Output Schema

The output is a `robot_sf.scenario_matrix.v1` scenario matrix (a single-scenario
list) with hypothesis metadata. It validates against the shared JSON Schema via
`robot_sf.benchmark.scenario_schema.validate_scenario_list`.

Key metadata fields:

| Field | Meaning |
|-------|---------|
| `generation_method` | `deterministic_regulation_compile_v1` |
| `hypothesis` | Always `true` |
| `required_manual_review` | Always `true` |
| `claim_boundary` | States this is not executed evidence |
| `compiled_parameters` | Audit trail of every extracted parameter |
| `unmatched_clauses` | Clauses the compiler could not interpret |
| `compilation_warnings` | Soft warnings (e.g. no speed extracted) |
| `clearance_requirement_m` | Recorded clearance (metadata only) |
| `authoring.status` / `authoring.benchmark_evidence` | `draft` / `false` |
| `plausibility.status` | `unverified` |

## Validity Reporting

The tool and library return a three-axis validity report:

```yaml
compilation_validity:
  valid: true               # false only if nothing was extracted
  extracted_count: 3
  unmatched_clause_count: 2
  warnings: [...]
  errors: []
schema_validity:
  valid: true               # from validate_scenario_matrix_metadata + validate_scenario_list
  metadata_errors: []
  item_errors: []
scenario_validity:
  status: not_assessed      # always not_assessed; requires runner + review
  reason: "..."
```

CLI exit codes keep the axes separable for automation:

| Exit | Meaning |
|------|---------|
| 0 | Compiled and schema-valid (still a hypothesis) |
| 2 | Input/usage error (missing/invalid regulation record) |
| 3 | Compiled but the generated config is schema-invalid |

## Limitations

1. **Deterministic only.** No LLM, no fuzzy matching. Ambiguous or unmapped
   clauses are recorded, not guessed.
2. **Approximate maps.** Each template maps to an existing SVG that approximates
   the regulation's setting. `map_file` is emitted relative to the output YAML
   (`--output-yaml`) or repo-relative (`--stdout`), matching the failure-record
   converter convention.
3. **No executability proof.** The compiled scenario may reference a map whose
   POIs/route geometry do not match the implied parameters. Schema validity does
   not imply the scenario runs.
4. **Clearance is metadata only.** Nothing in the simulator enforces a clearance
   constraint from this field; it is recorded for audit/review.
5. **Single density regime.** Only one `ped_density` is emitted per scenario;
   regulations specifying per-zone densities are out of scope for v1.
6. **No evidence claims.** Outputs are hypotheses until executed and reviewed.

## Human Review Requirement

Before treating a compiled scenario as anything more than a draft:

1. Read `unmatched_clauses` and `compilation_warnings`; decide how to encode
   anything dropped.
2. Verify the selected map exists and is the intended setting.
3. Confirm extracted parameters match the regulation's intent (watch the
   multi-speed warning).
4. Execute the scenario through the benchmark runner in a controlled setting.
5. Record execution results separately; do not treat the compiled config as
   evidence. Promotion to any evidence use goes through normal scenario
   certification.

## Usage

```bash
# Compile to stdout (repo-relative map path)
uv run python scripts/tools/convert_regulation_to_scenario.py \
    --record configs/scenarios/contracts/issue_6054_regulation_source_example.yaml \
    --stdout

# Compile to a file (output-relative map path) and print the validity report
uv run python scripts/tools/convert_regulation_to_scenario.py \
    --record <regulation-record.v1.yaml> \
    --output-yaml output/regulation_scenarios/example.scenario.yaml \
    --report
```

## Relationship To Existing Work

This prototype mirrors the structure of
`scripts/tools/convert_failure_record_to_scenario.py` (issue #4760): same
input-envelope → deterministic-template → hypothesis-output pattern, same
map-path resolution convention, and the same `required_manual_review` /
`benchmark_evidence: false` posture. It differs by (a) taking a *textual*
regulation excerpt and compiling parameters from it, and (b) reporting
compilation/schema/scenario validity on explicitly separate axes.

It reuses `robot_sf.benchmark.scenario_schema.validate_scenario_list` and
`validate_scenario_matrix_metadata` without modifying them.

## Related Files

- `scripts/tools/convert_regulation_to_scenario.py` — compiler implementation
- `configs/scenarios/contracts/issue_6054_regulation_source_example.yaml` — example input
- `configs/scenarios/single/issue_6054_regulation_to_scenario.yaml` — compiled example output
- `configs/scenarios/sets/issue_6054_regulation_to_scenario.yaml` — optional set manifest
- `tests/scenarios/test_convert_regulation_to_scenario.py` — test suite
- `robot_sf/benchmark/scenario_schema.py` — reused schema validators (not modified)
- `docs/context/issue_4760_failure_record_to_scenario.md` — analogous converter note
