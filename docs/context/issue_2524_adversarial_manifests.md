# Issue #2524 Adversarial Scenario Manifest Generation (2026-06-07)

Status: proposal/implementation evidence only. This note does not claim planner weakness,
adversarial coverage, or benchmark-strength evidence.

Related surfaces:
- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2524
- Adversarial generation roadmap: `docs/context/issue_2468_adversarial_generation_roadmap.md`
- Generation protocol: `docs/context/issue_1457_adversarial_generation_protocol.md`
- Frozen campaign manifest: `docs/context/issue_1500_adversarial_manifest.md`
- Panel candidate manifest: `docs/context/issue_2270_panel_candidate_manifest.md`
- Evidence: `docs/context/evidence/issue_2524_adversarial_manifests/summary.json`

## Result

This issue implements the first bounded `adversarial_scenario_manifest.v1` candidate generator
and validator, following the common-manifest recommendation from
[issue #2468](https://github.com/ll7/robot_sf_ll7/issues/2468). The generator accepts any
`SearchSpaceConfig`-compatible search space and produces deterministic candidate manifests with
fail-closed validation.

## What Was Built

### Core module: `robot_sf/adversarial/scenario_manifest.py`

- `AdversarialScenarioManifest` dataclass with `schema_version`, `source`, `generator`,
  `candidate_controls`, `validation`, `execution_status`, and `evidence_boundary`.
- `build_manifest()` builds a validated manifest from a `CandidateSpec`.
- `validate_candidate_manifest()` checks all constraints from the task:
  - non-finite values -> degenerate
  - out-of-bounds search-space values -> invalid
  - non-positive speed, negative timing, too-short route -> degenerate
  - duplicate normalized control hashes -> degenerate (warning)
- `validate_manifest_payload()` validates serialized manifest payloads and rejects malformed schema
  or missing control fields before any generated candidate is reused.
- `compute_control_hash()` produces a deterministic 16-char SHA-256 hash of rounded control values.
- `generate_manifests()` batch-generates N manifests from `RandomCandidateSampler` + `SearchSpaceConfig`.
- Serialization: `to_yaml()` / `from_yaml()` round-trip through plain YAML.

### CLI: `scripts/tools/generate_adversarial_scenario_manifests.py`

Accepts `--search-space`, `--scenario-template`, `--count`, `--seed`, `--output-dir`.
Writes `candidate_{index:04d}.yaml` per candidate plus `summary.json` with counts and rejection reasons.

### Tests: `tests/adversarial/test_adversarial_scenario_manifest.py`

- Deterministic generation
- Valid/invalid/degenerate classification for each error class
- Duplicate hash detection
- YAML round-trip
- Serialized payload validation for bad schema, missing controls, degenerate controls, and
  duplicate controls
- CLI output shape and error handling

### Output schema (`adversarial_scenario_manifest.v1`)

```yaml
schema_version: adversarial_scenario_manifest.v1
execution_status: generated_only
evidence_boundary: "diagnostic-only: no planner weakness, ..."
source:
  scenario_template: crossing_ttc.yaml
  search_space: crossing_ttc_space.yaml
  map_id: classic_cross_trap
  scenario_name: crossing_ttc_template
generator:
  family: random
  generator_id: RandomCandidateSampler
  seed: 42
  candidate_index: 0
candidate_controls:
  start: {x: 1.0, y: 2.0}
  goal: {x: 5.0, y: 2.0}
  spawn_time_s: 0.0
  pedestrian_speed_mps: 1.0
  pedestrian_delay_s: 0.0
  scenario_seed: 7
validation:
  status: valid
  errors: []
  warnings: []
  normalized_control_hash: a1b2c3...
```

## Evidence

`docs/context/evidence/issue_2524_adversarial_manifests/summary.json` contains a compact
16-candidate generation summary from the crossing_ttc search space.

Validation on this branch:

```bash
uv run pytest tests/adversarial/test_adversarial_scenario_manifest.py -q
uv run pytest tests/adversarial -q
uv run ruff check robot_sf/adversarial/scenario_manifest.py robot_sf/adversarial/__init__.py \
  scripts/tools/generate_adversarial_scenario_manifests.py \
  tests/adversarial/test_adversarial_scenario_manifest.py
uv run ruff format --check robot_sf/adversarial/scenario_manifest.py robot_sf/adversarial/__init__.py \
  scripts/tools/generate_adversarial_scenario_manifests.py \
  tests/adversarial/test_adversarial_scenario_manifest.py
uv run python scripts/tools/generate_adversarial_scenario_manifests.py \
  --search-space configs/adversarial/crossing_ttc_space.yaml \
  --scenario-template configs/scenarios/templates/crossing_ttc.yaml \
  --count 16 --seed 42 \
  --output-dir output/adversarial/issue2524_manifest_generation_smoke
python -m json.tool docs/context/evidence/issue_2524_adversarial_manifests/summary.json
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
```

Observed result: focused manifest tests passed with `39 passed`; the adjacent adversarial suite
passed with `88 passed`; Ruff check and format check passed. The CLI smoke generated 16 ignored
local candidate YAML manifests plus a summary under
`output/adversarial/issue2524_manifest_generation_smoke`, all classified valid. The tracked summary
JSON parsed successfully, and the context catalog consistency check passed.

## Claim Boundary

- **Diagnostic-only**: the generated manifests are validator-checked candidates that have not been
  run through any planner. No planner weakness, adversarial coverage, or benchmark-strength claim
  is made.
- The generator is bounded to `SearchSpaceConfig`-compatible random search. RL, diffusion, or
  LLM-assisted generators would need their own adapters to emit the same manifest schema.

## Follow-Up Risks

- The manifest schema is intentionally minimal. Additional fields (pedestrian lists, route overrides,
  objective weights) should be added when the panel candidate manifest (#2270) or the RL adversary
  (#2470) needs them.
- The `evidence_boundary` field is a string caveat, not an enforced gate. A future certifier or
  benchmark runner should reject manifests that claim benchmark evidence without separate promotion.
