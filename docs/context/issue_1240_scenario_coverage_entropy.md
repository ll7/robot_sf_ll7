# Issue #1240 Scenario Coverage Entropy

Date: 2026-05-16

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1240>

## Decision

Add a v1 config-only scenario coverage entropy report for scenario-set curation. The report is a
diagnostic planning aid, not a benchmark-success, safety, or scenario-promotion metric.

The v1 feature contract uses authored scenario/config fields only:

- archetype/family metadata,
- density label plus a coarse `ped_density` bucket,
- flow and evaluation scope,
- map file name,
- single-pedestrian count bucket,
- wait-behavior and static-marker booleans,
- optional/stress marker,
- seed-count bucket,
- max-episode-step bucket,
- optional platform variant.

Novelty is nearest-neighbor Jaccard distance over those key/value feature tokens. Coverage entropy
is normalized Shannon entropy over all feature tokens. The report must preserve the feature list and
the explicit caveat that these values are diagnostic only.

## Pilot

Command:

```bash
uv run --active python scripts/tools/scenario_coverage_entropy.py \
  configs/scenarios/sets/station_platform_candidate_pack_issue736.yaml \
  --output-json output/scenario_coverage/issue1240_station_platform.json \
  --output-markdown output/scenario_coverage/issue1240_station_platform.md
```

Observed summary:

- schema: `scenario_coverage_entropy.v1`,
- mode: `config_only`,
- scenarios: `4`,
- feature count: `13`,
- coverage entropy: `0.947034`,
- redundant candidates: `0`,
- novel candidates: `4`.

The most novel row was `station_platform_waiting_passengers_medium` with novelty `0.631579` and
distinct `wait_behavior`, density, flow, and pedestrian-count features. This supports retaining it
for further investigation, but it does not prove benchmark value. The optional dense stress row
remains a stress candidate and should not be promoted without runtime and failure-semantics proof.

## Validation

Red proof:

```bash
uv run --active pytest tests/benchmark/test_scenario_coverage.py -q
```

The new test initially failed with:

```text
ModuleNotFoundError: No module named 'robot_sf.benchmark.scenario_coverage'
```

Green proof:

```bash
uv run --active pytest tests/benchmark/test_scenario_coverage.py -q
uv run --active pytest \
  tests/benchmark/test_scenario_coverage.py \
  tests/benchmark/test_runner_scenario_matrix_manifest.py \
  -q
uv run --active ruff check \
  robot_sf/benchmark/scenario_coverage.py \
  scripts/tools/scenario_coverage_entropy.py \
  tests/benchmark/test_scenario_coverage.py
uv run --active python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_1240_scenario_coverage_entropy.md \
  --path docs/context/README.md \
  --path docs/README.md \
  --path configs/scenarios/README.md
git diff --check
uv run --active python scripts/tools/scenario_coverage_entropy.py \
  configs/scenarios/sets/station_platform_candidate_pack_issue736.yaml \
  --output-json output/scenario_coverage/issue1240_station_platform.json \
  --output-markdown output/scenario_coverage/issue1240_station_platform.md
```

Results: targeted scenario coverage tests passed, the include-aware scenario loader regression tests
also passed, Ruff passed, docs proof consistency passed, `git diff --check` passed, and the CLI
smoke wrote ignored outputs under `output/scenario_coverage/`.

The first full PR-readiness run passed the full test suite but failed changed-file coverage because
`robot_sf/benchmark/scenario_coverage.py` was at `78.7%`, below the `80%` minimum. Additional
focused tests now cover minimal legacy rows, empty/duplicate input errors, Markdown rendering, and
artifact writers; the focused coverage for `scenario_coverage.py` is `95.48%`.

## Follow-Up Boundary

Future episode-augmented coverage should be a separate issue. It must validate threshold-profile
and seed-set consistency before comparing outcome distributions, and it should record output
artifacts in benchmark manifests if integrated into camera-ready campaign tooling.
