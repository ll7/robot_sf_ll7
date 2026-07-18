<!-- AI-GENERATED (robot_sf#5602 / robot_sf#5593, 2026-07-17) - NEEDS-REVIEW -->

# Issue #5602 — Scenario-Evidence Crosswalk Instance: Release 0.0.3 Scenario Matrix

Evidence status: **schema/taxonomy join only; no benchmark, ranking, or paper-facing claim.**

This is a generated `scenario_evidence_crosswalk.v1` instance (issue #5602's tooling, `robot_sf/benchmark/scenario_evidence_crosswalk.py` + `scripts/tools/export_scenario_evidence_crosswalk.py`) over the full 48-scenario matrix that release 0.0.3 / 0.0.3.post1 pins (`configs/scenarios/classic_interactions_francis2023.yaml`, referenced by `configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_release_v0_0_3.yaml` and `configs/benchmarks/paper_experiment_matrix_v1.yaml`). Matrix checksum matches the release-pinned value exactly:

- `matrix_sha256` (release manifest, `paper_experiment_matrix_v2_h600_s30_release_v0_0_3.yaml`): `d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5`
- `sha256(configs/scenarios/classic_interactions_francis2023.yaml)` at generation time: `d9e148e4b544b4c7e2b6ba98e599aef47046d114e0e25645f021946674cb9dc5` (match confirmed)

No new simulation was run to produce this artifact. It is pure config/schema/report generation from already-tracked repository inputs.

## What is populated

- Scenario identity, family, geometry group (explicitly labelled descriptive, not causal), map id, seeds, source config hash: one row for all 48 canonical scenarios, deterministically ordered.
- `predicate_availability.motivated_not_exported_predicates`: the three taxonomy-motivated predicates (`oscillatory_control`, `late_evasive`, `occlusion_near_miss`) per row, so a reader can see what predicate lane *would* attach if a predicate export were supplied.

## What is explicitly `unavailable` / `excluded`, and why

- `predicate_availability.export_status = "unavailable"` for all 48 rows (`predicate_export_available: 0` in the summary). No `--predicate-export` manifest was supplied. A real trace-level predicate export (issue #5593, `scripts/tools/export_trace_predicates.py`) requires a completed campaign bundle with a recorded `safety_predicates` block per episode. The one locally present campaign that spans this scenario matrix at scale, `output/issue4365-s30-run/13378/` (job 13378, see `../issue_4365_job_13378_closeout/README.md`), is explicitly recorded there as **artifact-integrity blocked**: its non-PPO episode files contain duplicate-appended commit blocks and its PPO file mixes partial prior blocks with one complete final block, so per-scenario episode counts cannot be trusted without the exact final-commit slice logic that closeout note performs. Feeding that bundle through the predicate exporter without reproducing that reconciliation would risk laundering a flagged bundle into a newly-pinned artifact, so it was deliberately not used here. Populating this field with real data is future work gated on either a clean completed campaign bundle or a documented reconciliation of job 13378.
- `evidence.eligibility = "excluded"` / `exclusion_reason = "no_evidence_bundle_provided"` for all 48 rows (`eligible_scenarios: 0`). No `--evidence-catalog` was supplied; this run only exercises the taxonomy + predicate-lane join, not trace/replay/case-capsule eligibility. Fails closed by design (issue #5602 acceptance criteria: "Trace/capsule eligibility fails closed on degraded or provenance-incomplete inputs").

## Summary

| Field | Value |
| --- | --- |
| `schema_version` | `scenario_evidence_crosswalk.v1` |
| Source | `configs/scenarios/classic_interactions_francis2023.yaml` |
| Repo commit | `f1aead35f` (`f1aead3`) |
| `content_sha256` | `20ecce670caea6d35ddd4f55af4e013dd3fb2ec823df26ded63df5bf990a79d6` |
| Scenarios | 48 |
| Eligible (evidence) | 0 |
| Excluded (evidence) | 48 |
| Predicate export available | 0 |
| Predicate unavailable | 48 |

## Claim boundary

Benchmark-evidence metadata join only (verbatim from the artifact's own `claim_boundary` field): "Geometry groups are descriptive topology labels and never imply a causal failure mechanism; predicate availability is consumed from the #5593 export lane when present and is explicit 'unavailable' otherwise; validated mechanisms require a validated causal report and are not derived here. No private manuscript claims are encoded." This bundle does not certify scenario feasibility, does not rank planners, does not run a benchmark campaign, and does not establish a paper-facing or dissertation claim by itself. It answers one narrow question mechanically: for the release 0.0.3 scenario matrix, is each predicate type actually exported (vs. only motivated), scenario by scenario -- and today the honest answer is "not yet, for any row," which this artifact now makes checkable in one place instead of requiring issue archaeology.

## Reproduce

```bash
uv run python scripts/tools/export_scenario_evidence_crosswalk.py \
  configs/scenarios/classic_interactions_francis2023.yaml \
  --output-json docs/context/evidence/issue_5602_scenario_evidence_crosswalk_release_0_0_3_2026-07-17/scenario_evidence_crosswalk.json \
  --output-markdown docs/context/evidence/issue_5602_scenario_evidence_crosswalk_release_0_0_3_2026-07-17/scenario_evidence_crosswalk.md \
  --output-csv docs/context/evidence/issue_5602_scenario_evidence_crosswalk_release_0_0_3_2026-07-17/scenario_evidence_crosswalk.csv
```

Validation (both the repo's own JSON Schema validator and independent `jsonschema.validate` against `robot_sf/benchmark/schemas/scenario_evidence_crosswalk.v1.json` pass with zero errors):

```bash
uv run python -c "
import json
from robot_sf.benchmark.scenario_evidence_crosswalk import validate_scenario_evidence_crosswalk
d = json.load(open('docs/context/evidence/issue_5602_scenario_evidence_crosswalk_release_0_0_3_2026-07-17/scenario_evidence_crosswalk.json'))
print(validate_scenario_evidence_crosswalk(d))
"
```

## Cross-links

- robot_sf#5602: scenario-evidence crosswalk schema/builder/CLI (tooling, merged; this bundle is a generated instance of it).
- robot_sf#5593: trace-level predicate export lane + crosswalk bridge (`build_crosswalk_predicate_export`, merged); the predicate lane this instance would consume once a clean campaign bundle is available.

<!-- /AI-GENERATED -->
