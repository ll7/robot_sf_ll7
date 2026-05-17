# Issue #1271 Seed-Sensitivity Explorer

## Goal

Issue #1271 adds a small API for replaying one adversarial candidate across explicit scenario
seeds so maintainers can classify a counterexample as a stable failure mode or a brittle
single-seed artifact.

## Contract

The v1 surface is `robot_sf.adversarial.seed_sensitivity.run_seed_sensitivity`. It accepts an
existing `SearchConfig`, a `CandidateSpec`, an explicit seed list, an output directory, and an
evaluator compatible with `run_adversarial_search`. The helper writes replay candidate bundles and
`seed_sensitivity_summary.json` with schema version `adversarial-seed-sensitivity.v1`.

The summary records:

* replay seeds and per-seed outcomes,
* failure persistence rate over evaluated replays,
* objective score spread when objective scores are available,
* fail-closed certification exclusions,
* classification: `stable_failure`, `brittle_failure`, or `no_failure`.

Seed perturbations are intentionally allowed outside the original search-space seed range because
the candidate geometry is fixed and the seed list is the controlled independent variable. Other
contract checks still come from `SearchConfig.validate`, candidate materialization, certification,
and the injected evaluator.

## Evidence Boundary

This is development evidence for adversarial analysis and a feeder to Issue #1237 failure archives.
It is not a durable benchmark claim by itself. Generated summaries under `output/` remain local
until promoted through the repository artifact policy with provenance, checksums, and a durable
storage pointer.

Fail-closed certification exclusions are recorded explicitly and do not count as persisted
failures. Fallback or rejected perturbations therefore remain caveats, not successful
counterexample evidence.

## Validation

Focused proof:

```bash
uv run --active pytest \
  tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_classifies_stable_and_brittle_failures \
  tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_records_fail_closed_rejected_perturbations \
  -q
```

The tests cover stable failure classification, brittle failure classification, objective spread,
summary-file creation, and fail-closed rejected perturbation accounting with injected evaluators so
the proof does not require a long benchmark run.

API smoke proof:

```bash
uv run --active python - <<'PY'
# Build SearchConfig from configs/scenarios/templates/crossing_ttc.yaml and
# configs/adversarial/crossing_ttc_space.yaml, inject a deterministic evaluator,
# and call run_seed_sensitivity over seeds [100, 101].
PY
```

Observed result: `stable_failure`, `failure_persistence_rate=0.5`,
`objective_score_spread=11.0`, summary written to
`output/adversarial/issue1271_seed_sensitivity_smoke/seed_sensitivity_summary.json`.
That output is disposable local smoke evidence unless promoted through the artifact policy.

Before PR handoff, also run:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Related

* GitHub issue: <https://github.com/ll7/robot_sf_ll7/issues/1271>
* Failure archive feeder: <https://github.com/ll7/robot_sf_ll7/issues/1237>
* Existing optimizer sampler context: [issue_1236_optimizer_adversarial_sampler.md](issue_1236_optimizer_adversarial_sampler.md)
