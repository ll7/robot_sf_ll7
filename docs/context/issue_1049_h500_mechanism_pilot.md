# Issue 1049 H500 Mechanism Pilot

Date: 2026-05-07

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1049>

## Goal

Run a small fixed-horizon versus h500 trace pilot that can distinguish route/time-budget relief,
exposure-enabled completion, and safety-regressed long-horizon behavior without treating aggregate
fixed-to-h500 deltas as causal proof.

## Evidence

Tracked compact evidence:

* `docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/README.md`
* `docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/representative_trace_summary.csv`
* `docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/representative_trace_summary.json`
* `docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/traces/`

Source aggregate analysis:

* `docs/context/issue_1045_h500_solvability_mechanisms.md`
* `docs/context/evidence/issue_1045_h500_solvability_mechanisms_2026-05-07/h500_solvability_cases.csv`

The selected traces use a temporary diagnostics registry under
`output/ai/autoresearch/1049_h500_mechanism_pilot/tmp_registry` with `algo: orca` and no parameter
overrides, matching the paper-matrix core ORCA row more closely than the policy-search ORCA
candidate. The temporary registry is not durable; the selected compact per-step traces and summary
tables are tracked under `docs/context/evidence/`.

## Selected Cells

| Mechanism target | Planner | Scenario | Seed | Fixed side | H500 side | Interpretation |
|---|---|---|---:|---|---|---|
| `budget_limited_clean_completion` | ORCA | `classic_bottleneck_low` | 111 | h100 reaches 100 steps without route completion. | h500 succeeds at step 102, with no near-miss, force-exposure, or collision events. | Clean route/time-budget relief. This trace does not support a wait-then-go pedestrian claim because there are no pedestrians in the recorded min-distance stream. |
| `exposure_enabled_completion` | ORCA | `classic_t_intersection_medium` | 111 | h100 reaches 100 steps without route completion; 9 force-exposure steps, comfort exposure sum 3.0, min pedestrian distance 2.235 m. | h500 succeeds at step 182; 50 force-exposure steps, comfort exposure sum 16.667, min pedestrian distance 1.413 m. | H500 enables completion while increasing interaction exposure and comfort pressure. The discrete `near_misses` counter stays zero, so this is not near-miss timing proof. |
| `safety_regressed_completion` | ORCA | `classic_merging_low` | 111 | h100 reaches 100 steps without route completion and no collision. | h500 reaches a collision at step 272 after force exposure begins at step 259. | Longer horizon exposes unsafe behavior hidden by fixed-horizon timeout. This seed is a safety-regression example, not a clean h500 completion. |

## Mechanism Conclusions

The pilot supports three bounded claims:

1. H500 can convert a strict fixed-horizon timeout into a clean completion when the remaining
   issue is route/time budget rather than pedestrian interaction.
2. H500 can convert a timeout into completion while increasing force/comfort exposure and reducing
   pedestrian clearance; this must be reported as exposure-aware completion, not a pure success.
3. H500 can also extend an interaction long enough to reveal a collision that fixed h100 would have
   reported only as an unfinished run.

The pilot does not support the stronger claim that representative h500 wins mostly come from
waiting until pedestrians pass. The selected ORCA h500 successes show low mean absolute linear
actions in the compact summaries, but the trace runner does not yet separate intentional yielding
from local-control conversion, route following, or simulator action clipping. Treat wait-then-go as
unsupported until video or richer planner-decision traces are available.

## Validation Commands

Representative trace runner command shape:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python \
  scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate paper_orca_baseline \
  --candidate-registry output/ai/autoresearch/1049_h500_mechanism_pilot/tmp_registry/candidate_registry.yaml \
  --stage full_matrix \
  --scenario-name <scenario_id> \
  --seed <seed> \
  --horizon <100-or-500> \
  --output-dir output/ai/autoresearch/1049_h500_mechanism_pilot/traces_paper_orca/<run_id>
```

Additional validation:

* `rtk uv run python -m py_compile scripts/validation/run_policy_search_step_diagnostics.py`

## Follow-Up Boundary

Issue #1056 classifies these findings in
`docs/context/issue_1056_h500_failure_classification.md` and distinguishes seed-level success,
exposure, and collision outcomes across scenario families. Issue #1055 should turn this into
exposure-aware tables. This pilot is intentionally small and should not be used as a full h500
ranking.
