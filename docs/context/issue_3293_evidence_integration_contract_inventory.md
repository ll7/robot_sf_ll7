# Issue #3293 — Evidence-Integration Contract Inventory (Design Slice)

**Status**: Proposal (design-stage). Not benchmark evidence, not calibration evidence, and not a
paper-facing claim.
**Evidence tier**: `proposal` (`evidence:proposal`, `type:synthesis`).
**Issue**: [#3293](https://github.com/ll7/robot_sf_ll7/issues/3293)
(`state:blocked-external-input`, `resource:external-data`).

## Claim boundary

This note and companion code (`robot_sf/research/evidence_integration_inventory.py`,
`scripts/tools/check_evidence_integration_inventory.py`) deliver only the local, no-data slice of
issue #3293. The inventory names evidence streams Robot SF could integrate, the provenance and
uncertainty fields they must carry, and a compact integration report for blockers, invalid
denominator combinations, and the next non-calibrated empirical action.

It deliberately does not:

- ingest real external data,
- compute calibration, fit to reality, or validate metadata field values,
- decide final evidence weighting across streams,
- make any safety, benchmark, operational, or paper-facing claim.

## Maintainer decision

Per maintainer decision on issue #3293 (2026-06-22): a real AMV/micromobility actuation source is
not realistically available (estimated probability below 5%), and implementation is hard-blocked
until a viable real source or collection method exists. The issue remains tracked to preserve
optionality. Any synthetic envelope built on this inventory stays explicitly `non_calibrated`.

AMV means autonomous mobility vehicle.

## Evidence categories

Integration must keep three evidence categories separate because they use different denominators.

| Category | Meaning | Denominator | Example stream |
| --- | --- | --- | --- |
| `calibration` | Adjusts or checks a model against measured reality. | Matched measurements or samples. | `amv_command_response`, `external_pedestrian_trajectory` |
| `benchmark` | Reproducible simulation campaign evidence under a frozen contract. | Scenario/seed episodes. | `simulation_trace` |
| `operational` | Pilot, fleet, or deployment evidence. | Operating time, missions, interventions, or incidents. | `pilot_fleet_operational` |

Pooling calibration residuals, benchmark success rates, and operational incident rates into one
denominator is the primary invalid combination this inventory rejects.

## Evidence-stream inventory

Source of truth: `robot_sf/research/evidence_integration_inventory.py` (`EVIDENCE_STREAMS`).

```bash
uv run python scripts/tools/check_evidence_integration_inventory.py --list
```

| Stream id | Category | Feasibility | Blocked until |
| --- | --- | --- | --- |
| `simulation_trace` | benchmark | `feasible_now` | Not blocked. |
| `amv_command_response` | calibration | `blocked_external` | Real AMV command-response source or field-measurement method. |
| `external_pedestrian_trajectory` | calibration | `partial_external` | Staged, license-cleared dataset via issue #3065 ingestion contract. |
| `pilot_fleet_operational` | operational | `blocked_external` | Pilot deployment or fleet data-sharing arrangement. |

Every stream must carry base provenance fields (`source_id`, `collection_method`,
`license_or_access`, `commit_or_version`, `denominator`, `scenario_link`) and base uncertainty
fields (`uncertainty_basis`, `sample_size`, `calibration_status`). Stream-specific extras are
declared in `EVIDENCE_STREAMS` and printed by `--list`.

The checker (`check_stream_metadata` / `--check`) is presence-only: it reports required keys missing
from a synthetic metadata record. It never inspects values.

## Integration report

The new consolidation capability is:

```bash
uv run python scripts/tools/check_evidence_integration_inventory.py --report
```

`build_integration_report()` / `--report` produces one machine-readable view with:

- `blockers_remaining`: all streams not `feasible_now`, including the hard-blocked
  `amv_command_response` and `pilot_fleet_operational` paths;
- `invalid_combinations`: explicit rules rejecting denominator pooling and AMV calibration claims
  without real command-response data;
- `next_empirical_action`: the first allowed empirical step, still blocked on the issue #3065
  real-trajectory staging contract and limited to a diagnostic, non-calibrated bounded comparison.

This report is the integration slice after the contract-inventory PR. It remains proposal-stage:
no data ingestion, benchmark campaign, calibration, operational safety, planner ranking, or
paper-facing claim is added.

## Conservative integration approaches

These candidate methods are sketches for later work; none is adopted as calibration-grade evidence
until real data exists.

- **Bounded comparison**: compare a simulation distribution against a staged public/external
  trajectory distribution and report only a bound or envelope. This can be diagnostic and
  `non_calibrated` after issue #3065 stages a license-cleared dataset.
- **Calibration check**: compare measured AMV command-response traces against an actuation model.
  This remains hard-blocked on real command-response data.
- **Bayesian-update sketch**: update uncertainty only within one declared evidence category and
  denominator. Cross-category pooling is not allowed.

## First feasible integration path

After issue #3065 lands a staging contract for license-cleared public trajectory data, wire the
inventory checker into that staging preflight and produce one non-calibrated bounded comparison
between `simulation_trace` metadata and the staged external trajectory manifest. The allowed claim
is diagnostic distribution comparison only: no AMV calibration, safety, planner ranking, benchmark,
or paper-facing claim.

## Related work

- #3065 — real-trajectory ingestion and artifact-staging contract.
- #3161, #1559, #2000, #2415 — prior comparison and data lanes named on issue #3293.
- `docs/context/artifact_evidence_vocabulary.md` — canonical evidence-category vocabulary.
- `docs/context/issue_2230_amv_actuation_evidence_ladder.md` and
  `docs/context/issue_2531_amv_trace_boundary.md` — existing AMV evidence-boundary notes.

## Validation

```bash
uv run pytest tests/unit/test_evidence_integration_inventory.py -q
uv run python scripts/tools/check_evidence_integration_inventory.py --list
uv run python scripts/tools/check_evidence_integration_inventory.py --report
uv run ruff check robot_sf/research/evidence_integration_inventory.py \
  scripts/tools/check_evidence_integration_inventory.py \
  tests/unit/test_evidence_integration_inventory.py
```
