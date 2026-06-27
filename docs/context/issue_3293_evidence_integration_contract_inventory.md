# Issue #3293 — Evidence-integration contract inventory (design slice)

**Status**: Proposal (design-stage). Not benchmark evidence, not a calibration or safety claim.
**Evidence tier**: `proposal` (`evidence:proposal`, `type:synthesis`).
**Issue**: [#3293](https://github.com/ll7/robot_sf_ll7/issues/3293)
(`state:blocked-external-input`, `resource:external-data`).

## Claim boundary (read first)

This note and its companion code (`robot_sf/research/evidence_integration_inventory.py`,
`scripts/tools/check_evidence_integration_inventory.py`) deliver **only the local, no-data slice**
of issue #3293: a *contract inventory* that names the evidence streams Robot SF could integrate and
the provenance + uncertainty fields each must carry. It deliberately does **not**:

- ingest any real or external data,
- compute calibration, fit anything to reality, or validate field *values*,
- decide final evidence weighting across streams,
- make any safety, benchmark, or paper-facing claim.

### Maintainer decision honored

Per the maintainer decision on issue #3293 (2026-06-22): a real AMV/micromobility actuation source
is **not realistically available** (estimated probability < 5%), and implementation that depends on
it is **hard-blocked**. The issue is kept tracked, not closed, to preserve optionality. This slice
respects that boundary — it is a non-calibrated design contract. Any synthetic envelope built on
this inventory stays explicitly `non_calibrated` and labeled as such (enforced by the mandatory
`calibration_status` uncertainty field).

## Evidence categories (must not be mixed)

Integration must keep three evidence categories separate because they use different denominators:

| Category | Meaning | Denominator | Example stream |
| --- | --- | --- | --- |
| `calibration` | Adjusts/checks a model against measured reality. | Matched samples. | AMV command-response; external pedestrian trajectories. |
| `benchmark` | Controlled, reproducible simulation under a frozen contract. | Scenario/seed episodes. | Simulation campaign traces. |
| `operational` | Real deployment / pilot / fleet operation. | Operating time, missions, interventions. | Pilot/fleet data. |

A calibration residual is not a benchmark success rate, and neither is an operational incident rate.
Pooling them into one denominator is the primary failure mode this inventory guards against.

## Evidence-stream inventory

Source of truth: `robot_sf/research/evidence_integration_inventory.py` (`EVIDENCE_STREAMS`).
Run `uv run python scripts/tools/check_evidence_integration_inventory.py --list` for the live JSON.

| Stream id | Category | Feasibility | Blocked until |
| --- | --- | --- | --- |
| `simulation_trace` | benchmark | `feasible_now` | — |
| `amv_command_response` | calibration | `blocked_external` | Real AMV command-response source / field-measurement method (#3293 decision). |
| `external_pedestrian_trajectory` | calibration | `partial_external` | Staged, license-cleared dataset via the ingestion contract (#3065). |
| `pilot_fleet_operational` | operational | `blocked_external` | A pilot deployment or fleet data-sharing arrangement. |

### Provenance + uncertainty requirements per stream

Every stream must carry the **base** fields, plus stream-specific extras:

- Base provenance: `source_id`, `collection_method`, `license_or_access`, `commit_or_version`,
  `denominator`, `scenario_link`.
- Base uncertainty: `uncertainty_basis`, `sample_size`, `calibration_status`.

`calibration_status` is mandatory for *every* stream so a synthetic or proxy envelope can never
silently pass as calibrated evidence. Stream-specific extras (e.g. `vehicle_id`/`actuation_units`
for AMV, `dataset_name`/`coordinate_frame`/`frame_rate_hz` for external trajectories) are declared
in `EVIDENCE_STREAMS` and reported by the `--list` CLI.

The checker (`check_stream_metadata` / `--check`) is **presence-only**: it reports which required
keys are missing from a synthetic metadata record. It never inspects values.

## Conservative integration approaches (sketch, not decided)

These are candidate methods to evaluate later; none is adopted here and none is calibration-grade
until real data exists:

- **Bounded comparison** — compare a simulation distribution to a public/external distribution and
  report only a bound/envelope, explicitly `non_calibrated`. Feasible for
  `external_pedestrian_trajectory` once a public dataset is staged via #3065.
- **Calibration check** — residual of model output vs measured response. Requires
  `amv_command_response` data → hard-blocked.
- **Bayesian-update sketch** — treat simulation as prior, real data as likelihood. Requires real
  observations → blocked; documented as a future direction only.

## Feasible now vs blocked

- **Feasible now (no external data):** the contract inventory and presence checker themselves;
  `simulation_trace` provenance/uncertainty declaration on existing campaign outputs.
- **Partial (public data, non-calibrated):** `external_pedestrian_trajectory` bounded comparison,
  gated on #3065 staging.
- **Blocked (external/real data):** `amv_command_response` and `pilot_fleet_operational` —
  calibration/operational claims must wait for real sources.

## First feasible integration — concrete follow-up

**Identified follow-up (not yet filed; issue creation is out of scope for this PR):**
Wire `check_evidence_integration_inventory.py` into the real-trajectory ingestion staging preflight
(#3065) so a staged `external_pedestrian_trajectory` manifest is presence-checked against this
inventory, and produce a single **non-calibrated bounded comparison** between a simulation trace
distribution and the staged public dataset. This is the lowest-risk first integration: it uses an
already-public dataset, stays `non_calibrated`, and makes no AMV/safety claim. It depends on #3065
landing its staging contract first.

## Related work

- #3065 — real-trajectory ingestion + artifact-staging contract (adjacent; this inventory references
  but does not duplicate its manifest/JSON-schema validator).
- #3161, #1559, #2000, #2415 — prior comparison/data lanes named on the issue.
- `docs/context/artifact_evidence_vocabulary.md` — canonical evidence-category vocabulary this note
  specializes for cross-stream integration.
- `docs/context/issue_2230_amv_actuation_evidence_ladder.md`,
  `docs/context/issue_2531_amv_trace_boundary.md` — existing AMV evidence-boundary notes.

## Validation

```bash
uv run pytest tests/research/test_evidence_integration_inventory.py -q
uv run python scripts/tools/check_evidence_integration_inventory.py --list
uv run ruff check robot_sf/research/evidence_integration_inventory.py \
  scripts/tools/check_evidence_integration_inventory.py \
  tests/research/test_evidence_integration_inventory.py
```
