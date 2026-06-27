# Issue #2415 AMV command-response trace staging manifest (2026-06-27)

This note records the local, buildable half of issue #2415. The full goal the issue asks for --
stage *real* AMV command-response actuation traces so the synthetic actuation envelope
(`robot_sf/benchmark/synthetic_actuation.py`) can be calibrated against measured command-response
dynamics -- is **blocked on external data**. Per the maintainer decision on #2415 (2026-06-22) no
realistic real-data source is currently available (maintainer-estimated probability < 5%), and
Robot SF never ingests or redistributes raw command-response traces.

What ships here is a **metadata-only staging manifest and preflight**, so that when a real trace
bundle is later staged through the `amv-calibration` external-data path
(`scripts/tools/manage_external_data.py`) a command-response calibration can ingest it without
guesswork. This preserves the issue's intent ("preserve as calibration-path gate for future AMV
work", "define trace-source acceptance criteria") without touching the hard-blocked implementation.

> Evidence tier: `analysis_only` staging manifest. This is **not** a staged trace bundle, an
> executed calibration, or a hardware-calibrated realism claim. Every report stamps
> `evidence_boundary = manifest_contract_only_no_trace_ingest_no_calibration_run_no_calibrated_claim`.

## What was added

- Manifest schema: `robot_sf/research/schemas/amv_command_response_trace_manifest.v1.json`
- Preflight module: `robot_sf/research/amv_command_response_trace_manifest.py`
  (`load_amv_trace_manifest`, `check_amv_trace_manifest`).
- Example manifest: `configs/research/amv_command_response_trace_manifest_issue_2415.yaml`
  (one trace bundle, `blocked-external-input` today).
- CLI: `scripts/validation/check_amv_command_response_trace_manifest_issue_2415.py`.
- Tests: `tests/research/test_amv_command_response_trace_manifest.py`.

## What the manifest declares per trace bundle

- **Provenance / license** -- source URL, license, `license_status`
  (`license-gated` | `open` | `unknown`), and citation.
- **Command / response / timing channels** -- a command-response trace must declare at least one
  command channel (e.g. `linear_velocity_command_mps`, `yaw_rate_command_rad_s`), one response
  channel (e.g. `measured_linear_velocity_mps`, `measured_yaw_rate_rad_s`), and the timing/latency
  fields a calibration needs (e.g. `timestamp_s`, `control_latency_s`, `sample_rate_hz`). These are
  schema-enforced (`minItems: 1`).
- **Calibration targets** -- the canonical synthetic-actuation envelope fields a calibrated trace
  would inform (`max_linear_accel_m_s2`, `max_linear_decel_m_s2`, `max_yaw_rate_rad_s`,
  `max_angular_accel_rad_s2`, `latency_mode`, `update_mode`). These are validated against the
  **live** `actuation_variability_fields()` vocabulary of the synthetic actuation envelope
  (`robot_sf/benchmark/synthetic_actuation.py`), so the manifest cannot silently drift from what
  calibration can actually consume.
- **Explicit external-data blocker** -- a `blocked-external-input` trace must name the staging
  issue(s) holding it (#2000 real command-response trace acquisition, #1585 calibration-source
  gate).
- **Redistribution** -- pinned to `none`; raw external command-response traces are never
  redistributed.

## Why route through `amv-calibration`

The `amv-calibration` AssetSpec in `scripts/tools/manage_external_data.py` already owns the
external-data path for AMV calibration provenance, and lists a "command-response trace bundle" as an
acceptable source class -- but it only validates *file presence* (any `.json`/`.yaml`/`.csv`/`.pdf`
in the bundle). It does not validate that a staged trace bundle declares the command/response/timing
structure and calibration targets a calibration needs. This manifest preflight fills that gap
without duplicating the asset registry: it is the structural acceptance contract layered on top of
the existing presence check.

## Fail-closed behavior

- A declared `staged` trace is reconciled against a live `manage_external_data.check_asset` presence
  probe (`--probe-live-staging`). If the files are not actually present, the trace fails closed
  (`effective_staged = false`) so a calibration can never run on nothing.
- An unknown calibration target, a blocked trace with no blocker issue, or a declared-vs-live
  mismatch makes the manifest `invalid`.
- With no staged trace (the state today), the manifest resolves to `blocked-external-input` rather
  than substituting a synthetic stand-in -- matching the issue's acceptance / stop rule and the
  #2415 maintainer decision.

## Validation

```
uv run pytest tests/research/test_amv_command_response_trace_manifest.py
uv run python scripts/validation/check_amv_command_response_trace_manifest_issue_2415.py
```

The CLI prints `manifest_status: blocked-external-input` and `calibration_ingest_allowed: false`
today, which is the expected (and only honest) result until a real source is found.

## Out of scope / residual risk

- No real command-response trace is ingested, staged, or copied. The calibration itself stays
  hard-blocked until #2000/#1585 identify a viable source.
- No synthetic AMV diagnostic is promoted into calibrated evidence; no paper-facing AMV actuation
  calibration claim is made or edited.
- The command/response/timing channel *names* in the example manifest are the expected schema for a
  command-response bundle; the exact channel set a future real source exposes may differ and would
  be reconciled when a source is accepted.

## Related

- Issue #2415 (this work), blockers #2000 / #1585; parent tracking #2000.
- Sibling staging contract: `docs/context/issue_3161_scenario_prior_staging_contract.md`
  (dataset-backed scenario priors; the same metadata-only / fail-closed pattern).
- Synthetic actuation envelope: `robot_sf/benchmark/synthetic_actuation.py`.
- External-data asset registry: `scripts/tools/manage_external_data.py` (`amv-calibration`).
