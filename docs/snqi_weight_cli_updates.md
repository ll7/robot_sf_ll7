# SNQI Weight CLI Updates

This note documents the addition of external/initial weight file validation to the
SNQI optimization and recomputation scripts.

## New Arguments

Optimization (`scripts/snqi_weight_optimization.py`):

* `--initial-weights-file PATH` – Optional JSON file providing a full weights
  mapping used for reference (not forced as starting point for evolution yet)
  and embedded in the output under `initial_weights` after validation.

Recompute (`scripts/recompute_snqi_weights.py`):

* `--external-weights-file PATH` – Optional JSON file containing a weights
  mapping which will be validated and then evaluated alongside the selected
  or compared strategies. Results appear under `external_weights` with
  statistics and a `correlation_with_recommended` field.

## Validation Rules

Centralized in `robot_sf/benchmark/snqi/weights_validation.py`:

* All required keys defined in `WEIGHT_NAMES` must be present
* Values must be convertible to float, finite, and > 0
* Extraneous keys are ignored with a warning (forward compatibility)
* Very large weights (>10) emit a warning

Failure to pass validation aborts the script with a non‑zero exit code.

## Testing

New test: `tests/test_snqi_external_weights_cli.py` exercises success and
failure paths for both scripts (missing key, non‑numeric value). All existing
SNQI tests remain green (259 passed, 1 skipped at time of change).

## Future Enhancements (Optional)

* Allow seeding differential evolution with `--initial-weights-file` values
* Provide a standalone `snqi-validate-weights` utility entry point
* Add range heuristics (e.g. recommended max) to validator output
