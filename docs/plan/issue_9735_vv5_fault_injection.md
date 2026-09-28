# VV-5 fault-injection packet (#9735)

## Goal

Measure whether current benchmark checks reject controlled defective inputs. This first packet
covers reset pedestrian overlap, respawn contact, wall-start overlap, and doubled collision
totals. It is diagnostic validation evidence, not release evidence.

## Scope and evidence

- Inject one fault at a time into small synthetic fixtures. No simulator, planner, map, release
  asset, or benchmark result is modified.
- Call the production `reset_spawn_clearance`, `build_spawn_validity`, and
  `analyze_release_rows` checks. Preserve both clean-control and mutant observations.
- Bind the report to SHA-256 of the checker source and map fixture. The PR pins its exact
  branch and base commits separately, so unrelated base movement does not rewrite this report.
- Count detection only when a clean control passes and the mutant fails the named check.

## Steps

1. Build a deterministic command with `--fault` selection and JSON/Markdown output.
2. Exercise every implemented fault against a clean control and save the detection matrix.
3. Test the machine-readable semantics and rerun the command with `--check`.
4. Open a PR and request independent exact-head review.

## Decision and stop rule

A miss or checker error is reported as such and cannot be counted as detection. The six
remaining issue faults are explicitly uncovered by this packet. The matrix's denominator is
only the four implemented injections; no release gate or dissertation claim is admitted from
these fixtures. The collision-total miss is tracked as release-blocking #9855. The next packet
should use versioned throwaway simulation fixtures for the planner, geometry, and
pedestrian-force faults.

## Recovery

The generator is read-only with respect to production inputs. Reports are regenerated from
the command and can be discarded or rebuilt without altering published artifacts.
