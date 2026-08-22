# Pinned scenario-archetype validation

The geometry and declared-vs-runtime parameter checkers are read-only diagnostics for the four
pinned scenario archetypes. Continuous integration (CI) runs them in fail-closed mode so an
accepted finding is allowed only when its exact current identity and evidence are recorded in
[`configs/scenarios/archetype_validation_waivers.yaml`](../../configs/scenarios/archetype_validation_waivers.yaml).

This contract does not change SVG maps, scenario configuration, parser behavior, runtime semantics,
simulation, training, benchmark results, or paper-facing claims. The current waiver rows reflect
the maintainer's [#7709 ruling](https://github.com/ll7/robot_sf_ll7/issues/7709#issuecomment-5381550262).

## Run the checks

Informational mode remains useful while investigating a change:

```bash
uv run python scripts/validation/check_scenario_archetype_geometry.py --json
uv run python scripts/validation/check_scenario_archetype_parameters.py --json
```

The blocking contract requires the checked-in waiver file explicitly:

```bash
uv run python scripts/validation/check_scenario_archetype_geometry.py \
  --fail-on-violation \
  --waiver-file configs/scenarios/archetype_validation_waivers.yaml
uv run python scripts/validation/check_scenario_archetype_parameters.py \
  --fail-on-violation \
  --waiver-file configs/scenarios/archetype_validation_waivers.yaml
```

Fail-on-violation mode rejects a missing waiver file and rejects any finding that does not match
exactly one waiver. A stale waiver, duplicate identity, changed measurement, changed route/zone
fingerprint, or new finding is an error; broad map-wide exemptions and tolerance increases are not
valid updates.

## Waiver schema

The file declares `schema: scenario_validation_waivers.v1` and has `geometry` and `parameters`
sections. Every row includes a non-empty `rationale` and `decision_ref`.

Geometry rows identify the `map`, `finding_type`, `route_kind`, and source `label`. Endpoint rows
also identify `end`, `zone_kind`, `zone_index`, and `expected_offset_to_centre_m`. Fragment rows
identify `first_disconnected_segment` and `expected_disconnected_fragment_count`. Missing-zone rows
identify `expected_route_count`.

Parameter rows identify `source`, `scenario`, `parameter`, and `expected_driver`, together with
`expected_declared_value` and `expected_runtime_value`.

Identity matching is one-to-one. The checker compares the measured evidence after matching the
identity, so a row remains a waiver only while the current diagnostic reproduces the documented
finding.

## Updating a row

1. Run both checkers in informational JSON mode and inspect the complete finding identity and
   measurement.
2. Obtain the maintainer or decision reference that classifies the finding as an accepted pinned
   contract. Do not use a waiver to hide an unreviewed map, runtime, or parameter change.
3. Add, remove, or edit only exact rows in the versioned YAML. Keep the current measurement or
   geometry fingerprint, a short rationale, and a durable decision link together.
4. Run the focused tests, both blocking commands, Ruff, and `git diff --check`. A new finding must
   fail before its exact disposition is reviewed.

The resulting evidence is limited to enforcement of the pinned diagnostic contract. Passing these
checks is not evidence of scenario feasibility, planner performance, safety, or benchmark validity.
