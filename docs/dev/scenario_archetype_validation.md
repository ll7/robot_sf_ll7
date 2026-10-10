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


## Release robot endpoint separation (#10063)

The existing geometry validator also audits all 48 versioned 0.0.8 scenarios and
three doorway widths, without creating or stepping environments:

```bash
uv run python scripts/validation/check_scenario_archetype_geometry.py \
  --release-zones \
  --endpoint-policy pedestrian_radius_v1 \
  --waiver-file configs/scenarios/release_0_0_8_endpoint_dispositions.yaml
```

CI runs this blocking command. The JSON report lists every robot spawn and goal
rectangle, including the implicit fourth corner, with single-pedestrian lanes
(start through resolved trajectory/goal) and crowd spawn rectangles within the
resolved pedestrian radius. Tangency and radius-only intersections count; a
1e-9 m conservative comparison allowance includes floating-point contact error. It uses
the scenario loader so YAML actor overrides are checked, and reports dormant
crowd zones explicitly. The default scenario radius is 0.4 m, conservatively
including the substrate's 0.35 m physical radius; this is a static lane check,
not a guarantee against dynamic contact or role-conditioned motion.

Every remaining intentional interaction has an exact matrix/scenario/zone/actor
identity and geometry fingerprint in the disposition file, with its rationale.
Opposite-end head-on passage, occupied destination entry and interior crossings
remain deliberate interactions. Missing, stale, duplicate or changed dispositions
fail; no scenario-wide exemption or global tolerance hides new overlap.

The overtaking spawn now spans y=4.0–4.5 beside h1's y=6.6 lane (formerly y=6), instead
of y=4.0–6.0 across that lane. The robot route stays at y=5, leaving a 1.6 m
passing gap. h1 retains start x=1.5 and authored initial speed 0.8 m/s
(effective SFM desired speed 1.04 m/s = 0.8 × 1.3). Its two-leg trajectory is
`(1.5,6.6) → (33,6.6) → (20,9.9)` through `poi_h1_pass` and `poi_h1_goal`.
The room grows from 40×10 to 40×12 and the upper wall moves y=9→11; the full
0.6 m parking region plus 0.4 m pedestrian body clears that wall by 0.1 m.
Parking clearance tests include the resolved robot footprint plus 1 m margin.
The 0.7 m/s cap preserves a prompt overtake rather than just ordering at the
first turn; the author-granted 600-step budget is inherited from the source
scenario without a matrix override. If an authored schedule exists, the
static witness also checks its budget. Schedule/feasibility-guard integration
is owned by #9999. The existing scenario `metadata.plausibility.notes` carries
D-085's guarded-PPO cell caveat: the policy runs outside its trained speed
range (2.0 m/s policy maximum versus the 0.7 m/s cap). The refute traces show
about 94% speed saturation, 0.5–1 m terminal misses and an inactive guard;
these are cap/budget effects, not parked-pedestrian obstruction. Stale
plausibility metrics are cleared. Station-platform reverse crowd spawn moves from
y=20–23 to y=16.5–19.5 and its departure route skirts the robot goal. Robot-crowding
uses x=6.5–14.5 instead of x=3–17 with density 0.21 preserving 24 pedestrians.
These release-only successor inputs leave historical SVGs and frozen artifacts intact.

Dormant crowd dispositions bind both density and `population_size`: a forced
positive count at density zero changes the fingerprint and requires a fresh
disposition. The audit still lists dormant geometry rather than omitting it.

Route crowds are also checked using the actual sampler support: a route anchor
plus independently clipped x/y offsets of +/-1.5 m (the default 3 m sidewalk
width). This is a square Minkowski sum, including diagonal segment corners,
not a round centreline buffer. The station bend at x=74 leaves a 1.5 m gap
between that entire support and the robot goal. Other shared moving-flow
intersections have individual exact dispositions; nominal support is not a
claim of runtime clearance or release admission.

## Full-footprint endpoint audit for 0.1.0 (#10091)

The default policy is now `robot_pedestrian_radii_v2`: it compares each full
robot rectangle with actor support using the sum of the effective robot and
pedestrian radii. Geometry fingerprints include both radii and the policy.
The explicit historical policy above preserves 0.0.8 fingerprints and CI's
existing disposition contract. Old waivers cannot authorize the new policy.

[The 0.1.0 endpoint dispositions](../scenario_endpoint_footprint_0_1_0.md)
record the 12 additional findings in seven scenarios and their authoring
implications. These are retained risks, not clearance certificates.
