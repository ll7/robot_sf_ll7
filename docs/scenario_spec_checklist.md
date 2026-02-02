Purpose: Provide a practical checklist for authoring benchmark scenarios and archetypes with clear intent, constraints, and validation expectations.

# Scenario Specification Checklist

Use this checklist when creating **per-scenario** files, **archetype** bundles, or **manifest** sets in `configs/scenarios/`.

## 1) Intent & Coverage
- [ ] Declare the interaction **intent** (e.g., crossing, head-on, overtaking, merging).
- [ ] State the **stress axis** (density, obstacle complexity, visibility, speed).
- [ ] Confirm which benchmark suite it targets (classic interactions, Francis 2023, custom set).

## 2) Required Structure
- [ ] `name` or `scenario_id` is set and unique.
- [ ] `map_file` points to a valid SVG/JSON map.
- [ ] `simulation_config.max_episode_steps` set to a realistic horizon.
- [ ] `metadata.archetype` and `metadata.density` filled (for coverage reporting).

## 3) Plausibility & Constraints
- [ ] Spawn/goal feasibility: robot start/goal lie in free space.
- [ ] Path length is reasonable (neither trivially short nor unreachable).
- [ ] Obstacle clearance: no spawn/goal inside obstacles.
- [ ] Pedestrian density matches intent (low/med/high). If outside recommended ranges, record rationale.

## 4) Failure Modes (Expected)
- [ ] Document expected failure cases (e.g., narrow passage stalls, merge conflicts).
- [ ] Note any known planner weaknesses or degenerate layouts.

## 5) Seeds & Reproducibility
- [ ] `seeds` provided for deterministic evaluation when required.
- [ ] For random sampling, note rationale in metadata or experiment docs.

## 6) Manifest & Aggregation
- [ ] Manifests use `includes` to assemble scenarios/archetypes.
- [ ] `map_search_paths` specified when maps are outside default roots.
- [ ] Scenario list order is intentional (deterministic order matters for resume).

## 7) Validation & Preview
- [ ] Run `robot_sf_bench validate-config --matrix <manifest>` and review summary counts.
- [ ] Run `robot_sf_bench preview-scenarios --matrix <manifest>` for warn-only plausibility checks.
- [ ] Confirm archetype/density coverage matches expected suite.

## 8) Documentation
- [ ] Update `configs/scenarios/README.md` if new structure conventions are introduced.
- [ ] If new archetypes are added, mention them in `docs/README.md` or the relevant suite doc.
