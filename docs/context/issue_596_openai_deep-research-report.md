# Prompt Template and Scenario Suite for 2D Local Navigation Sanity Testing

## Executive Summary

This report proposes a repository-aligned way to specify *simple, isolatable* 2D navigation scenarios (from 3 up to ~20+) intended to stress-test local navigation stacks and their integration points (coordinate transforms, goal handling, static obstacle avoidance, and first-order social navigation with one pedestrian). The target format matches the scenario YAML conventions and SVG map semantics used in the referenced `configs/scenarios/` and `maps/` directories. citeturn44view0turn14view0turn11view0

The core deliverable is a reusable **prompt template** that instructs a generator (human or LLM) to output: (i) a manifest or per-scenario YAML file(s) in the project’s “scenario list” schema, and (ii) minimal map-edit instructions consistent with the SVG labeling rules (spawn/goal zones, routes, obstacles, POIs, single pedestrians). citeturn44view0turn14view0turn28view0

A concrete **suite of 20 minimalist scenarios** is provided as a scenario matrix. Each scenario targets a specific failure mode (e.g., axis/diagonal symmetry breaks, angle wrapping, local minima, cornering oscillations, reciprocal deadlocks, slow obstruction). This design philosophy is consistent with known local-planning failure patterns (local minima/oscillation and non-convexity) and with common reciprocal avoidance behaviors in multi-agent settings. citeturn38view0turn41view0turn40view0

## Repository-grounded scenario and map primitives

The repository’s scenario directory supports **manifests** (via `includes`) that compose per-scenario YAML files into a single ordered scenario list. It also documents a standard metadata structure (including `plausibility` tracking and interaction metrics) and two map reference modes: `map_file` (path) and `map_id` (registry key resolved via `maps/registry.yaml`). citeturn44view0turn12view0

A minimal “single scenario” file in this schema is exemplified by `planner_sanity_simple.yaml`, which uses:
- `scenarios:`
- `name`
- `map_file`
- `simulation_config` (e.g., `max_episode_steps`, `ped_density`, `single_pedestrians`)
- `robot_config`
- `metadata`
- `seeds` citeturn11view0

The SVG maps are not free-form: the map editor documentation mandates **specific SVG element labels** (via `inkscape:label`), including:
- `obstacle` (rectangles)
- `robot_spawn_zone(_i)`, `robot_goal_zone(_j)` (rectangles)
- `robot_route_<spawn>_<goal>` (path with waypoints)
- `ped_spawn_zone(_i)`, `ped_goal_zone(_j)` and `ped_route_<spawn>_<goal>` (if crowd routes are used)
- optional `poi` circles (named waypoints)
- optional `single_ped_<id>_start` and `single_ped_<id>_goal` (circles for individually controlled pedestrians) citeturn14view0

The same document prescribes a verification workflow (`scripts/validation/verify_maps.py`) to validate SVG structure and compatibility. citeturn14view0

Single-pedestrian scenarios in `configs/scenarios/single/` show the intended pattern for “one pedestrian” interactions: `ped_density: 0.0` with a `single_pedestrians` list specifying at least an `id` and a goal reference (e.g., `goal_poi`), optionally overriding speed (`speed_m_s`) and annotating intent (`note`). citeturn28view0turn29view1

At the parsing layer, the map configuration code explicitly supports single pedestrians with fields including `id`, `start`, optional `goal`, optional `trajectory`, optional `speed_m_s`, and optional wait rules. This supports designing “simple but diagnostic” interactions by constraining a single pedestrian’s behavior without introducing full crowd randomness. citeturn43view1

Finally, obstacle handling in the navigation utilities is implemented as **continuous collision checks against line segments** (not a rasterized costmap). The module explicitly states it is not an occupancy grid and provides circle–segment intersection checks. Combined with the historical issue describing “edge-only” obstacle collision behavior (and spawn-inside-obstacle risk), this motivates cautious scenario design: avoid spawns inside obstacle interiors and prefer wall-like obstacles or thin rectangles when the intent is to represent barriers. citeturn34view0turn30view0

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["top-down 2D robot navigation obstacle avoidance map","robot navigation corridor corner top-down diagram","social navigation robot pedestrian interaction top-down simulation"],"num_per_query":1}

## Failure-mode taxonomy for minimalist local-navigation scenarios

Local navigation failures are often easiest to diagnose when scenarios eliminate confounds and vary **exactly one factor** at a time. Three broad classes are particularly useful for “sanity suites”:

Frame/geometry consistency failures arise when coordinate transforms, axis conventions, angle normalization, or map-to-planner projection are inconsistent. These typically present as mirrored steering, sign flips, diagonal drift, or discontinuities near ±π. A dedicated axis/diagonal sweep is a low-cost way to surface these integration faults early. citeturn14view0turn44view0

Static-obstacle interaction failures include oscillation in corridors, inability to pass narrow gaps, and local minima in concave geometries. Classic reactive planning literature notes oscillatory behavior in narrow corridors and local-minima issues for purely local methods; similarly, trajectory-optimization approaches emphasize that obstacle-induced non-convexity yields multiple local optima and requires careful benchmarking across topologies. citeturn38view0turn41view0

Dynamic-agent interaction failures become visible even with **one** pedestrian: deadlocks, excessive conservatism, “reciprocal dance” oscillations, or unsafe close passes. Reciprocal collision avoidance formulations explicitly discuss oscillations in reciprocal settings and define velocity-space constraints for collision-free motion; in this repository context, pedestrians are simulated via Social Force modeling, which supports controlled single-ped archetypes and crowd effects depending on density configuration. citeturn40view0turn45view0turn39view1

## Prompt template for generating scenario sets

The prompt below is designed to produce scenario files compatible with the repository’s scenario-layout conventions (manifest + per-scenario entries) and with the SVG map-labeling requirements.

```text
You are generating a set of N minimalist 2D navigation scenarios for robot_sf_ll7 to test local navigation algorithms and system integration.

Context (must follow):
- Scenario YAML format follows configs/scenarios/README.md:
  - Either a manifest file with `includes:` and optional `map_search_paths:`, OR
  - A file containing `scenarios:` as a list of scenario entries.
  - Each scenario entry should use fields seen in configs/scenarios/single/*.yaml:
    `name`, `map_id` OR `map_file`, `simulation_config`, optional `single_pedestrians`, `robot_config`, `metadata`, `seeds`.
- Map creation follows docs/SVG_MAP_EDITOR.md (Inkscape labeling rules):
  - Obstacles: rectangles labeled exactly `obstacle`.
  - Robot: rectangles labeled `robot_spawn_zone(_i)`, `robot_goal_zone(_j)` and a path labeled `robot_route_<spawn>_<goal>`.
  - Single pedestrian (optional): circles labeled `single_ped_<id>_start` (+ optional `single_ped_<id>_goal`) and/or POIs (circle class `poi`, label = POI name).
- Prefer deterministic, isolatable tests: vary one factor at a time. Keep ped_density=0 unless the scenario explicitly tests crowd effects.
- Output must be valid YAML for the scenario file(s) plus a short “map-edit plan” per scenario.

Task:
1) Choose N (default N=20) scenarios. If user requests N=3, generate 3 only.
2) For each scenario, provide:
   - `name`: concise, stable identifier (e.g., nav_sanity_open_8dir_east).
   - `purpose`: one-sentence statement of the failure mode targeted.
   - Map reference:
       - If using existing maps, use `map_id` (preferred) or `map_file`.
       - If creating a new map, propose a new SVG filename under maps/svg_maps/ and describe exactly which labeled elements to draw.
   - `simulation_config`: set `max_episode_steps` and set `ped_density`. If using single pedestrians, define entries like:
       - id: h1
         goal_poi: poi_h1_goal   # or use a direct goal if map supports it
         speed_m_s: <optional>
         wait_at: <optional list of {waypoint_index, wait_s, note}>
   - `metadata`: include at minimum:
       - `archetype`: short category label (e.g., coord_sanity, single_obstacle, cornering, single_ped_head_on)
       - `target_failure_mode`: coordinate_transform | angle_wrap | oscillation | local_minima | deadlock | clearance_regression | etc.
       - `expected_pass_criteria`: structured text (goal reached, no collision, bounded oscillation, etc.)
   - `seeds`: three integers for reproducibility.
3) Return:
   A) One manifest YAML that includes all generated per-scenario YAML files (or a single YAML containing all `scenarios:` entries).
   B) A scenario matrix summarizing each scenario (name, what changes, what it tests).

Constraints:
- Keep scenarios “simple”: one robot, optionally one pedestrian, and at most one static obstacle primitive unless the scenario explicitly tests a compound geometry (e.g., U-shape).
- Prefer orthogonal increments (open → goal behind → single obstacle → line wall → corner → one pedestrian).
- Do not invent repository APIs; stay within documented YAML fields and SVG label semantics.
```

This prompt directly mirrors: (i) the scenario layout and map reference rules, including manifests with `includes`, (ii) the `map_id`/`map_file` precedence and registry resolution, and (iii) the SVG label contract for obstacles, spawn/goal zones, routes, POIs, and single pedestrians. citeturn44view0turn14view0turn12view0turn29view1

## Proposed scenario suite

The table below defines 20 “sanity-first” scenarios. The emphasis is on **diagnostic isolation**: each scenario should make one failure mode obvious from logs and trajectory video.

| Scenario (suggested `name`) | Minimal scene setup | Primary failure mode and assertions |
|---|---|---|
| `nav_open_axis_east` | Open map; start center; goal due +x | Frame/sign errors; assert straight motion, bounded lateral error |
| `nav_open_axis_west` | Open map; goal due −x | Mirroring errors; assert symmetry vs east run |
| `nav_open_axis_north` | Open map; goal due +y | Axis swap; assert correct 90° rotation equivalence |
| `nav_open_axis_south` | Open map; goal due −y | Y-sign mismatch; assert symmetry vs north |
| `nav_open_diag_ne` | Open map; goal +x,+y | Diagonal normalization; assert similar speed/turn profile vs axial |
| `nav_open_diag_nw` | Open map; goal −x,+y | Quadrant handling; assert correct heading selection |
| `nav_open_diag_sw` | Open map; goal −x,−y | Angle wrapping; assert heading continuity near −π…π |
| `nav_open_diag_se` | Open map; goal +x,−y | Quadrant handling; assert consistent with other diagonals |
| `nav_goal_behind_short` | Start oriented nominally; goal “behind” by short distance | Backward-vs-turn logic; assert no oscillatory “spin-in-place” |
| `nav_goal_behind_long` | Same but goal farther; same heading constraint | Angle-to-goal stability; assert no sign flips under longer horizon |
| `nav_start_near_boundary` | Start near map boundary line; goal inward | Boundary collision and planner clearance; assert no boundary grazing |
| `nav_goal_near_start` | Goal placed very close to start | Near-goal tolerance; assert stable stop/no chatter |
| `nav_single_obstacle_on_path` | One small obstacle centered on nominal straight line | Basic avoidance; assert detour direction consistent and collision-free |
| `nav_single_obstacle_near_start` | Obstacle just outside spawn zone | Startup robustness; assert no immediate collision or freeze |
| `nav_single_obstacle_near_goal` | Obstacle adjacent to goal zone | Goal acquisition; assert planner can approach and terminate |
| `nav_thin_wall_perpendicular` | Thin “line” wall across corridor; gap offset | Gap selection; assert correct passage and no oscillation at doorway |
| `nav_L_corner_turn` | L-shaped wall requiring ~90° turn | Cornering; assert smooth curvature and no corner-cut collision |
| `nav_U_trap_local_minimum` | U-shaped concavity with goal outside/inside variants | Local minima; assert escape behavior or detect/report failure |
| `nav_single_ped_head_on` | Open map + one single pedestrian approaching head-on | Reciprocal deadlock/over-caution; assert progress without unsafe pass |
| `nav_single_ped_obstruction_slow` | Pedestrian ahead moving slowly in same lane | Overtaking vs following; assert bounded following distance and no collision |

Several “one pedestrian” archetypes already exist in the repository’s `francis2023_*` single-scenario suite, including frontal approach, blind corner, and slow obstruction patterns (via `speed_m_s`). These files provide concrete templates for `single_pedestrians` definitions and metadata conventions that can be reused directly or mirrored in new maps. citeturn28view0turn29view0turn29view1

## Implementation notes for `robot_sf_ll7` scenario files and SVG maps

A practical implementation strategy is to create a **small family of SVG maps** that each contain exactly one spawn–goal pair (and route) so that the scenario semantics are deterministic without requiring additional “route selection” fields in the scenario YAML. The SVG editor guide explicitly allows multiple spawn/goal zones and routes, but using one pair per map keeps early sanity checks unambiguous. citeturn14view0turn11view0

When using manifests, prefer `map_id` for portability (resolved via the map registry) and reserve `map_file` for experimental maps not yet registered. The scenario README specifies that `map_id` is resolved through `maps/registry.yaml` and takes precedence over `map_file` if both are provided. citeturn44view0turn12view0

For single-pedestrian sweeps (“one pedestrian in all orientations”), the most stable approach is to encode orientation implicitly by geometry: place `single_ped_h1_start` and `poi_h1_goal` at positions that define the pedestrian’s intended motion direction relative to the robot’s nominal path. The scenario YAML examples confirm that a single pedestrian can be specified by `id` plus `goal_poi`, and can be slowed down with `speed_m_s` when needed for obstruction tests. citeturn28view0turn29view1

Obstacle primitives should be implemented as rectangles labeled `obstacle`, with “line obstacles” represented as **thin rectangles**. This conforms to the documented map contract, while also matching the underlying collision logic that checks circle–segment intersections (effectively boundary-like geometry). Additionally, the historical obstacle issue motivates explicit checks that spawn/goal zones are not placed inside obstacle interiors. citeturn14view0turn34view0turn30view0

After creating or editing maps, run the map verifier to catch structural or labeling errors early. The SVG editor documentation provides the canonical invocation. citeturn14view0

## References

- [robot_sf_ll7 repository root](https://github.com/ll7/robot_sf_ll7) — Author: entity["people","ll7","github user"]; Publication: entity["company","GitHub","code hosting platform"]; Publication Date: n.d.; Access Date: 2026-03-27.  
  <!-- Establishes project scope (Gymnasium integration, SocialForce/PySocialForce usage) and top-level documentation pointers. --> citeturn45view0

- [Scenario layout documentation](https://github.com/ll7/robot_sf_ll7/blob/main/configs/scenarios/README.md) — Author: entity["people","ll7","github user"]; Publication: entity["company","GitHub","code hosting platform"]; Publication Date: n.d.; Access Date: 2026-03-27.  
  <!-- Defines manifests/includes, map_id vs map_file rules, plausibility metadata block conventions. --> citeturn44view0

- [Planner sanity example scenario YAML](https://github.com/ll7/robot_sf_ll7/blob/main/configs/scenarios/single/planner_sanity_simple.yaml) — Author: entity["people","ll7","github user"]; Publication: entity["company","GitHub","code hosting platform"]; Publication Date: n.d.; Access Date: 2026-03-27.  
  <!-- Minimal example of the scenario YAML schema used for single scenarios. --> citeturn11view0

- [Frontal approach single-ped scenario YAML](https://github.com/ll7/robot_sf_ll7/blob/main/configs/scenarios/single/francis2023_frontal_approach.yaml) — Author: entity["people","ll7","github user"]; Publication: entity["company","GitHub","code hosting platform"]; Publication Date: n.d.; Access Date: 2026-03-27.  
  <!-- Concrete template for single_pedestrians with goal_poi and metadata fields. --> citeturn28view0

- [Pedestrian obstruction single-ped scenario YAML](https://github.com/ll7/robot_sf_ll7/blob/main/configs/scenarios/single/francis2023_pedestrian_obstruction.yaml) — Author: entity["people","ll7","github user"]; Publication: entity["company","GitHub","code hosting platform"]; Publication Date: n.d.; Access Date: 2026-03-27.  
  <!-- Shows speed override (speed_m_s) for controlled obstruction tests. --> citeturn29view1

- [Map registry](https://github.com/ll7/robot_sf_ll7/blob/main/maps/registry.yaml) — Author: entity["people","ll7","github user"]; Publication: entity["company","GitHub","code hosting platform"]; Publication Date: n.d.; Access Date: 2026-03-27.  
  <!-- Provides map_id → SVG path mapping for portable scenario configs. --> citeturn12view0

- [SVG maps directory](https://github.com/ll7/robot_sf_ll7/tree/main/maps/svg_maps) — Author: entity["people","ll7","github user"]; Publication: entity["company","GitHub","code hosting platform"]; Publication Date: n.d.; Access Date: 2026-03-27.  
  <!-- Enumerates available SVG maps (including planner_sanity_open.svg and classic interaction maps). --> citeturn13view0

- [SVG map editor contract](https://github.com/ll7/robot_sf_ll7/blob/main/docs/SVG_MAP_EDITOR.md) — Author: entity["people","ll7","github user"]; Publication: entity["company","GitHub","code hosting platform"]; Publication Date: n.d.; Access Date: 2026-03-27.  
  <!-- Defines the authoritative SVG labeling rules (obstacles, spawn/goal zones, routes, POIs, single pedestrians) and validation command. --> citeturn14view0

- [Obstacle collision issue](https://github.com/ll7/robot_sf_ll7/issues/55) — Author: entity["people","JuliusMiller","github user"]; Publication: entity["company","GitHub","code hosting platform"]; Publication Date: 2024-09-18 (issue opened); Access Date: 2026-03-27.  
  <!-- Motivates obstacle/spawn placement cautions and “wall-like” obstacle representations. --> citeturn30view0

- [Continuous collision checker implementation](https://github.com/ll7/robot_sf_ll7/blob/main/robot_sf/nav/occupancy.py) — Author: entity["people","ll7","github user"]; Publication: entity["company","GitHub","code hosting platform"]; Publication Date: n.d.; Access Date: 2026-03-27.  
  <!-- Confirms segment-based collision checks (not a raster occupancy grid), relevant for obstacle primitive design. --> citeturn34view0

- [Map config and single pedestrian parsing](https://github.com/ll7/robot_sf_ll7/blob/main/robot_sf/nav/map_config.py) — Author: entity["people","ll7","github user"]; Publication: entity["company","GitHub","code hosting platform"]; Publication Date: n.d.; Access Date: 2026-03-27.  
  <!-- Documents supported fields for single pedestrians (id, start, goal, trajectory, speed_m_s, wait rules). --> citeturn43view1

- [Test scenarios YAML (legacy/simple schema)](https://github.com/ll7/robot_sf_ll7/blob/main/test_scenarios/simple_test.yaml) — Author: entity["people","ll7","github user"]; Publication: entity["company","GitHub","code hosting platform"]; Publication Date: n.d.; Access Date: 2026-03-27.  
  <!-- Shows an alternate, coordinate-explicit test format that may inspire ultra-minimal “empty map” checks. --> citeturn19view0

- [The Dynamic Window Approach to Collision Avoidance (PDF)](https://www.ri.cmu.edu/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf) — Author: entity["people","Dieter Fox","robotics researcher"]; Publication: Carnegie Mellon University Robotics Institute (technical report PDF); Publication Date: 1997; Access Date: 2026-03-27.  
  <!-- Canonical local planner reference highlighting obstacle avoidance under kinematic constraints and discussing oscillations/local issues. --> citeturn38view0

- [Reciprocal n-body Collision Avoidance (ORCA) (PDF)](https://gamma.cs.unc.edu/ORCA/publications/ORCA.pdf) — Author: entity["people","Jur van den Berg","robotics researcher"]; Publication: University of North Carolina at Chapel Hill (PDF); Publication Date: n.d. (year not extracted from the PDF header); Access Date: 2026-03-27.  
  <!-- Formalizes reciprocal collision avoidance and notes oscillations in reciprocal settings, relevant to single-ped/head-on tests. --> citeturn40view0

- [Integrated online trajectory planning and optimization in distinctive topologies (TEB context) (PDF)](https://files.davidqiu.com/research/papers/2017_rosmann_TEB%20Planner%20Integrated%20online%20trajectory%20planning%20and%20optimization%20in%20distinctive%20topologies.pdf) — Author: entity["people","Christoph Rösmann","robotics researcher"]; Publication: *Robotics and Autonomous Systems* (Elsevier) (PDF copy); Publication Date: Available online 2016-11-12 (per PDF front matter); Access Date: 2026-03-27.  
  <!-- Motivates topology-sensitive scenario coverage and non-convex local optima that should be probed by U-shapes/corners/bottlenecks. --> citeturn41view0

- [Social Force Model for Pedestrian Dynamics (arXiv landing page)](https://arxiv.org/abs/cond-mat/9805244) — Author: entity["people","Dirk Helbing","physicist"]; Publication: entity["organization","arXiv","preprint repository"]; Publication Date: 1998-05-20 (submission date); Access Date: 2026-03-27.  
  <!-- Provides primary description of the Social Force concept used for pedestrian modeling lineage in the project. --> citeturn39view1

- [Social force model for pedestrian dynamics (PDF copy)](https://ics-websites.science.uu.nl/docs/vakken/mcrws/papers_new/Helbing_Molnar%20-%201995%20-%20Social%20force%20model%20for%20pedestrian%20dynamics.pdf) — Author: entity["people","Dirk Helbing","physicist"]; Publication: Utrecht University course materials (PDF mirror); Publication Date: 1995; Access Date: 2026-03-27.  
  <!-- Alternative accessible PDF form of the Social Force paper (mirror). --> citeturn39view0