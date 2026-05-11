# Issue 1130 Negative Route-Clearance Repair

Issue: [#1130](https://github.com/ll7/robot_sf_ll7/issues/1130)

## Goal

Repair the underlying route geometry for the negative-clearance scenarios identified during the
route-clearance audit, without changing planner behavior or benchmark metrics.

## Changes

Two SVG maps were adjusted:

* `maps/svg_maps/classic_merging.svg`
  * rerouted `robot_route_0_0` through the open Y-merge corridor instead of across the central
    stem obstacle,
  * rerouted `ped_route_0_0` through the same obstacle-free merge corridor because SVG inspection
    found it also crossed the central stem.
* `maps/svg_maps/classic_station_platform.svg`
  * rerouted `robot_route_0_0` through the clear passage left of the elevator core instead of
    through the elevator obstacle.

The edits keep the same spawn/goal zones and preserve the intended merging and station-platform
interaction layouts. They change only route centerlines that previously crossed obstacle interiors.

## Evidence

`svg_inspect.py --strict error` reports no findings for both edited maps, and direct clearance
checks show nonnegative robot footprint margins:

* `classic_merging.svg`: robot route minimum center distance `1.563773 m`, margin `0.563773 m`
* `classic_station_platform.svg`: robot route minimum center distance `3.5 m`, margin `2.5 m`

The map regression test `tests/maps/test_route_clearance_maps.py` locks both conditions:

* no robot or pedestrian route crosses obstacle interiors for the edited maps,
* affected robot routes keep a nonnegative footprint margin.

## Interpretation Boundary

This repair addresses only the three negative route/obstacle overlaps from #1105/#1130. Other
paper-matrix route-clearance warnings with zero or low positive margins remain stress-geometry
caveats rather than map defects.
