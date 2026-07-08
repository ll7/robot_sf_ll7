# Issue #4761: Diagnostic Social Preference Labels

**Issue:** <https://github.com/ll7/robot_sf_ll7/issues/4761>

**Status:** Implemented

**Source:** SPLC paper (Social Preference Learning for Crowd Robot Navigation, arXiv:2607.01925).
This config does not implement SPLC and does not claim learned or validated social preferences.

## What This Is

A diagnostic label schema and annotation pipeline for post-run analysis of crowd-navigation
episode traces. The labels capture social-compliance dimensions (clearance, TTC, oscillation,
etc.) that are distinct from hard safety predicates and benchmark gates.

## What This Is Not

- **Not an RL reward.** These labels must not be used as planner rewards or training signals.
- **Not a benchmark gate.** These labels do not affect pass/fail verdicts in benchmark campaigns.
- **Not human preference data.** The threshold bands are placeholder defaults aligned to existing
  safety contracts, not calibrated human preference evidence.

## Files

| File | Purpose |
| --- | --- |
| `configs/diagnostics/social_preference_labels.yaml` | Label schema with 7 social preference labels, threshold bands, and claim boundary |
| `robot_sf/benchmark/social_preference_labels.py` | Config validation and ``annotate_episode_social_preferences`` annotation pipeline |
| `scripts/analysis/annotate_social_preference_labels.py` | CLI to annotate episode JSONL files with labels |
| `tests/benchmark/test_social_preference_labels_config.py` | Config contract tests (loaded from the original issue) |
| `tests/benchmark/test_social_preference_labels.py` | Annotation pipeline, CLI, and summary tests |

## Label Schema

The v1 schema defines seven labels:

| Label | Metric Family | Preferred Direction | Automation |
| --- | --- | --- | --- |
| ``clearance`` | safety_proxemics | maximize | threshold_band |
| ``ttc_margin`` | safety_ttc | maximize | threshold_band |
| ``pedestrian_displacement`` | pedestrian_impact | minimize | manual/proxy |
| ``path_blocking`` | path_corridor_interaction | avoid | not_available |
| ``oscillation`` | motion_quality | minimize | threshold_band |
| ``detour_burden`` | path_efficiency | minimize | manual/proxy |
| ``recovery_smoothness`` | post_conflict_motion_quality | minimize | threshold_band |

Each label can be annotated as:

- ``satisfied`` (good band) / ``violated`` (poor band) / ``caution`` when a threshold band matches
- ``not_available`` when required trace fields or candidate metrics are missing
- ``uncertain`` when no band matches definitively

## Usage

Annotate a single episode or batch of episodes from JSONL::

    uv run python scripts/analysis/annotate_social_preference_labels.py \
      --episodes-jsonl output/episodes.jsonl \
      --schema configs/diagnostics/social_preference_labels.yaml \
      --output-jsonl output/social_preference_annotations.jsonl \
      --summary-json output/social_preference_summary.json

## Hard Safety vs Preference Labels

Hard safety predicates (e.g. ``min_clearance_m < 0`` for collision) are fail-closed contract
checks used in benchmark rows and planner verification. Social preference labels are diagnostic
dimensions that live at a higher level: they describe the quality of the robot's social behavior
without determining pass or fail for the benchmark contract.

See ``robot_sf/benchmark/safety_predicates.py`` for the hard safety predicate definitions and
``robot_sf/benchmark/thresholds.py`` for the collision/near-miss distance contracts.

## Follow-Up

The path_blocking label is currently ``not_available`` because no canonical path-corridor occupancy
metric exists. Pedestrian displacement and detour burden are manual/proxy labels that require
scenario baseline comparisons. These would benefit from a trace export contract for pedestrian
route estimates and conflict-window attribution.