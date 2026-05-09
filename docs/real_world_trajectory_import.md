# Real-World Trajectory Import

Issue: [#1091](https://github.com/ll7/robot_sf_ll7/issues/1091)

The first supported real-world trajectory import path is deliberately narrow:
Stanford Drone Dataset (SDD) annotations in the original text format. The official SDD project page
lists the dataset under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 license:
<https://cvgl.stanford.edu/projects/uav_data/>.

Robot SF does not redistribute SDD files. Download or stage the dataset separately, keep the license
terms with the staged data, and pass a local `annotations.txt` file to the importer.

## Command

```bash
rtk uv run python scripts/tools/import_sdd_scenarios.py \
  --annotations /path/to/stanford-drone-dataset/<scene>/<video>/annotations.txt \
  --out-dir output/imported_scenarios/sdd_bookstore_0_v1 \
  --dataset-id sdd_bookstore_0_v1 \
  --meters-per-pixel 0.0247 \
  --frame-rate-hz 30 \
  --min-track-points 12 \
  --max-pedestrians 4 \
  --stride 5 \
  --max-waypoints 24
```

The command writes:

* `<dataset-id>.map.yaml`: a Robot SF YAML map containing imported single-pedestrian trajectories.
* `<dataset-id>.scenario.yaml`: a benchmark scenario that references the generated map.
* `<dataset-id>.provenance.json`: dataset, license, normalization, and source-track metadata.

The generated scenario can be loaded like any other scenario config:

```bash
rtk uv run robot_sf_bench run \
  --matrix output/imported_scenarios/sdd_bookstore_0_v1/sdd_bookstore_0_v1.scenario.yaml \
  --out output/benchmarks/sdd_bookstore_0_v1/episodes.jsonl \
  --algo goal \
  --repeats 1 \
  --horizon 120 \
  --workers 1 \
  --no-video \
  --benchmark-profile baseline-safe \
  --no-resume \
  --fail-fast
```

## Normalization Contract

The importer reads SDD rows as:

```text
track_id xmin ymin xmax ymax frame lost occluded generated label
```

Only rows with `lost == 0` and the selected label, `Pedestrian` by default, are imported. Bounding
box centers are converted from pixels to local meters with `--meters-per-pixel`, shifted into a
positive local coordinate frame, and optionally Y-flipped with `--y-flip-height-px`.

The first point of each selected track becomes the pedestrian start position. Later sampled points
become the explicit Robot SF trajectory. The importer records source track ids, frame ranges,
normalization settings, and the SDD license in both scenario metadata and provenance JSON.

## Limits

This path is a reproducible importer and scenario-generation contract, not a claim that every SDD
scene is benchmark-ready. Imported scenarios still need scenario-level review for map context,
scale calibration, robot start/goal placement, and whether the resulting interaction is meaningful
for a given planner study.

The first version supports one annotation format and rectangular obstacle-free generated maps. A
future generalization should add scene-specific maps, calibrated homographies, and broader dataset
adapters only after the first SDD path has review evidence. The first real-data curation pass is
tracked in [#1126](https://github.com/ll7/robot_sf_ll7/issues/1126).
