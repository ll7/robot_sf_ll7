# Review telemetry and annotation overlays (SREV-12)

Command-line glossary: SREV is the scenario-review portfolio; a review overlay specification
generates transparent RGBA overlay frame sequences (`overlay-frames.v1`), layer availability
reports (`layer-availability.v1`), and timestamp mapping receipts (`overlay-mapping-receipt.v1`);
a component request (`component-request.v1`) is the envelope that invokes one scenario-review
workbench component; shared contracts live in `robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.render.review_overlays` generates transparent overlay frames and mapping
receipts from simulation timeline and scenario data:
- **Simulation time as telemetry authority**: synchronized overlays require an explicit
  presentation-timestamp map; guessing frame/fps alignment is strictly forbidden.
- **Fail-closed layer availability**: if presentation-timestamp mapping is missing, synchronized
  layers are disabled with stable diagnostic reason code `missing_presentation_timestamp_map`
  (never assume frame/fps alignment).
- **Synchronized telemetry layers**:
  - `paths`: past and planned actor trajectories in world coordinates (meters).
  - `controls`: diagnostic control readout panel (velocity, angular speed, action, goal) with units.
  - `clearance`: clearance distance around the robot in meters.
  - `markers`: scenario event and checkpoint markers at world coordinates.
  - `labels`: actor and status labels positioned with an overlap-prevention layout to ensure legibility.
- **Pause and speed handling**: pauses freeze simulation time and repeat source telemetry state;
  speed changes resample by declared policy (`nearest`, `linear`, `hold`).
- **Telemetry gap and actor lifecycle tracking**: dropped telemetry frames and actor disappearances
  (despawn, exit sensor range) are tracked and disclosed in the mapping receipt.
- **Pixel-metric freedom**: all spatial quantities remain strictly in world units (meters, seconds,
  radians). No metrics are derived from screen pixels.
- **Offline rendering**: transparent RGBA frames are generated locally without external renderer
  or cloud dependencies.

## Command

```bash
uv run python -m robot_sf.render.review_overlays \
  --input tests/fixtures/scenario_review/review_overlays/request.json \
  --config tests/fixtures/scenario_review/review_overlays/config.json \
  --output output/scenario_review/srev-12-smoke
```

Inspect capability descriptor:

```bash
uv run python -m robot_sf.render.review_overlays --descriptor
```

## Output

- `mapping_receipt.json` (`overlay-mapping-receipt.v1`): complete timestamp and layer mapping receipt:
  - `presentation`: dimensions (`width`, `height`), `fps`, `speed`.
  - `layers`: per-layer availability and enablement status.
  - `frames`: per-frame presentation and source timestamps, pause flag, telemetry status, actors present, and artifact SHA-256 digests.
  - `dropped_telemetry`: explicit log of frames where telemetry data was missing or dropped.
  - `actor_lifecycle`: actor appearance and disappearance events.
  - `provenance`: component version, output directory, and admission status.
- `layer_availability.json` (`layer-availability.v1`): summary of available and disabled layers.
- `visualization_spec.json` (`visualization-spec.v1`): compatible visualization spec document.
- `frames/frame_XXXX.png`: sequence of transparent RGBA overlay image frames.

## Unavailable reasons and limits

- If presentation-timestamp mapping is omitted or invalid, synchronized layers are disabled with
  reason `missing_presentation_timestamp_map: explicit presentation-timestamp map is required; guessing frame/fps alignment is forbidden`,
  and the component reports `partial` status.
- If unsupported capabilities are requested, the component reports `unavailable` with
  `unsupported required capabilities: <list>`.
- Non-finite dimensions, coordinates, or bounds fail closed (`failed` status) with descriptive reasons.
- Existing output directories are rejected (`output collision`) to protect prior runs and durable data.
- Evidence boundary: overlay rendering provides diagnostic visualization tooling only. No benchmark
  admission, planner change, or scientific claim follows from overlay generation.
