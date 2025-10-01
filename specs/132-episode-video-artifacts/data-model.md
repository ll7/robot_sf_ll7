# Data Model: Episode Video Artifacts

## Entities

### Episode Video Artifact
- episode_id: string
- path: string (relative or absolute)
- format: 'mp4'
- filesize_bytes: integer (>= 0)
- frames: integer (>= 0)
- renderer: enum ['synthetic','sim-view','none']
- notes: string (optional)

### Renderer Mode (enum)
- synthetic
- sim-view
- none

### Performance Sample (doc artifact)
- encode_ms_per_frame: number
- overhead_percent: number
- environment: { os, python, cpu/gpu }
- notes: string
