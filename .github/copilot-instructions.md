# Copilot Instructions

ALWAYS use the official [dev_guide](../docs/dev_guide.md) as the primary reference for development-related tasks.
It is everyones guide on how to use this repository effectively.

## Additional Instructions

- Use scriptable interfaces instead of cli interfaces when possible.
- Make everything reproducible.
- Central point to link new documentation pages is `docs/README.md`.
  - Link new documentation (sub-)pages in the appropriate section.
- For any changes that affect users, update the `CHANGELOG.md` file.
- Source the environment before using python or uv `source .venv/bin/activate`.

## Recent Changes
- 270-imitation-report: Added Python 3.11 (existing uv-managed environment)
- 001-map-verification: Added Python 3.11 (uv-managed virtual environment) + `robot_sf.gym_env` factories + unified configs, Loguru logging, SVG parsing utilities already present in `robot_sf.maps`, optional geometry helpers (Shapely)
- 001-performance-tracking: Added Python 3.11 (uv-managed virtual environment) + `robot_sf` core modules, Loguru logging, `psutil` for CPU/memory metrics, optional NVIDIA/NVML bindings, optional TensorBoard event writer, YAML/JSON helpers already present in repo


## Active Technologies
- Python 3.11 (existing uv-managed environment) (270-imitation-report)
- File-based (JSONL episode records, JSON manifests, YAML configs, NPZ trajectories, ZIP model checkpoints) (270-imitation-report)

