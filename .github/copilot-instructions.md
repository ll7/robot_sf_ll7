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
- 001-performance-tracking: Added Python 3.11 (uv-managed virtual environment) + `robot_sf` core modules, Loguru logging, `psutil` for CPU/memory metrics, optional NVIDIA/NVML bindings, optional TensorBoard event writer, YAML/JSON helpers already present in repo
- 001-ppo-imitation-pretrain: Added Python 3.11 (uv-managed virtual environment) + Stable-Baselines3, imitation (HumanCompatibleAI), Gymnasium, Loguru, NumPy, PyTorch

- 243-clean-output-dirs: Added Python 3.11 (uv-managed virtual environment) + Python standard library (`pathlib`, `json`, `shutil`), Loguru, pytest, uv CLI

## Active Technologies
- Python 3.11 (uv-managed virtual environment) + `robot_sf` core modules, Loguru logging, `psutil` for CPU/memory metrics, optional NVIDIA/NVML bindings, optional TensorBoard event writer, YAML/JSON helpers already present in repo (001-performance-tracking)
- Structured JSON/Markdown manifests in `output/` (respecting `ROBOT_SF_ARTIFACT_ROOT`), optional TensorBoard log directory (001-performance-tracking)

