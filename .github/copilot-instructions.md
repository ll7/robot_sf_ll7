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

- 243-clean-output-dirs: Added Python 3.11 (uv-managed virtual environment) + Python standard library (`pathlib`, `json`, `shutil`), Loguru, pytest, uv CLI

## Active Technologies

- Python 3.11 (uv-managed virtual environment) + Python standard library (`pathlib`, `json`, `shutil`), Loguru, pytest, uv CLI (243-clean-output-dirs)
- Local filesystem (repository-relative `output/` tree or overridden path) (243-clean-output-dirs)
