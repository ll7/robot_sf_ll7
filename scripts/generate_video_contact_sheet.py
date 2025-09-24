"""Placeholder contact-sheet script for episode video artifacts (T020).

This script reserves the CLI/API surface for generating thumbnail contact sheets
from per-episode video artifacts. Implementation is deferred until the video
artifact pipeline stabilizes. The eventual flow will:
  1. Load episode JSONL records and locate associated MP4 paths.
  2. Sample representative frames (start/mid/end) per episode.
  3. Assemble a grid image with annotations (episode_id, scenario_id, seed).
  4. Write the gallery under the videos directory alongside perf snapshots.

Developers may extend `generate_contact_sheet` once performance budgets and
renderer fidelity targets are finalized.
"""

from __future__ import annotations

from pathlib import Path


def generate_contact_sheet(_episodes_jsonl: Path, _output_path: Path) -> None:  # pragma: no cover
    """Reserved implementation hook for future contact-sheet generation."""
    raise NotImplementedError("Contact-sheet generation is pending renderer fidelity improvements")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(
        "Use `generate_contact_sheet` from application code once implemented; "
        "no CLI entry point is available yet."
    )
