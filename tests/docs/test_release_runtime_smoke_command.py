"""Contract checks for the current full benchmark release instructions."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE_DOC = ROOT / "docs" / "RELEASE.md"
FULL_MANIFEST = "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml"
SMOKE_RECEIPT = "output/benchmarks/camera_ready/<smoke_id>/release/release_result.json"


def _full_release_command(text: str) -> str:
    """Return the fenced command that launches the current full release."""
    section = text.split("## Release Execution", maxsplit=1)[1]
    for command in re.findall(r"```bash\n(.*?)```", section, flags=re.DOTALL):
        if FULL_MANIFEST in command:
            return command
    raise AssertionError("full benchmark release command is missing")


def test_full_v02_release_requires_exact_source_runtime_smoke_receipt() -> None:
    """The documented full launch must pass the fresh 14-arm smoke result."""
    text = RELEASE_DOC.read_text(encoding="utf-8")
    command = _full_release_command(text)

    assert "canonical 14-arm runtime smoke at the exact release source commit" in text
    assert "--runtime-smoke-receipt" in command
    assert SMOKE_RECEIPT in command
