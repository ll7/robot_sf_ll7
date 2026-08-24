"""Contract checks for the current full benchmark release instructions."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE_DOC = ROOT / "docs" / "RELEASE.md"
FULL_MANIFEST = "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml"
SMOKE_MANIFEST = (
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml"
)
SMOKE_RECEIPT = "output/benchmarks/camera_ready/<smoke_id>/release/release_result.json"


def _full_release_command(text: str) -> str:
    """Return the fenced command that launches the current full release."""
    section = text.split("## Release Execution", maxsplit=1)[1]
    for command in re.findall(r"```bash\n(.*?)```", section, flags=re.DOTALL):
        if FULL_MANIFEST in command:
            return command
    raise AssertionError("full benchmark release command is missing")


def _runtime_smoke_run_command(text: str) -> str:
    """Return the fenced run-mode command that produces release_result.json."""
    section = text.split("## Runtime-Smoke Run Mode", maxsplit=1)[1]
    for command in re.findall(r"```bash\n(.*?)```", section, flags=re.DOTALL):
        if SMOKE_MANIFEST in command:
            return command
    raise AssertionError("runtime-smoke run-mode command is missing")


def test_full_v02_release_requires_exact_source_runtime_smoke_receipt() -> None:
    """The documented full launch must pass the fresh 14-arm smoke result."""
    text = RELEASE_DOC.read_text(encoding="utf-8")
    command = _full_release_command(text)

    assert "canonical 14-arm runtime smoke at the exact release source commit" in text
    assert "--runtime-smoke-receipt" in command
    assert SMOKE_RECEIPT in command


def test_runtime_smoke_run_command_is_documented() -> None:
    """The exact run-mode command must be documented with required arguments."""
    text = RELEASE_DOC.read_text(encoding="utf-8")
    command = _runtime_smoke_run_command(text)

    assert "--mode run" in command
    assert "--manifest" in command and SMOKE_MANIFEST in command
    assert "--campaign-id" in command
    assert "--checkpoint-receipt" in command
    assert "release_result.json" in text
    # The smoke staging command is documented in the same section.
    assert "preflight_campaign_checkpoints.py" in text
    staging_index = text.index("preflight_campaign_checkpoints.py")
    run_mode_index = text.index("--mode run")
    assert staging_index < run_mode_index
