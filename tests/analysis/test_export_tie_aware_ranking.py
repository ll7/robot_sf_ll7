"""Tests for the tie-aware ranking exporter CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.analysis.export_tie_aware_ranking import main

if TYPE_CHECKING:
    from pathlib import Path


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    """A JSON input produces both deterministic output surfaces."""
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "out" / "ranking.json"
    summary_path = tmp_path / "out" / "ranking.md"
    input_path.write_text(
        json.dumps(
            {
                "metric": {"id": "loss", "higher_is_better": False},
                "display_order": ["b", "a"],
                "rows": [
                    {"key": "a", "score": 1, "support": {"n": 2, "N": 2}},
                    {"key": "b", "score": 2, "support": {"n": 2, "N": 2}},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                str(input_path),
                "--output",
                str(output_path),
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "tie_aware_ranking.v1"
    assert [item["key"] for item in payload["items"]] == ["b", "a"]
    assert "# Tie-aware ranking summary" in summary_path.read_text(encoding="utf-8")
