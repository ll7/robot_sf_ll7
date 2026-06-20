"""Focused tests for the pedestrian archetype reporting packet builder."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "build_pedestrian_archetype_report.py"
)
_SPEC = importlib.util.spec_from_file_location("build_pedestrian_archetype_report", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_REPORT_BUILDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REPORT_BUILDER)


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("", "non-empty example_compositions"),
        ("- cautious\n- standard\n", "mapping at top level"),
        ("plain string\n", "mapping at top level"),
    ],
)
def test_load_example_compositions_rejects_empty_or_non_mapping_yaml(
    tmp_path: pathlib.Path, contents: str, match: str
) -> None:
    """Empty and non-mapping registry files should fail with clear ValueErrors."""
    config = tmp_path / "pedestrian_archetypes.yaml"
    config.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        _REPORT_BUILDER._load_example_compositions(config)


def test_format_markdown_uses_dated_issue_heading() -> None:
    """Generated evidence READMEs should follow the issue-note heading convention."""
    markdown = _REPORT_BUILDER._format_markdown(
        {
            "status": "composition_report_only",
            "evidence_date": "2026-06-20",
            "config_path": "configs/research/pedestrian_archetypes_v1.yaml",
            "population_size": 30,
            "claim_boundary": "No benchmark claim.",
            "reports": {
                "homogeneous_standard": {
                    "archetypes": {"standard": {"realized_count": 30}},
                    "speed_factor_min": 1.0,
                    "speed_factor_max": 1.0,
                    "assignment_order_sha1": "d445fb25af0e",
                }
            },
        }
    )

    assert markdown.startswith("# Issue #3206 Pedestrian Archetype Reporting Packet (2026-06-20)")
