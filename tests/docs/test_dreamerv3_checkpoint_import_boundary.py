"""Tests for the DreamerV3 checkpoint import boundary note."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTE = ROOT / "docs" / "context" / "issue_1190_dreamerv3_checkpoint_import_boundary.md"
DOCS_README = ROOT / "docs" / "README.md"
CONTEXT_README = ROOT / "docs" / "context" / "README.md"
DESIGN_NOTE = ROOT / "docs" / "context" / "issue_782_dreamerv3_pretraining_design.md"


def test_dreamerv3_checkpoint_import_boundary_records_fail_closed_decision() -> None:
    """The #1190 note should preserve the no-go import-boundary evidence."""
    text = NOTE.read_text(encoding="utf-8")

    assert "Fail closed" in text
    assert "Do not run the imported-vs-scratch BR-08 gate comparison" in text
    assert "Ray/RLlib version: `2.53.0`" in text
    assert "DreamerV3.from_checkpoint" in text
    assert "DreamerV3.get_checkpointable_components" in text
    assert "`learner_group`" in text
    assert "`dreamer_model.world_model`" in text
    assert "requires RLlib-specific checkpoint surgery" in text
    assert "does not contain a retained source DreamerV3 checkpoint" in text
    assert "benchmark_socnav_grid_br08_gate.yaml" in text
    assert "output/dreamerv3/" in text


def test_dreamerv3_checkpoint_import_boundary_is_linked() -> None:
    """Normal docs entry points should expose the #1190 no-go note."""
    note_name = "issue_1190_dreamerv3_checkpoint_import_boundary.md"

    assert note_name in DOCS_README.read_text(encoding="utf-8")
    assert note_name in CONTEXT_README.read_text(encoding="utf-8")
    assert note_name in DESIGN_NOTE.read_text(encoding="utf-8")
