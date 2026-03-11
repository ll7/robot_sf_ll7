"""Focused unit tests for map verification rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.maps.verification.context import VerificationStatus
from robot_sf.maps.verification.rules import (
    RuleSeverity,
    RuleViolation,
    ValidationRule,
    apply_all_rules,
    check_file_readable,
    check_file_size,
    check_required_layers,
    check_valid_svg,
    get_rule_by_id,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_svg(path: Path, *, labeled: bool = True) -> Path:
    label_attr = 'inkscape:label="obstacles"' if labeled else ""
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
     width="10" height="10">
  <g {label_attr}></g>
</svg>
""",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        (RuleSeverity.ERROR, VerificationStatus.FAIL),
        (RuleSeverity.WARNING, VerificationStatus.WARN),
        (RuleSeverity.INFO, VerificationStatus.PASS),
    ],
)
def test_rule_violation_status_mapping(
    severity: RuleSeverity, expected: VerificationStatus
) -> None:
    """Severity values should map to expected verification statuses."""
    violation = RuleViolation(rule_id="R", severity=severity, message="m", remediation="r")
    assert violation.status == expected


def test_validation_rule_apply_converts_exceptions_to_error_violation(tmp_path: Path) -> None:
    """ValidationRule.apply should convert check exceptions into deterministic violations."""

    def exploding_check(_path: Path):
        raise RuntimeError("boom")

    rule = ValidationRule(
        rule_id="X001",
        name="Explode",
        description="explodes",
        severity=RuleSeverity.ERROR,
        check_func=exploding_check,
    )

    violations = rule.apply(tmp_path / "dummy.svg")

    assert len(violations) == 1
    assert violations[0].rule_id == "X001"
    assert violations[0].severity == RuleSeverity.ERROR
    assert "Rule execution failed" in violations[0].message


def test_check_file_readable_covers_missing_and_directory(tmp_path: Path) -> None:
    """check_file_readable should fail for missing paths and directories."""
    missing = tmp_path / "missing.svg"
    missing_violations = check_file_readable(missing)
    assert len(missing_violations) == 1
    assert missing_violations[0].rule_id == "R001"

    directory_path = tmp_path / "as_directory"
    directory_path.mkdir()
    directory_violations = check_file_readable(directory_path)
    assert len(directory_violations) == 1
    assert "Path is not a file" in directory_violations[0].message


def test_check_valid_svg_parse_error_and_generic_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """check_valid_svg should handle XML parse errors and generic parse exceptions."""
    broken_svg = tmp_path / "broken.svg"
    broken_svg.write_text("<svg><g", encoding="utf-8")
    parse_violations = check_valid_svg(broken_svg)
    assert len(parse_violations) == 1
    assert parse_violations[0].rule_id == "R002"

    def raise_generic(_path: Path):
        raise OSError("io failure")

    monkeypatch.setattr("robot_sf.maps.verification.rules.ET.parse", raise_generic)
    generic_violations = check_valid_svg(tmp_path / "any.svg")
    assert len(generic_violations) == 1
    assert "Failed to parse SVG" in generic_violations[0].message


def test_check_file_size_warns_for_large_file_and_passes_small_file(tmp_path: Path) -> None:
    """check_file_size should only emit warning when file exceeds the 5MB limit."""
    small_svg = tmp_path / "small.svg"
    small_svg.write_text("<svg></svg>", encoding="utf-8")
    assert check_file_size(small_svg) == []

    large_svg = tmp_path / "large.svg"
    large_svg.write_bytes(b"x" * (6 * 1024 * 1024))
    large_violations = check_file_size(large_svg)
    assert len(large_violations) == 1
    assert large_violations[0].rule_id == "R003"
    assert large_violations[0].severity == RuleSeverity.WARNING


def test_check_required_layers_covers_warning_info_and_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """check_required_layers should emit warning/info as appropriate and tolerate parser failure."""
    unlabeled_svg = _write_svg(tmp_path / "unlabeled.svg", labeled=False)
    unlabeled_violations = check_required_layers(unlabeled_svg)
    assert any(v.rule_id == "R004" for v in unlabeled_violations)

    labeled_svg = _write_svg(tmp_path / "labeled.svg", labeled=True)
    labeled_violations = check_required_layers(labeled_svg)
    assert any(v.rule_id == "R005" for v in labeled_violations)

    def raise_generic(_path: Path):
        raise RuntimeError("cannot parse")

    monkeypatch.setattr("robot_sf.maps.verification.rules.ET.parse", raise_generic)
    assert check_required_layers(tmp_path / "whatever.svg") == []


def test_rule_registry_helpers_return_expected_entries(tmp_path: Path) -> None:
    """Rule lookup and apply-all behavior should remain stable."""
    assert get_rule_by_id("R001") is not None
    assert get_rule_by_id("unknown") is None

    valid_svg = _write_svg(tmp_path / "valid.svg", labeled=True)
    violations = apply_all_rules(valid_svg)
    assert isinstance(violations, list)
