"""Validation rule definitions for map verification.

This module implements the rule engine for validating SVG maps.
Rules check geometric consistency, metadata completeness, and
structural integrity.

Rule Categories
---------------
- Geometry: Closed polygons, non-intersecting walls, layer ordering
- Metadata: Required fields, spawn points, goal regions
- Instantiation: Runtime environment creation compatibility
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from loguru import logger

from robot_sf.maps.verification.context import VerificationStatus


class RuleSeverity(str, Enum):
    """Rule violation severity levels."""

    ERROR = "error"  # Must fix - causes FAIL
    WARNING = "warning"  # Should fix - causes WARN
    INFO = "info"  # Informational only


@dataclass
class RuleViolation:
    """Represents a single rule violation."""

    rule_id: str
    severity: RuleSeverity
    message: str
    remediation: str

    @property
    def status(self) -> VerificationStatus:
        """Map severity to verification status."""
        if self.severity == RuleSeverity.ERROR:
            return VerificationStatus.FAIL
        elif self.severity == RuleSeverity.WARNING:
            return VerificationStatus.WARN
        else:
            return VerificationStatus.PASS


@dataclass
class ValidationRule:
    """Definition of a validation rule."""

    rule_id: str
    name: str
    description: str
    severity: RuleSeverity
    check_func: Callable[[Path], list[RuleViolation]]

    def apply(self, map_path: Path) -> list[RuleViolation]:
        """Apply this rule to a map.

        Parameters
        ----------
        map_path : Path
            Path to the SVG map file

        Returns
        -------
        list[RuleViolation]
            List of violations found (empty if passes)
        """
        try:
            return self.check_func(map_path)
        except Exception as e:  # noqa: BLE001 - broad catch converts arbitrary rule errors into violations
            logger.error(f"Rule {self.rule_id} failed: {e}")
            return [
                RuleViolation(
                    rule_id=self.rule_id,
                    severity=RuleSeverity.ERROR,
                    message=f"Rule execution failed: {e}",
                    remediation="Check map file structure and rule implementation",
                )
            ]


# Rule check functions


def check_file_readable(map_path: Path) -> list[RuleViolation]:
    """Check that the SVG file is readable."""
    violations = []

    if not map_path.exists():
        violations.append(
            RuleViolation(
                rule_id="R001",
                severity=RuleSeverity.ERROR,
                message=f"Map file not found: {map_path}",
                remediation=f"Ensure file exists at {map_path}",
            )
        )
    elif not map_path.is_file():
        violations.append(
            RuleViolation(
                rule_id="R001",
                severity=RuleSeverity.ERROR,
                message=f"Path is not a file: {map_path}",
                remediation="Provide path to SVG file, not directory",
            )
        )

    return violations


def check_valid_svg(map_path: Path) -> list[RuleViolation]:
    """Check that the file is valid XML/SVG."""
    import xml.etree.ElementTree as ET

    violations = []

    try:
        ET.parse(map_path)
    except ET.ParseError as e:
        violations.append(
            RuleViolation(
                rule_id="R002",
                severity=RuleSeverity.ERROR,
                message=f"Invalid SVG/XML: {e}",
                remediation="Fix XML syntax errors; validate with Inkscape or XML linter",
            )
        )
    except Exception as e:  # noqa: BLE001 - broad catch to prevent verification crash on unexpected parse issues
        violations.append(
            RuleViolation(
                rule_id="R002",
                severity=RuleSeverity.ERROR,
                message=f"Failed to parse SVG: {e}",
                remediation="Check file encoding and XML structure",
            )
        )

    return violations


def check_file_size(map_path: Path) -> list[RuleViolation]:
    """Check that file size is within reasonable limits."""
    violations = []

    MAX_SIZE_MB = 5
    file_size_mb = map_path.stat().st_size / (1024 * 1024)

    if file_size_mb > MAX_SIZE_MB:
        violations.append(
            RuleViolation(
                rule_id="R003",
                severity=RuleSeverity.WARNING,
                message=f"Large SVG file: {file_size_mb:.1f} MB (limit: {MAX_SIZE_MB} MB)",
                remediation="Optimize SVG, remove unused layers, or simplify geometry",
            )
        )

    return violations


def check_required_layers(map_path: Path) -> list[RuleViolation]:
    """Inspect SVG for labeled layers and provide guidance.

    Enhancement:
    - Adds layer statistics (labeled vs total groups).
    - Emits INFO rule R005 when labels exist to surface improvement hints.
    - Warns (R004) only when there are zero labeled groups.
    """
    import xml.etree.ElementTree as ET

    violations: list[RuleViolation] = []

    try:
        tree = ET.parse(map_path)
        root = tree.getroot()

        ns = {"inkscape": "http://www.inkscape.org/namespaces/inkscape"}
        # Use attribute wildcard search to avoid default-namespace tag matching issues
        labeled_groups = root.findall(".//*[@inkscape:label]", ns)
        all_groups = root.findall(".//*")

        labeled_count = len(labeled_groups)
        total_groups = len(all_groups)

        if labeled_count == 0:
            violations.append(
                RuleViolation(
                    rule_id="R004",
                    severity=RuleSeverity.WARNING,
                    message="No labeled layers found (missing Inkscape 'label' attributes)",
                    remediation="Open the SVG in Inkscape and assign descriptive layer labels (e.g. obstacles, spawns, waypoints) to <g> elements.",
                )
            )
        else:
            violations.append(
                RuleViolation(
                    rule_id="R005",
                    severity=RuleSeverity.INFO,
                    message=f"Layer stats: {labeled_count} labeled / {total_groups} total <g> groups",
                    remediation="Ensure critical semantics (obstacles, spawns, waypoints) have clear labels; split overly large generic groups.",
                )
            )
    except Exception as e:  # noqa: BLE001 - layer inspection errors are non-critical
        logger.debug(f"Could not check layers: {e}")

    return violations


# Rule registry

VALIDATION_RULES: list[ValidationRule] = [
    ValidationRule(
        rule_id="R001",
        name="File Readable",
        description="Map file must exist and be readable",
        severity=RuleSeverity.ERROR,
        check_func=check_file_readable,
    ),
    ValidationRule(
        rule_id="R002",
        name="Valid SVG",
        description="Map must be valid XML/SVG",
        severity=RuleSeverity.ERROR,
        check_func=check_valid_svg,
    ),
    ValidationRule(
        rule_id="R003",
        name="File Size",
        description="Map file size should be under 5 MB",
        severity=RuleSeverity.WARNING,
        check_func=check_file_size,
    ),
    ValidationRule(
        rule_id="R004",
        name="Required Layers",
        description="Warn when no Inkscape-labeled layer groups (<g @inkscape:label>) are present",
        severity=RuleSeverity.WARNING,
        check_func=check_required_layers,
    ),
]


def get_rule_by_id(rule_id: str) -> ValidationRule | None:
    """Get a rule by its ID."""
    for rule in VALIDATION_RULES:
        if rule.rule_id == rule_id:
            return rule
    return None


def apply_all_rules(map_path: Path) -> list[RuleViolation]:
    """Apply all validation rules to a map.

    Parameters
    ----------
    map_path : Path
        Path to the SVG map file

    Returns
    -------
    list[RuleViolation]
        All violations found across all rules
    """
    all_violations = []

    for rule in VALIDATION_RULES:
        violations = rule.apply(map_path)
        all_violations.extend(violations)

    return all_violations
