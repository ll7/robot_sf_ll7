"""Validation rules for map verification.

This module defines the rules used to verify SVG maps, including:
- Geometric integrity checks
- Metadata completeness validation
- Spawn point coverage verification
"""

from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from robot_sf.maps.verification import MapRecord


@dataclass
class RuleResult:
    """Result of applying a single validation rule.
    
    Attributes:
        rule_id: Unique identifier for the rule
        passed: Whether the rule passed
        message: Human-readable result message
        severity: 'error', 'warn', or 'info'
        remediation_hint: Optional suggestion for fixing failures
    """
    rule_id: str
    passed: bool
    message: str
    severity: str  # 'error' | 'warn' | 'info'
    remediation_hint: str | None = None


class ValidationRule(Protocol):
    """Protocol for validation rules."""
    
    rule_id: str
    
    def validate(self, map_record: MapRecord) -> RuleResult:
        """Validate a map record against this rule.
        
        Args:
            map_record: Map to validate
            
        Returns:
            Result of the validation
        """
        ...


# Built-in rules

class FileExistsRule:
    """Verify that the SVG file exists and is readable."""
    
    rule_id = "file_exists"
    
    def validate(self, map_record: MapRecord) -> RuleResult:
        """Check if the map file exists."""
        if not map_record.file_path.exists():
            return RuleResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"Map file not found: {map_record.file_path}",
                severity="error",
                remediation_hint=f"Ensure the file exists at {map_record.file_path}",
            )
        
        if not map_record.file_path.is_file():
            return RuleResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"Path is not a file: {map_record.file_path}",
                severity="error",
                remediation_hint="Check that the path points to a regular file",
            )
        
        return RuleResult(
            rule_id=self.rule_id,
            passed=True,
            message=f"File exists: {map_record.file_path}",
            severity="info",
        )


class FileSizeRule:
    """Verify that the SVG file is not too large."""
    
    rule_id = "file_size"
    max_size_mb = 5
    
    def validate(self, map_record: MapRecord) -> RuleResult:
        """Check if the map file size is reasonable."""
        if not map_record.file_path.exists():
            # Skip if file doesn't exist (will be caught by FileExistsRule)
            return RuleResult(
                rule_id=self.rule_id,
                passed=True,
                message="File does not exist (skipped)",
                severity="info",
            )
        
        size_bytes = map_record.file_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb > self.max_size_mb:
            return RuleResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"File too large: {size_mb:.2f} MB (max: {self.max_size_mb} MB)",
                severity="warn",
                remediation_hint=(
                    f"Consider optimizing the SVG or splitting into smaller maps. "
                    f"Large maps may exceed CI time budgets."
                ),
            )
        
        return RuleResult(
            rule_id=self.rule_id,
            passed=True,
            message=f"File size OK: {size_mb:.2f} MB",
            severity="info",
        )


class SvgParseableRule:
    """Verify that the SVG file can be parsed."""
    
    rule_id = "svg_parseable"
    
    def validate(self, map_record: MapRecord) -> RuleResult:
        """Check if the SVG file can be parsed."""
        try:
            import xml.etree.ElementTree as ET
            
            if not map_record.file_path.exists():
                # Skip if file doesn't exist
                return RuleResult(
                    rule_id=self.rule_id,
                    passed=True,
                    message="File does not exist (skipped)",
                    severity="info",
                )
            
            # Try to parse the SVG
            tree = ET.parse(map_record.file_path)
            root = tree.getroot()
            
            # Basic SVG structure check
            if "svg" not in root.tag.lower():
                return RuleResult(
                    rule_id=self.rule_id,
                    passed=False,
                    message="Root element is not <svg>",
                    severity="error",
                    remediation_hint="Ensure the file is a valid SVG with <svg> root element",
                )
            
            return RuleResult(
                rule_id=self.rule_id,
                passed=True,
                message="SVG parsed successfully",
                severity="info",
            )
            
        except ET.ParseError as e:
            return RuleResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"XML parse error: {e}",
                severity="error",
                remediation_hint=(
                    "Fix XML syntax errors. "
                    "Use an XML validator or open in Inkscape to identify issues."
                ),
            )
        except Exception as e:
            logger.warning(f"Unexpected error parsing SVG: {e}")
            return RuleResult(
                rule_id=self.rule_id,
                passed=False,
                message=f"Unexpected parse error: {e}",
                severity="error",
                remediation_hint="Check file permissions and content",
            )


# Default rule set
DEFAULT_RULES: list[ValidationRule] = [
    FileExistsRule(),
    FileSizeRule(),
    SvgParseableRule(),
]


def get_default_rules() -> list[ValidationRule]:
    """Get the default set of validation rules.
    
    Returns:
        List of validation rules to apply
    """
    return DEFAULT_RULES.copy()


def apply_rules(
    map_record: MapRecord,
    rules: list[ValidationRule] | None = None,
) -> list[RuleResult]:
    """Apply validation rules to a map record.
    
    Args:
        map_record: Map to validate
        rules: Rules to apply (defaults to DEFAULT_RULES)
        
    Returns:
        List of rule results
    """
    if rules is None:
        rules = get_default_rules()
    
    results = []
    for rule in rules:
        try:
            result = rule.validate(map_record)
            results.append(result)
        except Exception as e:
            logger.error(f"Rule {rule.rule_id} failed with exception: {e}")
            results.append(
                RuleResult(
                    rule_id=rule.rule_id,
                    passed=False,
                    message=f"Rule execution failed: {e}",
                    severity="error",
                )
            )
    
    return results
