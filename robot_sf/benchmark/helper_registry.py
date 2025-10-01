"""Helper registry data structures for reusable helper consolidation.

This module provides the core data structures and types used by the helper catalog
system to organize and document reusable helper functions extracted from examples
and scripts.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HelperCategory:
    """Groups related helper capabilities (e.g., environment_setup, recording, benchmarking).

    Attributes:
        key: Unique identifier in snake_case (e.g., 'environment_setup')
        description: Human-readable description of the category
        target_module: Python import path (e.g., 'robot_sf.benchmark.utils')
        default_owner: Maintainer or team alias
    """

    key: str
    description: str
    target_module: str
    default_owner: str


@dataclass
class HelperCapability:
    """Describes a reusable helper function or class.

    Attributes:
        name: Unique function/class name within target module
        category_key: Foreign key to HelperCategory
        summary: Single sentence docstring requirement
        inputs: List of parameters with type hints
        outputs: Return type / side effects description
        dependencies: List of external modules relied upon
        tests: Path to validating test/validation script
        docs_link: URL or relative doc path
    """

    name: str
    category_key: str
    summary: str
    inputs: list[str]
    outputs: str
    dependencies: list[str]
    tests: str
    docs_link: Optional[str] = None


@dataclass
class ExampleOrchestrator:
    """Represents an example or script that should act purely as an orchestrator.

    Attributes:
        path: Filesystem path under examples/ or scripts/
        owner: Maintainer alias
        requires_recording: Whether this orchestrator needs recording capabilities
        notes: Free-form text for deviations or special requirements
    """

    path: str
    owner: str
    requires_recording: bool
    notes: str = ""


@dataclass
class OrchestratorUsage:
    """Links orchestrators to the helpers they consume.

    Attributes:
        orchestrator_path: Path to the orchestrator file
        helper_name: Name of the helper function/class
        integration_notes: Configuration overrides or usage notes
    """

    orchestrator_path: str
    helper_name: str
    integration_notes: str = ""


@dataclass
class RegressionCheck:
    """Defines validation commands required to prove behavior parity.

    Attributes:
        command: String command to run for validation
        description: Human-readable description of what is being validated
        frequency: How often this should be run (e.g., pre-commit, nightly)
    """

    command: str
    description: str
    frequency: str = "pre-commit"


# Registry type definitions for convenience
HelperRegistry = dict[str, list[HelperCapability]]
CategoryRegistry = dict[str, HelperCategory]
OrchestratorRegistry = dict[str, ExampleOrchestrator]
UsageRegistry = list[OrchestratorUsage]
RegressionRegistry = list[RegressionCheck]
