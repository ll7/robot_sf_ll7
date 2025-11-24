"""Custom exceptions for the research reporting module.

This module defines exception classes specific to research report generation,
providing clear error messages and context for debugging failures in the
automated reporting pipeline.

Exception Hierarchy:
    ResearchError (base)
    ├── ReportGenerationError (general report generation failures)
    ├── ValidationError (data/schema validation failures)
    ├── AggregationError (metric aggregation failures)
    ├── StatisticalTestError (statistical analysis failures)
    └── FigureGenerationError (plotting/rendering failures)
"""

from __future__ import annotations


class ResearchError(Exception):
    """Base exception for all research module errors."""


class ReportGenerationError(ResearchError):
    """Raised when report generation fails.

    Examples:
        - Template rendering fails
        - Output directory creation fails
        - Required data missing for report sections
    """


class ValidationError(ResearchError):
    """Raised when data or schema validation fails.

    Examples:
        - JSON schema validation fails
        - Required fields missing from metadata
        - Data types don't match expected schema
        - File integrity checks fail
    """


class AggregationError(ResearchError):
    """Raised when metric aggregation fails.

    Examples:
        - No valid seeds found for aggregation
        - Bootstrap sampling fails
        - Incompatible metric shapes across seeds
    """


class StatisticalTestError(ResearchError):
    """Raised when statistical analysis fails.

    Examples:
        - Insufficient samples for t-test (< 2 paired observations)
        - Invalid p-value computation
        - Effect size calculation fails
    """


class FigureGenerationError(ResearchError):
    """Raised when figure generation fails.

    Examples:
        - Matplotlib backend not available
        - Invalid data for plotting
        - File save operation fails
        - PDF export not supported
    """
