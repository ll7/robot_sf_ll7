"""Benchmark module for robot social navigation evaluation.

This module provides tools for running benchmarks, collecting metrics,
and analyzing robot navigation performance in social environments.
"""

from robot_sf.benchmark.errors import AggregationMetadataError
from robot_sf.benchmark.helper_catalog import (
    load_trained_policy,
    prepare_classic_env,
    run_episodes_with_recording,
)
from robot_sf.benchmark.helper_registry import (
    ExampleOrchestrator,
    HelperCapability,
    HelperCategory,
    OrchestratorUsage,
    RegressionCheck,
)

__all__ = [
    "AggregationMetadataError",
    "ExampleOrchestrator",
    "HelperCapability",
    "HelperCategory",
    "OrchestratorUsage",
    "RegressionCheck",
    "load_trained_policy",
    "prepare_classic_env",
    "run_episodes_with_recording",
]
