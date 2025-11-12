"""Helpers for loading and validating the examples manifest.

This package exposes data structures and utilities that power documentation
rendering and automated smoke tests for the ``examples/`` directory.
"""

from .manifest_loader import (
    ExampleCategory,
    ExampleManifest,
    ExampleScript,
    ManifestValidationError,
    load_manifest,
)

__all__ = [
    "ExampleCategory",
    "ExampleManifest",
    "ExampleScript",
    "ManifestValidationError",
    "load_manifest",
]
