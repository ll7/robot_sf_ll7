"""Sphinx configuration for the lightweight Robot SF documentation site."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

project = "Robot SF"
author = "Robot SF contributors"
copyright = "2026, Robot SF contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
]

# Only test explicit doctest/testcode directives in documentation, not unmanaged docstring examples.
doctest_test_doctest_blocks = ""

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
# The curated builder sends the exact complement through a file to avoid exec's
# argument-size limit. Replace defaults, matching the old -D override semantics.
if exclusions_file := os.environ.get("ROBOT_SF_SPHINX_EXCLUSIONS_FILE"):
    exclude_patterns = json.loads(Path(exclusions_file).read_text(encoding="utf-8"))

# Warning-class suppressions are intentionally absent. The canonical documentation
# build is the curated strict build (scripts/dev/sphinx_strict_build.sh): it builds
# only the docs/index.rst toctree closure, promotes warnings to errors, and allows
# only cross-references to existing repository documents that the curated site does
# not build. A full-tree build is unsupported and noisy by design.

html_theme = "sphinx_rtd_theme"
html_title = "Robot SF Documentation"
html_short_title = "Robot SF"
html_show_sourcelink = True

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# Keep docs import-time light-weight when optional extras are missing.
autodoc_mock_imports = [
    "pandas",
    "stable_baselines3",
    "tensorboard",
    "torch",
]

myst_heading_anchors = 3
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
]
myst_substitutions = {
    "repo_root": str(ROOT),
}

linkcheck_anchors = False
linkcheck_ignore = [
    r"https://github\.com/.*",
]
