"""Sphinx configuration for the lightweight Robot SF documentation site."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

project = "Robot SF"
author = "Robot SF contributors"
copyright = "2026, Robot SF contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

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
suppress_warnings = [
    # The first site pass intentionally exposes a curated nav over a much larger Markdown corpus.
    "toc.not_included",
    # Existing repo-relative links often point outside docs/ or to GitHub-flavored anchors.
    "myst.xref_missing",
    # Some historical notes start below H1 or end on Markdown transitions.
    "myst.header",
    "docutils",
    # Historical docs include mermaid/jsonc fences and illustrative JSON with ellipses.
    "misc.highlighting_failure",
]

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
