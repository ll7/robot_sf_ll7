"""Minimal-environment import guards for standalone docs-evidence tooling."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STANDALONE_DOCS_EVIDENCE_SCRIPTS = (REPO_ROOT / "scripts/dev/check_docs_evidence_integrity.py",)


@pytest.mark.parametrize("script", STANDALONE_DOCS_EVIDENCE_SCRIPTS)
def test_docs_evidence_script_imports_with_pyyaml_only(script: Path, tmp_path: Path) -> None:
    """Standalone CI scripts import without the repository package installed.

    ``-S`` removes site packages and the temporary ``yaml`` stub represents the
    docs-evidence CI job's only third-party dependency.  A ``robot_sf`` sentinel
    turns an accidental package import into a deterministic failure.
    """

    (tmp_path / "yaml.py").write_text("# minimal PyYAML import sentinel\n", encoding="utf-8")
    (tmp_path / "robot_sf.py").write_text(
        "raise RuntimeError('standalone docs-evidence script imported robot_sf')\n",
        encoding="utf-8",
    )
    environment = {**os.environ, "PYTHONPATH": str(tmp_path)}

    result = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "standalone docs-evidence script imported robot_sf" not in result.stderr
