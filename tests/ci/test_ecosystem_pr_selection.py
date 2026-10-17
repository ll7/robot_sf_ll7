"""Real PR marker selection must execute the ecosystem source digest witness."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("corrupt", [False, True])
def test_ecosystem_digest_runs_in_pr_selection(tmp_path, corrupt):
    """Run the real witness and inject only a synthetic schema-source mismatch."""
    plugin = tmp_path / "digest_probe.py"
    plugin.write_text("""
def pytest_collection_modifyitems(items):
    from scripts.tools import build_robot_sf_ecosystem_contract as builder
    original = builder._resolve_public_input
    def mismatched(*args, **kwargs):
        resolved = original(*args, **kwargs)
        if 'schema' in resolved.public_record.get('input_id', ''):
            resolved.public_record['sha256'] = '0' * 64
        return resolved
    builder._resolve_public_input = mismatched
""")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/benchmark/test_ecosystem_contract.py",
        "-k",
        "committed_contract_is_canonical_digest_bound_and_source_current",
        "-m",
        "not slow",
        "-q",
        "-o",
        "addopts=",
    ]
    if corrupt:
        cmd += ["-p", "digest_probe"]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == (1 if corrupt else 0), result.stdout + result.stderr
    assert ("stale authoritative input digest" if corrupt else "1 passed") in result.stdout
