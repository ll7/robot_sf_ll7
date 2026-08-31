"""Tests for the validation-helper inventory tool (issue #7900)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dev.audit_validation_helpers import (
    MIN_CLUSTER_SIZE,
    SCHEMA,
    _cluster,
    _is_validation_helper,
    _scan_file_for_helpers_and_calls,
    run_inventory,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "audit_validation_helpers.py"

IDENTICAL_HELPER = """def validate_finite(value):
    if not isinstance(value, (int, float)):
        raise ValueError("not a number")
    return value
"""


def _write(root: Path, name: str, content: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_validation_helper_detection_by_name_and_body() -> None:
    assert _is_validation_helper("validate_finite", "x = 1")
    assert _is_validation_helper("parse_float", "return float(v)")
    assert _is_validation_helper("_check_mapping", "raise ValueError('bad')")
    assert not _is_validation_helper("compute_velocity", "return v * dt")


def test_scan_records_signature_and_features(tmp_path: Path) -> None:
    path = _write(tmp_path, "mod.py", IDENTICAL_HELPER)
    records, _ = _scan_file_for_helpers_and_calls(path, tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record.qualified_name == "mod.validate_finite"
    assert record.signature == "(value)"
    assert record.normalized_body_hash
    assert record.source_digest
    assert record.return_paths >= 1
    assert "ValueError" in record.raises
    assert record.features["none_policy"] == "unknown"


def test_identical_definitions_cluster(tmp_path: Path) -> None:
    for name in ("a.py", "b.py", "c.py"):
        _write(tmp_path, name, IDENTICAL_HELPER)
    records = []
    for path in sorted(tmp_path.glob("*.py")):
        records.extend(_scan_file_for_helpers_and_calls(path, tmp_path)[0])
    clusters = _cluster(records)
    assert len(clusters) == 1
    members = next(iter(clusters.values()))
    assert len(members) >= MIN_CLUSTER_SIZE


def test_different_bodies_do_not_cluster(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", IDENTICAL_HELPER)
    _write(tmp_path, "b.py", IDENTICAL_HELPER.replace('"not a number"', '"nope"'))
    records = []
    for path in sorted(tmp_path.glob("*.py")):
        records.extend(_scan_file_for_helpers_and_calls(path, tmp_path)[0])
    clusters = _cluster(records)
    # Two distinct normalized bodies -> two size-1 groups; neither is a
    # candidate cluster (requires >= MIN_CLUSTER_SIZE identical definitions).
    assert len(clusters) == 2
    assert all(len(group) < MIN_CLUSTER_SIZE for group in clusters.values())


def test_run_inventory_counts_call_sites(tmp_path: Path) -> None:
    _write(tmp_path, "a.py", IDENTICAL_HELPER)
    _write(tmp_path, "b.py", IDENTICAL_HELPER)
    _write(tmp_path, "c.py", IDENTICAL_HELPER)
    _write(
        tmp_path,
        "user.py",
        "from a import validate_finite\nx = validate_finite(1.0)\ny = validate_finite(2)\n",
    )
    report = run_inventory(tmp_path)
    assert report["schema"] == SCHEMA
    assert report["scan"]["helper_count"] == 3
    assert report["cluster_count"] == 1
    cluster_members = next(iter(report["candidate_clusters"].values()))
    by_name = {m["qualified_name"]: m for m in cluster_members}
    assert by_name["a.validate_finite"]["call_sites"] == 2
    assert by_name["b.validate_finite"]["call_sites"] == 0
    assert by_name["c.validate_finite"]["call_sites"] == 0


def test_exact_call_site_attribution_regression(tmp_path: Path) -> None:
    """A caller importing helper from a.py only attributes calls to a.py, not b.py."""
    _write(tmp_path, "a.py", IDENTICAL_HELPER)
    _write(tmp_path, "b.py", IDENTICAL_HELPER)
    _write(
        tmp_path,
        "user.py",
        "from a import validate_finite\nx = validate_finite(1.0)\ny = validate_finite(2.0)\n",
    )
    report = run_inventory(tmp_path)
    records, _ = _scan_file_for_helpers_and_calls(tmp_path / "a.py", tmp_path)
    assert len(records) == 1
    # Run full inventory and check all records by re-scanning or inspecting clusters
    _write(tmp_path, "c.py", IDENTICAL_HELPER)
    report = run_inventory(tmp_path)
    cluster = next(iter(report["candidate_clusters"].values()))
    lookup = {m["qualified_name"]: m["call_sites"] for m in cluster}
    assert lookup["a.validate_finite"] == 2
    assert lookup["b.validate_finite"] == 0
    assert lookup["c.validate_finite"] == 0


def test_import_alias_and_module_import_attribution(tmp_path: Path) -> None:
    """Aliased and module-attribute imports resolve to their exact target module."""
    _write(tmp_path, "a.py", IDENTICAL_HELPER)
    _write(tmp_path, "b.py", IDENTICAL_HELPER)
    _write(tmp_path, "c.py", IDENTICAL_HELPER)
    _write(
        tmp_path,
        "user_alias.py",
        "from a import validate_finite as my_check\nx = my_check(1.0)\n",
    )
    _write(
        tmp_path,
        "user_mod.py",
        "import b as mod_b\ny = mod_b.validate_finite(2.0)\n",
    )
    report = run_inventory(tmp_path)
    cluster = next(iter(report["candidate_clusters"].values()))
    lookup = {m["qualified_name"]: m["call_sites"] for m in cluster}
    assert lookup["a.validate_finite"] == 1
    assert lookup["b.validate_finite"] == 1
    assert lookup["c.validate_finite"] == 0


def test_same_module_calls_and_unresolved_method_calls(tmp_path: Path) -> None:
    """Same-module bare calls resolve locally while object methods remain unassigned."""
    same_mod = """def validate_finite(value):
    if not isinstance(value, (int, float)):
        raise ValueError("not a number")
    return value

def helper():
    validate_finite(10)
"""
    _write(tmp_path, "a.py", same_mod)
    _write(tmp_path, "b.py", IDENTICAL_HELPER)
    _write(tmp_path, "c.py", IDENTICAL_HELPER)
    _write(
        tmp_path,
        "user_obj.py",
        "class Obj:\n    def validate_finite(self, v):\n        return v\n\no = Obj()\no.validate_finite(99)\n",
    )
    report = run_inventory(tmp_path)
    cluster = next(iter(report["candidate_clusters"].values()))
    lookup = {m["qualified_name"]: m["call_sites"] for m in cluster}
    assert lookup["a.validate_finite"] == 1
    assert lookup["b.validate_finite"] == 0
    assert lookup["c.validate_finite"] == 0
    assert report["scan"]["unresolved_calls"] >= 1


def test_syntax_error_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path, "bad.py", "def broken(:\n")
    with pytest.raises((SyntaxError, RuntimeError)):
        run_inventory(tmp_path)


def test_repeated_runs_byte_stable(tmp_path: Path) -> None:
    _write(tmp_path, "m.py", IDENTICAL_HELPER)
    first = json.dumps(run_inventory(tmp_path), sort_keys=True)
    second = json.dumps(run_inventory(tmp_path), sort_keys=True)
    assert first == second


def test_cli_emits_deterministic_json(tmp_path: Path) -> None:
    _write(tmp_path, "m.py", IDENTICAL_HELPER)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    report = json.loads(proc.stdout)
    assert report["schema"] == SCHEMA
    assert report["scan"]["helper_count"] == 1


def test_cli_markdown_report(tmp_path: Path) -> None:
    for name in ("a.py", "b.py", "c.py"):
        _write(tmp_path, name, IDENTICAL_HELPER)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--markdown"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "Validation-helper inventory" in proc.stdout
    assert "candidate clusters: 1" in proc.stdout
