"""Gate-contract tests for the base_sensitive marker and stale-base gate.

These tests verify the infrastructure described in issue #5559:
- The base_sensitive marker exists and is discoverable.
- The gate script can identify base-sensitive test files.
- The base_sensitive subset runs and returns structured results.
- A simulated stale-base scenario is caught by the gate.

Scope: CPU-only, no benchmark or paper claims.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_SCRIPT = REPO_ROOT / "scripts" / "dev" / "check_base_sensitive_gates.py"


class TestBaseSensitiveMarker:
    """Verify the base_sensitive pytest marker is registered and functional."""

    def test_marker_registered_in_pyproject(self) -> None:
        """The marker must appear in pyproject.toml markers list."""
        pyproject = REPO_ROOT / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        assert "base_sensitive" in content, "base_sensitive marker not registered in pyproject.toml"

    def test_marker_can_select_tests(self) -> None:
        """pytest -m base_sensitive must select at least one test."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "base_sensitive",
                "--collect-only",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, f"pytest collect failed: {result.stderr}"
        # Output format: "N/M tests collected ..." or "N test(s) collected"
        last_line = result.stdout.strip().splitlines()[-1]
        assert "test" in last_line and "collected" in last_line, (
            f"Unexpected output format: {last_line}"
        )
        # Extract count: first number in "19/15928 tests collected" or "19 test(s) collected"
        first_part = last_line.split()[0]
        # Could be "19/15928" or just "19"
        if "/" in first_part:
            count = int(first_part.split("/")[0])
        else:
            count = int(first_part)
        assert count >= 3, (
            f"Expected at least 3 base_sensitive tests, got {count}. "
            "The marker is not tagging the identified base-sensitive test surfaces."
        )


class TestGateScript:
    """Verify the gate script can discover files and run the subset."""

    def test_script_list_files_mode(self) -> None:
        """The --list-files mode must return known base-sensitive files."""
        result = subprocess.run(
            [sys.executable, str(GATE_SCRIPT), "--list-files", "--json"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert data["count"] >= 3, f"Gate script found only {data['count']} base-sensitive files"
        files = data["base_sensitive_files"]
        # Must include the three canonical surfaces
        found = {Path(f) for f in files}
        expected_stems = {
            "test_optional_import_guard_inventory",
            "test_snqi_pareto_pinned_artifact",
            "test_runner_circuit_breaker",
        }
        actual_stems = {p.stem for p in found}
        missing = expected_stems - actual_stems
        assert not missing, f"Missing base-sensitive files: {missing}"

    def test_subset_run_under_two_minutes(self) -> None:
        """The base_sensitive subset must run in under 2 minutes."""
        start = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-m", "base_sensitive", "-q"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=120,
            check=False,
        )
        elapsed = time.monotonic() - start
        assert elapsed < 120, (
            f"base_sensitive subset took {elapsed:.1f}s (>2 min). The gate subset must be fast."
        )
        assert result.returncode == 0, (
            f"base_sensitive subset failed:\n{result.stdout}\n{result.stderr}"
        )


class TestSimulatedStaleBase:
    """Simulate a stale-base merge race scenario and verify gate catches it.

    This test demonstrates that two individually green PRs can break main
    when combined, using the optional-import guard inventory ratchet as
    the mechanism (since adding a new optional-import guard spelling
    without updating the fixture would fail in combined-base).
    """

    def test_fixture_staleness_is_caught_by_subset(self, tmp_path: Path) -> None:
        """Tampering with the fixture should cause the subset to fail.

        Simulates: PR A updates a file in robot_sf/ adding a new import
        guard, and PR B independently runs. When combined, the fixture
        no-longer matches.
        """

        import json as stdlib_json

        fixture_path = REPO_ROOT / "tests" / "fixtures" / "optional_import_guards.json"
        original = fixture_path.read_text(encoding="utf-8")
        fixture_data = stdlib_json.loads(original)

        # Tamper: lower one ceiling so the real inventory assertion must fail.
        tampered = dict(fixture_data)
        spellings = dict(tampered.get("spellings", {}))
        assert spellings, "The optional-import fixture must contain spellings."
        first_key = next(iter(spellings))
        tampered_spellings = {**spellings}
        entry = {**tampered_spellings[first_key]}
        entry["count_ceiling"] = 0
        tampered_spellings[first_key] = entry
        tampered["spellings"] = tampered_spellings

        tampered_path = tmp_path / "optional_import_guards.json"
        tampered_path.write_text(stdlib_json.dumps(tampered, indent=2), encoding="utf-8")

        original_data = stdlib_json.loads(original)
        original_ceiling = original_data["spellings"][first_key]["count_ceiling"]
        tampered_ceiling = tampered["spellings"][first_key]["count_ceiling"]
        assert original_ceiling != tampered_ceiling

        # Load the real inventory test and point its fixture at the tampered copy.
        # This executes the production assertion without mutating tracked files
        # or paying for the whole base-sensitive subset.
        probe = """
import importlib.util
import sys
from pathlib import Path

module_path = Path(sys.argv[1])
fixture_path = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("inventory_under_test", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load inventory test module")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.FIXTURE = fixture_path
module.TestOptionalImportGuardInventory().test_no_new_unblessed_spelling_and_no_count_growth()
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                probe,
                str(
                    Path(__file__).resolve().parents[1] / "test_optional_import_guard_inventory.py"
                ),
                str(tampered_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=30,
            check=False,
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0, (
            "Inventory assertion should fail against the tampered fixture"
        )
        assert "RATCHET VIOLATION" in output, output[:1000]

    def test_gate_catches_new_unblessed_spelling(self, tmp_path: Path) -> None:
        """Adding a new optional-import guard must fail the gate.

        Previously this test wrote a temp file into ``robot_sf/`` so the
        scanner would find it, then cleaned it up in a ``finally`` block.
        Under xdist the outer inventory worker could observe that file while
        the nested base-sensitive subprocess was running, causing a spurious
        RATCHET VIOLATION for the outer scan.  See issue #5722.

        The fix: write the unblessed guard into a dedicated subdirectory
        inside ``tmp_path`` (which is process-local and invisible to other
        workers) and pass ``OPTIONAL_IMPORT_SCAN_ROOT`` to the subprocess so
        the inventory scanner targets that isolated tree instead of
        ``robot_sf/``.  No tracked file is touched; cleanup is automatic when
        pytest removes ``tmp_path``.
        """
        # Create a minimal scan tree in tmp_path — a single Python file with
        # an unblessed optional-import guard spelling.  Use a novel combination
        # (ImportError+ZeroDivisionError) that is not in the committed fixture
        # so the "NEW unblessed" check triggers even when scanning only this
        # isolated tree (a bare ImportError IS already blessed and would not
        # exceed the ceiling for a single file).  See issue #5722.
        scan_root = tmp_path / "scan_root"
        scan_root.mkdir()
        guard_code = '"""Module to test gate detection of new import guards."""\n\n'
        guard_code += "from __future__ import annotations\n\n"
        guard_code += "try:\n"
        guard_code += "    import non_existent_module_for_gate_test\n"
        guard_code += "except (ImportError, ZeroDivisionError):\n"
        guard_code += "    non_existent_module_for_gate_test = None\n"
        (scan_root / "gate_spelling_check_dynamic.py").write_text(guard_code, encoding="utf-8")

        # The base_sensitive subset should now fail because the scan root
        # contains an unblessed spelling not present in the fixture.
        import os

        env = {**os.environ, "OPTIONAL_IMPORT_SCAN_ROOT": str(scan_root)}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "base_sensitive",
                "-q",
                "tests/test_optional_import_guard_inventory.py",
                "--no-header",
                "-k",
                "test_no_new_unblessed_spelling_and_no_count_growth",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
            check=False,
            env=env,
        )
        assert result.returncode != 0, (
            "base_sensitive subset should fail when a new "
            "unblessed optional-import spelling is added. "
            "This is the green-alone-red-together failure mode "
            "the gate is designed to catch."
        )
        assert "NEW unblessed" in result.stdout or "RATCHET" in result.stdout, (
            f"Expected ratchet failure message, got: {result.stdout[:500]}"
        )


class TestXdistIsolationRegression:
    """Regression proof for issue #5722: dynamic fixtures are process-local.

    These tests verify that the fix is durable:
    - No file is written to ``robot_sf/`` during gate contract testing.
    - The ``OPTIONAL_IMPORT_SCAN_ROOT`` env-var correctly isolates the
      inventory scanner so concurrent xdist workers cannot see the temp
      fixture.
    """

    def test_no_temp_file_written_to_robot_sf(self, tmp_path: Path) -> None:
        """gate_spelling_check_dynamic.py must never appear in robot_sf/.

        Regression proof: the old implementation wrote the temp file to
        ``robot_sf/test_gate_spelling_check_dynamic.py``, which was visible
        to concurrent xdist workers running the inventory scan.  After the
        fix, that path must not exist during or after the gate test runs.
        """
        dynamic_path = REPO_ROOT / "robot_sf" / "test_gate_spelling_check_dynamic.py"
        assert not dynamic_path.exists(), (
            f"Regression: {dynamic_path.relative_to(REPO_ROOT)} exists "
            "inside robot_sf/. The gate contract test must use tmp_path for "
            "dynamic fixtures so they are invisible to concurrent xdist scans "
            "(issue #5722)."
        )

    def test_optional_import_scan_root_env_var_isolates_scan(self, tmp_path: Path) -> None:
        """OPTIONAL_IMPORT_SCAN_ROOT must redirect the inventory scanner.

        Creates a minimal scan tree in tmp_path, sets the env var, and
        asserts the subprocess reports the unblessed spelling from that tree
        (not from robot_sf/).  This verifies the env-var isolation mechanism
        end-to-end without touching any tracked file.
        """
        import os

        # Build a minimal scan tree with one unblessed guard.
        scan_root = tmp_path / "isolated_scan"
        scan_root.mkdir()
        guard_code = (
            '"""Isolated scan tree for xdist regression proof."""\n\n'
            "try:\n"
            "    import non_existent_regression_module\n"
            "except (ImportError, ZeroDivisionError):\n"
            "    non_existent_regression_module = None\n"
        )
        (scan_root / "regression_probe.py").write_text(guard_code, encoding="utf-8")

        env = {**os.environ, "OPTIONAL_IMPORT_SCAN_ROOT": str(scan_root)}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "base_sensitive",
                "-q",
                "tests/test_optional_import_guard_inventory.py",
                "--no-header",
                "-k",
                "test_no_new_unblessed_spelling_and_no_count_growth",
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
            check=False,
            env=env,
        )
        # The subprocess should fail and report the unblessed spelling.
        assert result.returncode != 0, (
            "Subprocess should fail: the isolated scan tree contains an "
            "unblessed 'ImportError' guard not present in the committed fixture."
        )
        assert "NEW unblessed" in result.stdout, (
            f"Expected 'NEW unblessed' in subprocess output.\n"
            f"stdout: {result.stdout[:600]}\n"
            f"stderr: {result.stderr[:200]}"
        )
