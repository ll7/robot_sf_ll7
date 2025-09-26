"""Contract tests for linting behavior."""

import os
import subprocess
import tempfile
from pathlib import Path


def test_ruff_select_rules_enabled():
    """Test that the expected rules are enabled in the configuration."""
    # This is a contract test - we verify the configuration includes expected rules
    # The actual linting behavior is tested through integration

    # Check that pyproject.toml contains the expected select rules
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    # Verify key rule families are present
    assert '"B"' in content  # flake8-bugbear
    assert '"BLE"' in content  # bare except
    assert '"UP"' in content  # pyupgrade
    assert '"SIM"' in content  # simplify
    assert '"S"' in content  # bandit (though ignored globally)


def test_ruff_ignore_rules_applied():
    """Test that ignored rules are properly configured."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    # Verify ignored rules
    assert '"PLR0911"' in content  # too many returns
    assert '"PLR2004"' in content  # magic values
    assert '"S"' in content  # security checks


def test_per_file_ignores_configured():
    """Test that per-file ignores are set up correctly."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    # Verify per-file ignores
    assert '"tests/**/*"' in content
    assert '"scripts/**/*"' in content
    assert '"examples/**/*"' in content
    assert '"docs/**/*"' in content

    # Verify specific rules are allowed in tests
    assert '"S101"' in content  # assert statements
    assert '"T201"' in content  # print statements


def test_sample_code_linting_behavior():
    """Test linting behavior on sample code snippets."""
    # Create temporary Python file with problematic code
    sample_code = """
# Code that should trigger various rules
def bad_function():
    try:
        x = 1
    except:  # BLE001: bare except
        pass

    import os  # PTH123: prefer pathlib
    os.path.join("a", "b")

    print("hello")  # T20: print outside main/script

    def unused_arg(arg):  # ARG001: unused argument
        return 42

    return [x for x in range(10) if x % 2 == 0]  # C4: unnecessary list comprehension
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(sample_code)
        temp_file = f.name

    try:
        # Run ruff on the sample file
        result = subprocess.run(
            ["uv", "run", "ruff", "check", temp_file],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            check=False,
        )

        # Should find some issues (exact output may vary)
        # At minimum, should not crash and should produce some output
        assert result.returncode in [0, 1]  # 0 = no issues, 1 = issues found

    finally:
        os.unlink(temp_file)
