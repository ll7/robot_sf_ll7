"""Contract tests for type checking API."""

import subprocess


class TestTypeCheckingAPI:
    """Contract tests for the type checking API endpoints."""

    def test_run_type_check_endpoint_contract(self):
        """Test the run_type_check endpoint contract.

        Contract: run_type_check command should execute successfully
        and return diagnostic information.
        """
        # Execute the type check command
        result = subprocess.run(
            ["uvx", "ty", "check", ".", "--exit-zero"],
            capture_output=True,
            text=True,
            cwd=".",
            check=False,
        )

        # Contract: Command should execute without error
        assert result.returncode == 0, f"Type check failed: {result.stderr}"

        # Contract: Accept either the old diagnostic summary or the
        # newer success message. If the checker prints "All checks passed!"
        # treat that as zero diagnostics.
        if "All checks passed!" in result.stdout:
            # No diagnostics reported; count is zero
            count = 0
        else:
            # Older output contains a line like: "Found <n> diagnostics"
            assert "Found" in result.stdout, "Expected 'Found' in output"
            assert "diagnostics" in result.stdout, "Expected 'diagnostics' in output"

            # Contract: Should be parseable as containing a number
            lines = result.stdout.strip().split("\n")
            found_line = next((line for line in lines if "Found" in line), None)
            assert found_line is not None, "No 'Found' line in output"

            # Extract diagnostic count (should be >= 0)
            parts = found_line.split()
            count_str = next((part for part in parts if part.isdigit()), None)
            assert count_str is not None, f"Could not extract count from: {found_line}"
            count = int(count_str)

        assert count >= 0, f"Invalid diagnostic count: {count}"
